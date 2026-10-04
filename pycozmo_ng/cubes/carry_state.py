"""Carry state — does the robot have a cube on its lift? Checked on demand.

The lift (forklift) carries a grabbed cube *in front of and above* the
camera, so a cube that rides the forks leaves the camera's field of view.
That gives a simple post-grab visibility heuristic:

* right after a grab the fork is still rising — nothing is judged until
  the lift settles again (``LIFT_IN_POS`` status flag) or its height
  reads up;
* fork up + grabbed cube still visible -> it lies on the floor, the grab
  missed -> ``NOT_CARRYING`` — and the verdict **resolves**: seeing the
  cube is final until the next grab, so driving away afterwards (cube
  leaves the frame) cannot flip it back to ``CARRYING``;
* fork up + cube gone from the world map -> it rides the forks, out of
  frame -> ``CARRYING``;
* fork down -> nothing can ride a lowered fork -> ``NOT_CARRYING``.

Everything else (e.g. lift up but no witnessed grab) is ``UNKNOWN``.
Nothing polls in the background — :attr:`CarryDetector.state` is
evaluated whenever a caller asks for it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.logger import logger

__all__ = ["CarryDetector", "CarryState"]


class CarryState(Enum):
    """What the robot currently knows about the cube on its lift."""

    #: Cannot tell — e.g. the lift is up but no grab was witnessed.
    UNKNOWN = "unknown"
    #: A grabbed cube is out of frame with the forks up: it is on the lift.
    CARRYING = "carrying"
    #: No cube on the lift (forks down, or the cube is still in view).
    NOT_CARRYING = "not_carrying"


class CarryDetector:
    """Answers "am I carrying a cube?" whenever asked.

    Args:
        get_cube_state: ``Callable[[int], CubeState | None]`` — world-map
            lookup by cube *index* (0-2). ``updated_at`` is the monotonic
            timestamp of the last frame that saw the cube, i.e. how fresh
            its visibility is.
        get_lift_height: Current lift height in mm (``0.0`` = forks down).
            Callable so the detector never holds a ``Client``.
        get_lift_in_pos: ``Callable[[], bool]`` — the robot's ``LIFT_IN_POS``
            status flag: ``False`` while the lift motor is travelling,
            ``True`` when it has settled. Optional; when omitted the lift
            is assumed settled, so only the height reading decides.
        clock: Monotonic clock (injectable for tests).

    The heuristic runs in two phases, per the "wait for the fork" rule:
    after :meth:`note_grab` the detector waits for the lift to finish
    rising (observed motion + settle, or height reading up) before it
    judges anything — a mid-travel or stale height reading is never a
    verdict. Once the fork is up, visibility alone decides: cube still
    in frame = grab missed, cube gone for at least ``LOST_S`` =
    riding the forks. Seeing the cube resolves the observation: the
    ``NOT_CARRYING`` verdict latches, so the check *ends* — driving away
    (cube leaves the frame) cannot flip it back to ``CARRYING``. Only a
    fresh :meth:`note_grab` re-arms the check. A fork that is settled
    *down* and never rose after the grab is authoritative
    ``NOT_CARRYING`` and forgets the observation (a lowered lift means
    the cube was placed).

    Limitation (inherent to visibility-based detection): a missed grab
    whose cube rolled out of view *without ever being seen* after the
    grab reads as ``CARRYING``; a cube that re-enters view while
    ``CARRYING`` resolves the other way (visible = not on the forks).
    Re-check after ``clear()`` + a fresh grab to resynchronise.
    """

    #: Lift height above which a cube could actually ride the forks.
    #: ``MIN_LIFT_HEIGHT`` (fully down) is 32 mm, ``MAX`` (fully up) 92 mm.
    MIN_CARRY_LIFT_MM = 60.0
    #: Detections run at ~30 Hz (the approach's freshness check reads
    #: age 0.0s), so only this long without a fused update means the cube
    #: is out of the frame. Deliberately short: the last sighting is
    #: typically the grab itself, and a long window would count that
    #: stale pre-grab sighting as "still in frame" right after the forks
    #: rise — a false NOT_CARRYING that latches (live log: fork up at
    #: grab+1.3s, sighting age 1.3s, resolved wrongly).
    LOST_S = 0.3
    #: ``set_lift_height`` returns as soon as the command is sent, and the
    #: firmware acknowledges before the motor starts. If no lift motion
    #: has been seen this long after a grab, the rise never happened —
    #: the fork is genuinely down.
    LIFT_TRANSIT_S = 2.0

    def __init__(
        self,
        get_cube_state: Callable[[int], Any] | None = None,
        get_lift_height: Callable[[], float] | None = None,
        get_lift_in_pos: Callable[[], bool] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._get_cube_state = get_cube_state
        self._get_lift_height = get_lift_height
        self._get_lift_in_pos = get_lift_in_pos
        self._clock = clock
        self._lock = threading.Lock()
        self._cube_index: int | None = None
        self._grabbed_at = 0.0
        #: Lift motion seen since ``note_grab`` (the commanded rise).
        self._saw_lift_motion = False
        #: The cube was seen after the grab: the check ended — the
        #: NOT_CARRYING verdict latches until the next ``note_grab``.
        self._resolved = False
        #: Last state written to the log, to log transitions only.
        self._last_logged: CarryState | None = None

    # ------------------------------------------------------------------
    # Observation hooks (called by CubeApproach)
    # ------------------------------------------------------------------

    def note_grab(self, cube_index: int) -> None:
        """Arm the observation after a grab: lift up, cube about to leave frame."""
        with self._lock:
            self._cube_index = int(cube_index)
            self._grabbed_at = self._clock()
            self._saw_lift_motion = False
            self._resolved = False
            self._last_logged = None
        logger.info(
            "Carry: observing cube %d (waiting for the fork to finish rising)",
            to_cube_id(cube_index),
        )

    def clear(self) -> None:
        """Forget the observation — forks lowered / cube placed."""
        with self._lock:
            index = self._cube_index
            self._cube_index = None
            self._resolved = False
            self._last_logged = None
        if index is not None:
            logger.info("Carry: cleared (cube %d)", to_cube_id(index))

    # ------------------------------------------------------------------
    # On-demand status
    # ------------------------------------------------------------------

    @property
    def state(self) -> CarryState:
        """Current carry state — evaluated now, nothing runs in background."""
        now = self._clock()
        with self._lock:
            index = self._cube_index
            grabbed_at = self._grabbed_at

        lift = self._lift_height_mm()
        in_pos = self._lift_in_pos()

        if index is None:
            state, detail = self._unarmed_state(lift, in_pos)
        else:
            state, detail = self._armed_state(index, grabbed_at, now, lift, in_pos)
        self._log_transition(state, detail)
        return state

    @property
    def is_carrying(self) -> bool | None:
        """``True`` / ``False``, or ``None`` when the state is unknown."""
        state = self.state
        if state is CarryState.CARRYING:
            return True
        if state is CarryState.NOT_CARRYING:
            return False
        return None

    @property
    def on_forks(self) -> bool:
        """Is a cube believed to ride the raised forks right now?

        ``True`` while the observation says so: state ``CARRYING``, or an
        armed grab whose verdict is still pending (the fork is still
        rising right after the grab). ``False`` when unarmed (nothing
        grabbed, or the observation was cleared) and when the check
        resolved the grab as missed (``NOT_CARRYING``). This is the
        "may I stack?" predicate for :class:`CubeStack`.
        """
        with self._lock:
            armed = self._cube_index is not None
        if not armed:
            return False
        return self.state is not CarryState.NOT_CARRYING

    @property
    def cube_id(self) -> int | None:
        """Cube ID (1-3) being observed, or ``None`` when not armed."""
        with self._lock:
            return None if self._cube_index is None else to_cube_id(self._cube_index)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _unarmed_state(self, lift: float, in_pos: bool) -> tuple[CarryState, str]:
        """No observation in progress: fork position alone is the answer."""
        if lift >= self.MIN_CARRY_LIFT_MM:
            return (
                CarryState.UNKNOWN,
                f"forks up but no grab witnessed (lift {lift:.1f}mm)",
            )
        if not in_pos:
            return CarryState.UNKNOWN, f"lift in motion (lift {lift:.1f}mm)"
        return CarryState.NOT_CARRYING, f"forks down (lift {lift:.1f}mm)"

    def _armed_state(
        self,
        index: int,
        grabbed_at: float,
        now: float,
        lift: float,
        in_pos: bool,
    ) -> tuple[CarryState, str]:
        """A grab is being observed: wait for the fork, then judge visibility."""
        cube_id = to_cube_id(index)

        if not in_pos:
            # The commanded rise is travelling — wait, never judge.
            with self._lock:
                self._saw_lift_motion = True
            return (
                CarryState.UNKNOWN,
                f"lift in motion after grabbing cube {cube_id} (lift {lift:.1f}mm)",
            )

        with self._lock:
            saw_motion = self._saw_lift_motion

        if saw_motion or lift >= self.MIN_CARRY_LIFT_MM:
            # The fork finished the commanded rise (or reads up): now,
            # and only now, visibility decides.
            age = self._seconds_since_seen(index, now)
            if age < self.LOST_S:
                with self._lock:
                    newly_resolved = not self._resolved
                    self._resolved = True
                if newly_resolved:
                    logger.info(
                        "Carry: cube %d seen after the grab - resolved, not carrying until the next grab",
                        cube_id,
                    )
                return (
                    CarryState.NOT_CARRYING,
                    (f"cube {cube_id} still in frame {age:.1f}s after the grab (lift {lift:.1f}mm)"),
                )
            with self._lock:
                resolved = self._resolved
            if resolved:
                # The check already ended when the cube was seen: losing
                # sight of it now (the robot drove away) must not
                # resurrect a carry verdict.
                return (
                    CarryState.NOT_CARRYING,
                    (
                        f"cube {cube_id} was seen after the grab and has since"
                        f" left view - left on the floor (lift {lift:.1f}mm)"
                    ),
                )
            return (
                CarryState.CARRYING,
                (f"cube {cube_id} out of frame for {age:.1f}s with forks up (lift {lift:.1f}mm)"),
            )

        if now - grabbed_at < self.LIFT_TRANSIT_S:
            # Command sent, motor not moving yet — wait for the rise.
            return (
                CarryState.UNKNOWN,
                f"waiting for the fork to rise (lift {lift:.1f}mm)",
            )

        # Settled low with no motion since the grab: the rise never
        # happened. Forks down is authoritative — not carrying, and the
        # cube was placed, so forget the observation.
        self.clear()
        return (
            CarryState.NOT_CARRYING,
            (f"forks still down {now - grabbed_at:.1f}s after the grab (lift {lift:.1f}mm), observation cleared"),
        )

    def _log_transition(self, state: CarryState, detail: str) -> None:
        """Log only when the reported state changes (reads run at ~30 Hz)."""
        with self._lock:
            if state is self._last_logged:
                return
            self._last_logged = state
        logger.info("Carry: %s - %s", state.value.replace("_", " "), detail)

    def _lift_height_mm(self) -> float:
        if self._get_lift_height is None:
            return 0.0
        try:
            return float(self._get_lift_height())
        except Exception:  # noqa: BLE001 — a broken read must not break reads
            logger.exception("Carry: lift height read failed")
            return 0.0

    def _lift_in_pos(self) -> bool:
        if self._get_lift_in_pos is None:
            return True
        try:
            return bool(self._get_lift_in_pos())
        except Exception:  # noqa: BLE001 — a broken read must not break reads
            logger.exception("Carry: lift in-position read failed")
            return True

    def _seconds_since_seen(self, index: int, now: float) -> float:
        if self._get_cube_state is None:
            return float("inf")
        try:
            state = self._get_cube_state(index)
        except Exception:  # noqa: BLE001 — a broken read must not break reads
            logger.exception("Carry: cube state read failed (index %d)", index)
            return float("inf")
        if state is None:
            # Never seen / forgotten — treat as out of frame.
            return float("inf")
        return now - float(state.updated_at)
