"""Carry state — does the robot have a cube on its lift? Checked on demand.

The lift (forklift) carries a grabbed cube *in front of and above* the
camera, so a cube that rides the forks leaves the camera's field of view.
That gives a simple post-grab visibility heuristic:

* cube still visible after the grab -> it is lying on the floor, the grab
  missed -> ``NOT_CARRYING``
* cube gone from the world map with the forks up -> it is riding the
  forks, out of frame -> ``CARRYING``

Everything else (lift up but no witnessed grab) is ``UNKNOWN``.  Nothing
polls in the background — :attr:`CarryDetector.state` is evaluated
whenever a caller asks for it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pycozmo.cubes.identity import to_cube_id
from pycozmo.logger import logger

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
        clock: Monotonic clock (injectable for tests).

    The heuristic only makes sense with the forks up: while they are down
    no cube can ride them, which is authoritative, so the detector reports
    ``NOT_CARRYING`` (and forgets any observation — a lowered lift means
    the cube was placed).

    Limitation (inherent to visibility-based detection): a missed grab
    whose cube rolled out of view reads as ``CARRYING``, and a cube driven
    out of frame while carrying reads as ``NOT_CARRYING`` if it reappears.
    Re-check after ``clear()`` + a fresh grab to resynchronise.
    """

    #: Lift height above which a cube could actually ride the forks.
    #: ``MIN_LIFT_HEIGHT`` (fully down) is 32 mm, ``MAX`` (fully up) 92 mm.
    MIN_CARRY_LIFT_MM = 60.0
    #: Detections blink out for a frame or two; only after this long
    #: without a fused update is the cube "out of the frame".
    LOST_S = 2.0
    #: ``set_lift_height`` returns as soon as the command is sent, so right
    #: after a grab the lift is still travelling. Low lift readings are
    #: ignored for this long so transit is not mistaken for a drop.
    LIFT_TRANSIT_S = 2.0

    def __init__(
        self,
        get_cube_state: Callable[[int], Any] | None = None,
        get_lift_height: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._get_cube_state = get_cube_state
        self._get_lift_height = get_lift_height
        self._clock = clock
        self._lock = threading.Lock()
        self._cube_index: int | None = None
        self._grabbed_at: float = 0.0

    # ------------------------------------------------------------------
    # Observation hooks (called by CubeApproach)
    # ------------------------------------------------------------------

    def note_grab(self, cube_index: int) -> None:
        """Arm the observation after a grab: lift up, cube about to leave frame."""
        with self._lock:
            self._cube_index = int(cube_index)
            self._grabbed_at = self._clock()
        logger.info(
            "Carry: observing cube %d (out of frame = carrying)",
            to_cube_id(cube_index),
        )

    def clear(self) -> None:
        """Forget the observation — forks lowered / cube placed."""
        with self._lock:
            index = self._cube_index
            self._cube_index = None
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
        if lift < self.MIN_CARRY_LIFT_MM:
            # Forks down: authoritative "not carrying" — and the cube was
            # placed, so forget the observation. Skipped while the lift is
            # still travelling up right after a grab (LIFT_TRANSIT_S).
            in_transit = index is not None and now - grabbed_at < self.LIFT_TRANSIT_S
            if not in_transit:
                if index is not None:
                    self.clear()
                return CarryState.NOT_CARRYING

        if index is None:
            # Lift up, but nobody saw a grab: could be an empty lift raised
            # from the browser, or a cube we were never told about.
            return CarryState.UNKNOWN

        return (
            CarryState.NOT_CARRYING
            if self._seconds_since_seen(index, now) < self.LOST_S
            else CarryState.CARRYING
        )

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
    def cube_id(self) -> int | None:
        """Cube ID (1-3) being observed, or ``None`` when not armed."""
        with self._lock:
            return None if self._cube_index is None else to_cube_id(self._cube_index)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _lift_height_mm(self) -> float:
        if self._get_lift_height is None:
            return 0.0
        try:
            return float(self._get_lift_height())
        except Exception:  # noqa: BLE001 — a broken read must not break reads
            logger.exception("Carry: lift height read failed")
            return 0.0

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
