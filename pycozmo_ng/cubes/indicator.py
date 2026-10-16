"""CubeIndicator — the cube LEDs that say what the robot is doing.

A cube is a four-LED lamp running on batteries, so its light has to earn its
power. This class is the single owner of those LEDs while it runs: it reads
state that already exists elsewhere — which cubes are connected
(:class:`~pycozmo_ng.cubes.manager.CubeManager`), which ones the camera can
currently see (fused cube poses in :class:`~pycozmo_ng.navigation.world_map.WorldMap`),
and which one a cube visit is working on
(:class:`~pycozmo_ng.navigation.cube_visit.CubeVisit`) — and sends only the
*changes*, plus a periodic re-assert of the steady state (a cube's BLE write can
be lost; see :meth:`CubeIndicator.tick`).

Each cube shows the highest-priority state that applies to it:

=================  ==========================================================
state              when
=================  ==========================================================
two green blinks   the cube connected (acknowledged once, then never again)
red                a cube visit (grab or stack) is working on this cube
dim blue           the robot can see the cube right now
off                everything else — an idle cube stays dark
=================  ==========================================================

Nothing is pushed in from outside: every state is derived, which is what makes
the sequence testable with doubles (see ``tests/test_cube_indicator.py``).
"""

import threading
import time
from collections.abc import Callable
from typing import Any

from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.lights import dim_blue_light, green_light, off_light, red_light
from pycozmo_ng.logger import logger_cubes
from pycozmo_ng.protocol.protocol_encoder import LightState

__all__ = ["CubeIndicator"]

#: Corners on the top face of a light cube.
CORNERS = 4


class CubeIndicator:
    """Drives the steady-state LED of every connected cube in the background.

    Args:
        cubes: The :class:`~pycozmo_ng.cubes.manager.CubeManager` that owns the
            connections and sends ``CubeLights``.
        world_map: Fused cube poses. The age of a pose (``updated_at``) is how
            "the robot can see the cube" is judged. Optional — without a map
            no cube is ever dim blue.
        visit: The cube-visit authority
            (:class:`~pycozmo_ng.navigation.cube_visit.CubeVisit`) — its
            ``active_cube_id`` is the cube being grabbed or stacked right now.
            Optional.
        poll_interval_s: Seconds between LED updates.
        seen_window_s: How long a fused cube pose counts as a sighting.
        refresh_interval_s: How long a steady state may go un-re-sent; see
            :meth:`tick` for why a change-only policy is not enough.
        clock: Monotonic clock (injectable for tests).
    """

    #: Seconds between LED updates. Also one blink frame.
    DEFAULT_POLL_INTERVAL_S = 0.1
    #: How long a fused cube pose counts as "the robot can see it" (seconds).
    DEFAULT_SEEN_WINDOW_S = 1.0
    #: How long a steady state may go un-re-sent (seconds). See :meth:`tick`.
    DEFAULT_REFRESH_INTERVAL_S = 2.0

    #: Connect acknowledgement: two flashes, one tick each (on, off, on, off).
    #: Shown once per cube per run — a keepalive reconnect is not a new cube.
    BLINK: tuple[bool, ...] = (True, False, True, False)
    #: Light for a connect blink.
    CONNECT_LIGHT: LightState = green_light
    #: Light while a cube visit is working on the cube.
    APPROACH_LIGHT: LightState = red_light
    #: Light while the robot can see the cube.
    SEEN_LIGHT: LightState = dim_blue_light
    #: Light when the cube has nothing to show (battery: stay dark).
    IDLE_LIGHT: LightState = off_light

    def __init__(
        self,
        cubes: Any,
        world_map: Any = None,
        visit: Any = None,
        *,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        seen_window_s: float = DEFAULT_SEEN_WINDOW_S,
        refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cubes = cubes
        self._world_map = world_map
        self._visit = visit
        self._poll_interval_s = float(poll_interval_s)
        self._seen_window_s = float(seen_window_s)
        self._refresh_interval_s = float(refresh_interval_s)
        self._clock = clock
        self._cli = None
        #: Cubes whose connection the indicator has already acknowledged.
        self._known: set[int] = set()
        #: Cubes that have had their connect blink this run. The keepalive
        #: reconnects a cube the robot dropped, and that must not blink again:
        #: the acknowledgement is for a cube joining, not for every blip.
        self._blinked: set[int] = set()
        #: Cube ID -> current blink frame, for cubes mid-blink.
        self._blink: dict[int, int] = {}
        #: Cube ID -> (last LED tuple sent, when). Kept so untouched cubes cost
        #: nothing between refreshes.
        self._sent: dict[int, tuple[tuple[LightState, ...], float]] = {}
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the indicator thread is running."""
        return self._thread is not None

    def start(self, cli) -> None:
        """Start showing states. No-op if already running.

        The indicator owns the LEDs from here, so the keepalive's periodic
        repaint is paused until :meth:`stop`.
        """
        if self._thread is not None:
            return
        self._cli = cli
        self._cubes.set_light_refresh(False)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CubeIndicator", daemon=True)
        self._thread.start()
        logger_cubes.info("Cube indicator started (blink on connect, dim blue when seen, red when going for a cube)")

    def stop(self) -> None:
        """Stop showing states and hand the LEDs back to the keepalive.

        The run is over, so the acknowledged-cube bookkeeping goes with it: the
        next :meth:`start` blinks every connected cube again.
        """
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop_event = None
        self._known.clear()
        self._blinked.clear()
        self._blink.clear()
        self._sent.clear()
        self._cubes.set_light_refresh(True)
        logger_cubes.info("Cube indicator stopped")

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def light_for(self, cube_id: int) -> LightState:
        """The LED state one cube should show right now.

        Priority: connect blink, then a cube visit (red), then a sighting
        (dim blue), otherwise dark.
        """
        frame = self._blink.get(cube_id)
        if frame is not None:
            return self.CONNECT_LIGHT if self.BLINK[frame] else self.IDLE_LIGHT
        if cube_id == self._approaching_cube_id():
            return self.APPROACH_LIGHT
        if cube_id in self._visible_cube_ids():
            return self.SEEN_LIGHT
        return self.IDLE_LIGHT

    def tick(self) -> None:
        """Send every LED state that changed — and re-assert the steady ones.

        Independent of :meth:`start`: the indicator owns the LEDs whenever it
        sends, and stepping the sequence without a robot is how it is tested
        (the manager decides what a send actually does).

        Two things can leave a cube showing the wrong colour, and both are
        invisible to a change-only policy: the robot relays ``CubeLights`` to
        the cube over BLE, where a write can simply be dropped (nothing
        retransmits it — see :meth:`~pycozmo_ng.cubes.manager.CubeManager.select_cube`
        for the other), and a dropped write used to stick until that cube's
        state next changed, which for an idle cube is never.  So a steady state
        is re-sent at least every ``refresh_interval_s``: the cost is two
        packets, and a wrong LED cannot outlive the next refresh.
        """
        # Own the LEDs: the keepalive's periodic green repaint would overwrite
        # every state we send — and burn battery repainting the same colour.
        if self._cubes.light_refresh:
            self._cubes.set_light_refresh(False)

        connected = self._cubes.connected_cubes
        self._note_connections(connected)

        now = self._clock()
        changed = {}
        for cube_id in connected:
            state = (self.light_for(cube_id),) * CORNERS
            sent = self._sent.get(cube_id)
            if sent is None or sent[0] != state or now - sent[1] >= self._refresh_interval_s:
                changed[cube_id] = state
        if changed:
            self._cubes.set_cube_states(self._cli, changed)
            for cube_id, state in changed.items():
                self._sent[cube_id] = (state, now)

        self._advance_blink()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _note_connections(self, connected: dict[int, int]) -> None:
        """Blink cubes that just joined, forget those gone.

        The blink is a *connection* acknowledgement, so each cube gets it once
        per run: a keepalive reconnect (the robot drops idle cubes and the
        keepalive brings them back) must not blink the cube over and over.
        """
        for cube_id in connected:
            if cube_id in self._known:
                continue
            self._known.add(cube_id)
            if cube_id in self._blinked:
                logger_cubes.debug("Cube %d reconnected - no blink (already acknowledged)", cube_id)
                continue
            self._blinked.add(cube_id)
            self._blink[cube_id] = 0
            logger_cubes.info("Cube %d connected - blinking twice", cube_id)

        for cube_id in [cube_id for cube_id in self._known if cube_id not in connected]:
            self._known.discard(cube_id)
            self._blink.pop(cube_id, None)
            self._sent.pop(cube_id, None)
            logger_cubes.info("Cube %d gone - indicator state dropped", cube_id)

    def _advance_blink(self) -> None:
        """Move each blinking cube to its next frame, dropping finished blinks."""
        for cube_id, frame in list(self._blink.items()):
            if frame + 1 < len(self.BLINK):
                self._blink[cube_id] = frame + 1
            else:
                del self._blink[cube_id]

    def _visible_cube_ids(self) -> set[int]:
        """Cube IDs (1-3) the robot can see — a recent fused camera measurement."""
        if self._world_map is None:
            return set()
        now = self._clock()
        return {
            to_cube_id(cube_index)
            for cube_index, cube_state in self._world_map.get_cubes().items()
            if now - float(cube_state.updated_at) <= self._seen_window_s
        }

    def _approaching_cube_id(self) -> int | None:
        """The cube a cube visit is working on right now, if any."""
        return getattr(self._visit, "active_cube_id", None)

    def _loop(self) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger_cubes.debug("Cube indicator tick failed", exc_info=True)
            if self._stop_event is not None and self._stop_event.wait(self._poll_interval_s):
                return
