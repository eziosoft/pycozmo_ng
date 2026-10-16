"""CubeIndicator — what a cube shows, and how little of it goes on the wire.

The indicator derives everything from collaborators that already exist, so the
whole sequence is exercised here against doubles: a fake manager (who is
connected, what was sent), a fake world map (which cubes the camera can see), a
fake clock, and a stand-in for the cube-visit authority.

Every test that touches the wire goes through the same seam the robot does:
``CubeManager.set_cube_states``. ``LightState`` has no ``__eq__``, so the
assertions compare corner colours as plain ints.
"""

import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from pycozmo_ng.cubes.identity import to_world_index
from pycozmo_ng.cubes.indicator import CORNERS, CubeIndicator
from pycozmo_ng.navigation.cube_visit import CubeVisit

#: Stands in for the Client — the fake manager ignores it.
CLI = object()


def corners(light) -> tuple[int, ...]:
    """One LightState as the four corner colours of a cube."""
    return (light.on_color,) * CORNERS


def spawn_now(name: str, fn: Callable[[Any], None]) -> None:
    """A spawn that runs the visit body immediately, with no motion token."""
    fn(None)


class FakeClock:
    """Monotonic clock under test control (seconds)."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeCubes:
    """Stands in for CubeManager: who is connected, and what was sent."""

    def __init__(self, connected=()) -> None:
        self._connected = {cube_id: cube_id for cube_id in connected}
        self.light_refresh = True
        self.sent: list[dict] = []

    @property
    def connected_cubes(self) -> dict[int, int]:
        return dict(self._connected)

    def connect(self, cube_id: int) -> None:
        self._connected[cube_id] = cube_id

    def disconnect(self, cube_id: int) -> None:
        self._connected.pop(cube_id, None)

    def set_cube_states(self, cli, states_by_cube) -> None:
        self.sent.append(states_by_cube)

    def set_light_refresh(self, enabled: bool) -> None:
        self.light_refresh = enabled

    def colors_sent(self) -> list[dict[int, tuple[int, ...]]]:
        """Everything sent so far, as {cube_id: (corner colours)} per tick."""
        return [
            {cube_id: tuple(state.on_color for state in states) for cube_id, states in frame.items()}
            for frame in self.sent
        ]


class FakeWorldMap:
    """Stands in for WorldMap: cube *index* -> a pose with ``updated_at``."""

    def __init__(self) -> None:
        self._cubes: dict[int, SimpleNamespace] = {}

    def get_cubes(self) -> dict[int, SimpleNamespace]:
        return dict(self._cubes)

    def see(self, cube_id: int, at: float) -> None:
        """Record a fused sighting of a cube at time ``at``."""
        self._cubes[to_world_index(cube_id)] = SimpleNamespace(updated_at=at)


def _make(connected=(), *, seen=(), at=0.0, visit=None, clock=None):
    cubes = FakeCubes(connected)
    world_map = FakeWorldMap()
    for cube_id in seen:
        world_map.see(cube_id, at)
    clock = clock or FakeClock(at)
    indicator = CubeIndicator(cubes, world_map=world_map, visit=visit, clock=clock)
    return indicator, cubes, world_map, clock


class TestCubeIndicatorState:
    """The state machine: which light each cube shows."""

    def test_idle_cube_is_dark(self) -> None:
        """A cube that is merely connected costs no battery."""
        indicator, _, _, _ = _make(connected=(1, 2, 3))
        assert indicator.light_for(1).on_color == CubeIndicator.IDLE_LIGHT.on_color

    def test_connect_blinks_twice(self) -> None:
        indicator, cubes, _, _ = _make(connected=(1,))
        for _ in range(4):
            indicator.tick()
        assert cubes.colors_sent() == [
            {1: corners(CubeIndicator.CONNECT_LIGHT)},
            {1: corners(CubeIndicator.IDLE_LIGHT)},
            {1: corners(CubeIndicator.CONNECT_LIGHT)},
            {1: corners(CubeIndicator.IDLE_LIGHT)},
        ]

    def test_blink_ends_into_the_steady_state(self) -> None:
        """A cube seen while it blinks lands on dim blue, not dark."""
        indicator, cubes, _, _ = _make(connected=(1,), seen=(1,))
        for _ in range(5):
            indicator.tick()
        assert cubes.colors_sent()[-1] == {1: corners(CubeIndicator.SEEN_LIGHT)}

    def test_seen_cube_is_dim_blue(self) -> None:
        indicator, _, _, _ = _make(connected=(1,), seen=(1,))
        assert indicator.light_for(1).on_color == CubeIndicator.SEEN_LIGHT.on_color
        assert indicator.light_for(1).on_color != CubeIndicator.IDLE_LIGHT.on_color

    def test_seen_light_is_dim_not_full_brightness(self) -> None:
        """Dim blue is the point — a long-lived state must not drain the cube."""
        from pycozmo_ng.lights import blue_light

        assert 0 < CubeIndicator.SEEN_LIGHT.on_color < blue_light.on_color

    def test_stale_sighting_is_not_seen(self) -> None:
        clock = FakeClock()
        indicator, _, _, _ = _make(connected=(1,), seen=(1,), at=0.0, clock=clock)
        clock.now = 10.0
        assert indicator.light_for(1).on_color == CubeIndicator.IDLE_LIGHT.on_color

    def test_approached_cube_is_red(self) -> None:
        visit = SimpleNamespace(active_cube_id=2)
        indicator, _, _, _ = _make(connected=(2,), visit=visit)
        assert indicator.light_for(2).on_color == CubeIndicator.APPROACH_LIGHT.on_color

    def test_red_beats_dim_blue(self) -> None:
        """The cube being picked up is, by definition, in view."""
        visit = SimpleNamespace(active_cube_id=2)
        indicator, _, _, _ = _make(connected=(2,), seen=(2,), visit=visit)
        assert indicator.light_for(2).on_color == CubeIndicator.APPROACH_LIGHT.on_color

    def test_other_cubes_keep_their_own_state(self) -> None:
        visit = SimpleNamespace(active_cube_id=2)
        indicator, _, _, _ = _make(connected=(1, 2, 3), seen=(1, 3), visit=visit)
        assert indicator.light_for(1).on_color == CubeIndicator.SEEN_LIGHT.on_color
        assert indicator.light_for(2).on_color == CubeIndicator.APPROACH_LIGHT.on_color
        assert indicator.light_for(3).on_color == CubeIndicator.SEEN_LIGHT.on_color

    def test_a_reconnect_does_not_blink_again(self) -> None:
        """The keepalive reconnects dropped cubes; that is not a new cube."""
        indicator, cubes, _, _ = _make(connected=(1,))
        for _ in range(4):
            indicator.tick()  # the connect blink finishes
        cubes.sent.clear()

        cubes.disconnect(1)
        indicator.tick()
        cubes.connect(1)
        for _ in range(4):
            indicator.tick()

        # The reconnected cube has its (dark) steady state re-asserted — once —
        # but it never blinks a second time.
        assert cubes.colors_sent() == [{1: corners(CubeIndicator.IDLE_LIGHT)}]
        assert indicator.light_for(1).on_color == CubeIndicator.IDLE_LIGHT.on_color

    def test_restarting_the_indicator_blinks_again(self) -> None:
        """A new run is a new session: acknowledge the connections once more."""
        indicator, cubes, _, _ = _make(connected=(1,))
        for _ in range(4):
            indicator.tick()
        indicator.stop()
        cubes.sent.clear()

        indicator.tick()

        assert cubes.colors_sent() == [{1: corners(CubeIndicator.CONNECT_LIGHT)}]

    def test_a_cube_seen_after_a_reconnect_is_blue_not_blinking(self) -> None:
        indicator, cubes, world_map, _ = _make(connected=(1,))
        for _ in range(4):
            indicator.tick()
        cubes.sent.clear()

        cubes.disconnect(1)
        indicator.tick()
        cubes.connect(1)
        world_map.see(1, 0.0)
        indicator.tick()

        assert cubes.colors_sent()[-1] == {1: corners(CubeIndicator.SEEN_LIGHT)}

    def test_no_world_map_means_never_seen(self) -> None:
        cubes = FakeCubes(connected=(1,))
        indicator = CubeIndicator(cubes, world_map=None, clock=FakeClock())
        assert indicator.light_for(1).on_color == CubeIndicator.IDLE_LIGHT.on_color

    def test_no_visit_means_nothing_is_approached(self) -> None:
        indicator, _, _, _ = _make(connected=(1,), seen=(1,))
        assert indicator.light_for(1).on_color == CubeIndicator.SEEN_LIGHT.on_color


class TestCubeIndicatorTraffic:
    """What actually goes to the cubes — the battery-relevant part."""

    def test_steady_state_is_sent_once(self) -> None:
        """Ticking a quiet cube must not repaint it ten times a second."""
        indicator, cubes, _, _ = _make(connected=(1,), seen=(1,))
        for _ in range(10):
            indicator.tick()
        steady = [frame for frame in cubes.colors_sent() if frame == {1: corners(CubeIndicator.SEEN_LIGHT)}]
        assert len(steady) == 1

    def test_only_the_changed_cube_is_sent(self) -> None:
        indicator, cubes, world_map, _ = _make(connected=(1, 2))
        for _ in range(4):
            indicator.tick()  # both blinks finish
        cubes.sent.clear()
        world_map.see(2, 0.0)
        indicator.tick()
        assert cubes.colors_sent() == [{2: corners(CubeIndicator.SEEN_LIGHT)}]

    def test_steady_state_is_re_asserted_after_the_refresh_interval(self) -> None:
        """A cube's BLE write can be lost; a change-only policy would stick."""
        indicator, cubes, _, clock = _make(connected=(1,))
        for _ in range(5):
            indicator.tick()  # the connect blink finishes; the cube stays dark
        cubes.sent.clear()

        # Nothing about the cube changes — only the clock.
        clock.now += CubeIndicator.DEFAULT_REFRESH_INTERVAL_S - 0.1
        indicator.tick()
        assert cubes.colors_sent() == []

        clock.now += 0.2
        indicator.tick()
        assert cubes.colors_sent() == [{1: corners(CubeIndicator.IDLE_LIGHT)}]

    def test_indicator_pauses_keepalive_refresh(self) -> None:
        """Otherwise the keepalive's green repaint overwrites every state."""
        indicator, cubes, _, _ = _make(connected=(1,))
        assert cubes.light_refresh is True
        indicator.start(CLI)
        try:
            assert cubes.light_refresh is False
        finally:
            indicator.stop()
        assert cubes.light_refresh is True

    def test_indicator_reclaims_the_leds(self) -> None:
        """A repaint re-enabled by something else is paused again on the next tick."""
        indicator, cubes, _, _ = _make(connected=(1,))
        cubes.light_refresh = True
        indicator.tick()
        assert cubes.light_refresh is False

    def test_start_is_idempotent(self) -> None:
        indicator, _, _, _ = _make(connected=(1,))
        indicator.start(CLI)
        try:
            indicator.start(CLI)
            assert indicator.running is True
        finally:
            indicator.stop()

    def test_stop_without_start_is_safe(self) -> None:
        indicator, _, _, _ = _make()
        indicator.stop()
        assert indicator.running is False

    def test_background_thread_actually_drives_the_cubes(self) -> None:
        """The real wiring: a thread, not just ``tick()``."""
        indicator, cubes, _, _ = _make(connected=(1,))
        indicator.start(CLI)
        try:
            deadline = time.monotonic() + 2.0
            while not cubes.sent and time.monotonic() < deadline:
                time.sleep(0.01)
            assert cubes.colors_sent()[0] == {1: corners(CubeIndicator.CONNECT_LIGHT)}
        finally:
            indicator.stop()


class TestVisitAnnouncesItsCube:
    """``CubeVisit.active_cube_id`` — how the indicator knows what to turn red."""

    def test_a_refused_visit_leaves_the_running_one_alone(self) -> None:
        visit = CubeVisit(lambda name, fn: None)  # a job that never finishes
        assert visit.begin("approach", lambda job: None, cube_id=1) is True
        assert visit.begin("stack", lambda job: None, cube_id=2) is False
        assert visit.active_cube_id == 1

    def test_visit_without_a_cube_reports_none(self) -> None:
        visit = CubeVisit(spawn_now)
        assert visit.begin("approach", lambda job: None) is True
        assert visit.active_cube_id is None

    def test_cube_is_reported_while_the_visit_runs(self) -> None:
        seen = []
        visit = CubeVisit(spawn_now)
        visit.begin("approach", lambda job: seen.append(visit.active_cube_id), cube_id=3)
        assert seen == [3]
        assert visit.active_cube_id is None  # cleared when the visit ended

    def test_cube_is_cleared_when_the_visit_raises(self) -> None:
        def boom(job) -> None:
            raise RuntimeError("grab failed")

        def spawn(name, fn) -> None:
            try:
                fn(None)
            except RuntimeError:
                pass

        visit = CubeVisit(spawn)
        visit.begin("approach", boom, cube_id=1)
        assert visit.active_cube_id is None
        assert visit.busy is False

    def test_the_indicator_reads_it(self) -> None:
        """The seam the two classes meet on, end to end."""
        visit = SimpleNamespace(active_cube_id=None)
        indicator, cubes, _, _ = _make(connected=(1, 2), visit=visit)
        for _ in range(4):
            indicator.tick()  # both blinks finish: both cubes dark
        cubes.sent.clear()
        visit.active_cube_id = 2
        indicator.tick()
        assert cubes.colors_sent() == [{2: corners(CubeIndicator.APPROACH_LIGHT)}]
        visit.active_cube_id = None
        indicator.tick()
        assert cubes.colors_sent()[-1] == {2: corners(CubeIndicator.IDLE_LIGHT)}


def test_blink_is_two_flashes() -> None:
    """Two 'on' frames in the blink means two flashes, as the ack promises."""
    assert CubeIndicator.BLINK.count(True) == 2
