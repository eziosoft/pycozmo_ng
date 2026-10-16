"""How cube LED packets go on the wire.

``CubeLights`` carries no address: the robot paints whichever cube the last
``CubeId`` selected (see ``docs/protocol.md``). That makes ``CubeId`` and the
``CubeLights`` after it a single unit, and two things can break it:

- another thread's ``CubeId`` landing between the two. Three callers paint —
  the indicator, the keepalive and the public API — so a pair must be
  indivisible, or the lights land on the wrong cube and the intended one keeps
  showing a stale colour;
- the keepalive repainting its default colour over states the indicator owns.
  Its flag is checked per cube, not once per repaint: the indicator can take
  over mid-loop, and at most that one cube is mis-painted (the indicator's own
  periodic refresh corrects it).

These are wire-shape tests: no robot, and the doubles are deliberately dumb.
"""

import threading
import time
from collections.abc import Callable
from typing import Any

from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.cubes.manager import CubeManager
from pycozmo_ng.lights import green_light, off_light
from pycozmo_ng.protocol.protocol_encoder import CubeLights

#: Factory ids only have to be distinct here.
FACTORY_ID = 0xB3660B44


class FakeConn:
    """Records packets, and can react to one landing."""

    def __init__(self, on_send: Callable[[Any], None] | None = None) -> None:
        self.sent: list[Any] = []
        self.on_send = on_send

    def send(self, packet) -> None:
        self.sent.append(packet)
        if self.on_send is not None:
            self.on_send(packet)


class FakeClient:
    """Just enough Client: a connection to send through."""

    def __init__(self, on_send: Callable[[Any], None] | None = None) -> None:
        self.conn = FakeConn(on_send)


def _manager(connected: tuple[int, ...] = (2, 3), *, on_send: Callable[[Any], None] | None = None) -> CubeManager:
    """A manager with ``connected`` cubes, and a fake client already injected.

    ``_cli`` is what the keepalive would have set; the test sets it directly
    because that is the only way to reach the keepalive's repaint without
    starting its thread.
    """
    tracking = CubeTrackingState()
    for cube_id in connected:
        tracking.cube_factory_id_map[cube_id] = FACTORY_ID + cube_id
        tracking.object_id_map[cube_id] = cube_id
    manager = CubeManager(tracking)
    manager._cli = FakeClient(on_send)
    return manager


def _packets(manager: CubeManager) -> list[str]:
    """What went on the wire, by packet name."""
    return [type(packet).__name__ for packet in manager._cli.conn.sent]


def _run_while_held(manager: CubeManager, target: Callable[[], None]) -> None:
    """Run ``target`` in a thread while the LED lock is held, then release.

    Holding the lock first is how the pair's indivisibility is checked
    deterministically: the call must not get a single packet out until the
    lock is free.
    """
    manager._led_lock.acquire()
    worker = threading.Thread(target=target)
    worker.start()
    time.sleep(0.05)
    assert manager._cli.conn.sent == []  # nothing half-sent
    manager._led_lock.release()
    worker.join(timeout=1.0)
    assert not worker.is_alive()


class TestLedPacketsAreIndivisible:
    """``CubeId`` + ``CubeLights`` must not be split by another sender."""

    def test_states_go_out_as_id_then_lights(self) -> None:
        manager = _manager(connected=(2,))

        manager.set_cube_states(manager._cli, {2: (off_light,) * 4})

        assert _packets(manager) == ["CubeId", "CubeLights"]

    def test_another_sender_cannot_split_a_pair(self) -> None:
        manager = _manager(connected=(2,))

        _run_while_held(manager, lambda: manager.set_cube_states(manager._cli, {2: (off_light,) * 4}))

        assert _packets(manager) == ["CubeId", "CubeLights"]

    def test_the_keepalive_repaint_takes_the_same_lock(self) -> None:
        manager = _manager(connected=(2,))

        _run_while_held(manager, manager._send_lights_locked)

        assert _packets(manager) == ["CubeId", "CubeLights"]

    def test_the_connect_path_selects_under_the_lock_too(self) -> None:
        """A connect's ``CubeId`` is a half-pair as well, and must be held."""
        manager = _manager(connected=())

        _run_while_held(manager, lambda: manager.select_cube(manager._cli, 7))

        assert _packets(manager) == ["CubeId"]


class TestKeepaliveYieldsTheLeds:
    """While the indicator owns the LEDs, the keepalive paints nothing."""

    def test_the_default_repaint_paints_every_connected_cube(self) -> None:
        manager = _manager(connected=(2, 3))

        manager._send_lights_locked()

        assert _packets(manager) == ["CubeId", "CubeLights"] * 2
        painted = [packet for packet in manager._cli.conn.sent if isinstance(packet, CubeLights)]
        assert [packet.states[0].on_color for packet in painted] == [green_light.on_color] * 2

    def test_nothing_is_painted_when_the_leds_are_owned_elsewhere(self) -> None:
        manager = _manager(connected=(2, 3))
        manager.set_light_refresh(False)

        manager._send_lights_locked()

        assert manager._cli.conn.sent == []

    def test_a_repaint_under_way_stops_at_the_next_cube(self) -> None:
        """The indicator can take over mid-loop — the flag is checked per cube."""
        manager: CubeManager | None = None

        def take_over(packet) -> None:
            if isinstance(packet, CubeLights):
                assert manager is not None
                manager.set_light_refresh(False)

        manager = _manager(connected=(2, 3), on_send=take_over)

        manager._send_lights_locked()

        # The first cube was already painted; the second was never selected.
        assert _packets(manager) == ["CubeId", "CubeLights"]
