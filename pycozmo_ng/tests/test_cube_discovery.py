"""Cube discovery — the robot announces cubes, and it can announce them late.

The robot announces a light cube with ``ObjectAvailable`` (0xF3) when it hears
it over BLE. Two things make that unreliable as a one-off: a cube switched on
after startup, and the lossy Cozmo AP (an announcement is a single packet that
nothing retransmits). Discovery used to happen once, so a cube missing from the
map at that moment was never connected — and never lit — for the whole session,
which looks exactly like "sometimes a cube doesn't light".

The manager now merges whatever the robot currently reports on every keepalive
tick. These tests drive those ticks with doubles.
"""

from typing import Any

from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.cubes.manager import CubeManager
from pycozmo_ng.protocol.protocol_encoder import CubeLights, ObjectType
from pycozmo_ng.util.object import Object

#: Where the robot would put a cube handed over for connection.
SLOT_OBJECT_ID = 7


class FakeConn:
    """Records packets; has no ``available_objects`` (that is the Client's)."""

    def __init__(self) -> None:
        self.sent: list[Any] = []

    def send(self, packet) -> None:
        self.sent.append(packet)


class FakeClient:
    """A Client as far as discovery is concerned: the objects it heard about."""

    def __init__(self, objects: dict[int, Object] | None = None, confirm: bool = True) -> None:
        self.conn = FakeConn()
        self.available_objects: dict[int, Object] = dict(objects or {})
        self.confirm = confirm

    def announce(self, factory_id: int, object_type: ObjectType) -> None:
        """The robot reports a cube (an ``ObjectAvailable`` arrived)."""
        self.available_objects[factory_id] = Object(factory_id=factory_id, object_type=object_type)

    def wait_for_factory_ids(self, factory_ids, connect: bool = True, timeout: float = 10.0):
        del connect, timeout
        return {factory_id: SLOT_OBJECT_ID for factory_id in factory_ids} if self.confirm else {}


def _manager(objects: dict[int, Object] | None = None, confirm: bool = True) -> tuple[CubeManager, FakeClient]:
    """A manager with a fake client injected, as the keepalive would set it."""
    manager = CubeManager(CubeTrackingState())
    client = FakeClient(objects, confirm)
    manager._cli = client
    return manager, client


class TestMergeAvailableObjects:
    """What the manager does with the robot's object reports."""

    def test_a_cube_reported_late_is_added_and_connected(self) -> None:
        manager, client = _manager()
        assert manager.ct.cube_factory_id_map == {}

        client.announce(0x5E62BCA3, ObjectType.Block_LIGHTCUBE3)
        manager._keepalive_tick()

        assert manager.ct.cube_factory_id_map == {3: 0x5E62BCA3}
        assert manager.ct.object_id_map == {3: SLOT_OBJECT_ID}

    def test_an_empty_map_still_looks_next_tick(self) -> None:
        """The early return used to leave nothing looking at all."""
        manager, client = _manager()
        manager._keepalive_tick()
        assert manager.ct.cube_factory_id_map == {}

        client.announce(0x3B3958B8, ObjectType.Block_LIGHTCUBE1)
        manager._keepalive_tick()

        assert manager.ct.cube_factory_id_map == {1: 0x3B3958B8}

    def test_a_late_cube_is_lit_by_the_next_tick(self) -> None:
        """End to end: announce, tick, and the default repaint reaches it."""
        manager, client = _manager()

        client.announce(0xB3660B44, ObjectType.Block_LIGHTCUBE2)
        manager._keepalive_tick()

        paints = [packet for packet in client.conn.sent if isinstance(packet, CubeLights)]
        assert len(paints) == 1

    def test_merging_is_idempotent(self) -> None:
        manager, client = _manager()
        client.announce(0xB3660B44, ObjectType.Block_LIGHTCUBE2)

        assert manager.merge_available_objects(client) == [2]
        assert manager.merge_available_objects(client) == []
        assert manager.ct.cube_factory_id_map == {2: 0xB3660B44}

    def test_non_cube_objects_are_ignored(self) -> None:
        """Only light cubes are ours; a charger must not take a cube slot."""
        manager, client = _manager()
        client.announce(0x12345678, ObjectType.Charger_Basic)

        assert manager.merge_available_objects(client) == []
        assert manager.ct.cube_factory_id_map == {}

    def test_a_connection_without_object_reports_is_tolerated(self) -> None:
        """Packet handlers get the Connection, which has no ``available_objects``."""
        manager, _ = _manager()
        assert manager.merge_available_objects(FakeConn()) == []

    def test_setup_cube_mapping_still_reports_what_it_found(self) -> None:
        manager, client = _manager()
        client.announce(0x3B3958B8, ObjectType.Block_LIGHTCUBE1)

        manager.setup_cube_mapping(client)

        assert manager.ct.cube_factory_id_map == {1: 0x3B3958B8}
