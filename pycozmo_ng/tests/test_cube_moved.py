"""A cube that is moved loses its world pose (``ObjectMoved``, 0xB4).

Cubes have accelerometers and the robot reports motion per object. A moved cube
may be anywhere, so the pose fused from the camera before the move is discarded
— an unknown position beats a wrong one — and the next sighting starts a fresh
pose instead of blending towards the new place through the old one.

The pieces are separate jobs with separate tests: ``WorldMap.forget_cube`` (the
pose is dropped), ``CubeManager.on_object_moved`` (the event is matched to the
right cube), the accelerometer stream that makes the event arrive at all, and
the Brain wiring that joins them.
"""

import numpy as np

from pycozmo_ng import Cozmo
from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.cubes.identity import to_world_index
from pycozmo_ng.cubes.manager import CubeManager
from pycozmo_ng.lights import off_light
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.protocol.protocol_encoder import (
    CubeId,
    CubeLights,
    ObjectConnectionState,
    ObjectMoved,
    SetPropSlot,
    StreamObjectAccel,
)

#: Factory id used for the cube under test (a real one, from a live session).
FACTORY_ID = 0xB3660B44


class FakeConn:
    """Records what the manager sends; stands in for the transport connection."""

    def __init__(self) -> None:
        self.sent: list = []

    def send(self, packet) -> None:
        self.sent.append(packet)


class FakeCli:
    """Just enough Client: the connection state the manager reads, and a conn."""

    def __init__(self, connected=None) -> None:
        self.connected_objects = dict(connected or {})
        self.conn = FakeConn()


def _manager(factory_map=None, connected=None):
    """A CubeManager with the given factory map and robot-reported objects."""
    tracking = CubeTrackingState()
    tracking.cube_factory_id_map.update(factory_map or {2: FACTORY_ID})
    manager = CubeManager(tracking)
    return manager, FakeCli(connected or {7: {"factory_id": FACTORY_ID, "object_type": 1}})


def _moved(object_id: int) -> ObjectMoved:
    return ObjectMoved(
        timestamp=1,
        object_id=object_id,
        active_accel_x=120.0,
        active_accel_y=-30.0,
        active_accel_z=980.0,
    )


class TestForgetCube:
    """The world map side: dropping one cube's fused pose."""

    def _map_with_cube(self) -> WorldMap:
        world_map = WorldMap()
        world_map.update_cube_measurement(0, np.array([0.0, 0.0, 200.0]), np.eye(3), 0.9, 0.0)
        return world_map

    def test_forgetting_drops_the_pose(self) -> None:
        world_map = self._map_with_cube()
        assert world_map.get_cube(0) is not None

        assert world_map.forget_cube(0) is True

        assert world_map.get_cube(0) is None
        assert 0 not in world_map.get_cubes()

    def test_forgetting_an_unknown_cube_reports_false(self) -> None:
        assert WorldMap().forget_cube(2) is False

    def test_other_cubes_are_untouched(self) -> None:
        world_map = self._map_with_cube()
        world_map.update_cube_measurement(1, np.array([0.0, 0.0, 200.0]), np.eye(3), 0.9, 0.0)

        world_map.forget_cube(0)

        assert world_map.get_cube(1) is not None

    def test_next_sighting_starts_from_scratch(self) -> None:
        """The new measurement is the pose — nothing blends towards it."""
        world_map = self._map_with_cube()
        world_map.forget_cube(0)

        world_map.update_cube_measurement(0, np.array([0.0, 0.0, 300.0]), np.eye(3), 0.9, 0.0)

        # Camera 200 -> 300 mm ahead is world X 230 -> 330 (camera setback 30 mm).
        cube = world_map.get_cube(0)
        assert cube is not None
        np.testing.assert_allclose(cube.position, [330.0, 0.0, 38.0], atol=1e-6)

    def test_a_remembered_pose_would_have_been_blended(self) -> None:
        """Why forgetting matters: the smoothing would drag the cube back."""
        world_map = self._map_with_cube()

        world_map.update_cube_measurement(0, np.array([0.0, 0.0, 300.0]), np.eye(3), 0.9, 0.0)

        # Same two measurements, but the pre-move pose was kept: the fused
        # position only moved a fraction of the way.
        cube = world_map.get_cube(0)
        assert cube is not None
        assert cube.position[0] < 300.0


class TestOnObjectMoved:
    """The manager side: matching the event to the right cube."""

    def test_moved_cube_is_reported_by_id(self) -> None:
        manager, cli = _manager()
        moved: list[int] = []
        manager.set_moved_handler(moved.append)

        manager.on_object_moved(cli, _moved(7))

        assert moved == [2]

    def test_only_the_moved_cube_is_reported(self) -> None:
        manager, cli = _manager(
            factory_map={1: 0x3B3958B8, 2: FACTORY_ID},
            connected={
                5: {"factory_id": 0x3B3958B8, "object_type": 1},
                6: {"factory_id": FACTORY_ID, "object_type": 2},
            },
        )
        moved: list[int] = []
        manager.set_moved_handler(moved.append)

        manager.on_object_moved(cli, _moved(6))

        assert moved == [2]

    def test_unknown_object_is_ignored(self) -> None:
        manager, cli = _manager()
        moved: list[int] = []
        manager.set_moved_handler(moved.append)

        manager.on_object_moved(cli, _moved(99))

        assert moved == []

    def test_without_a_handler_nothing_breaks(self) -> None:
        """The event arrives whether or not anything cares about it."""
        manager, cli = _manager()
        manager.on_object_moved(cli, _moved(7))  # must not raise

    def test_handler_can_be_removed(self) -> None:
        manager, cli = _manager()
        moved: list[int] = []
        manager.set_moved_handler(moved.append)
        manager.set_moved_handler(None)

        manager.on_object_moved(cli, _moved(7))

        assert moved == []


class TestAccelStreaming:
    """A moved cube is only reported if its accelerometer is streamed."""

    def _connect(self, manager, cli, object_id: int = 7) -> None:
        manager.on_object_connection_state(
            cli,
            ObjectConnectionState(object_id=object_id, factory_id=FACTORY_ID, object_type=1, connected=True),
        )

    def test_connecting_a_cube_enables_its_accel_stream(self) -> None:
        manager, cli = _manager()

        self._connect(manager, cli)

        streams = [p for p in cli.conn.sent if isinstance(p, StreamObjectAccel)]
        assert len(streams) == 1
        assert (streams[0].object_id, streams[0].enable) == (7, True)

    def test_connecting_records_the_cube(self) -> None:
        manager, cli = _manager()

        self._connect(manager, cli)

        assert manager.connected_cubes == {2: 7}

    def test_disconnect_all_cubes_darkens_before_releasing_slots(self) -> None:
        """A cube put away must not be left glowing — it keeps its last state."""
        manager, cli = _manager()
        self._connect(manager, cli)
        cli.conn.sent.clear()

        manager.disconnect_all_cubes(cli)

        sent = cli.conn.sent
        assert isinstance(sent[0], CubeId)
        lights = [p for p in sent if isinstance(p, CubeLights)]
        assert len(lights) == 1
        assert all(state.on_color == off_light.on_color for state in lights[0].states)

        slots = [p for p in sent if isinstance(p, SetPropSlot)]
        assert [p.slot for p in slots] == [0, 1, 2, 3, 4]
        assert all(p.factory_id == 0 for p in slots)
        assert manager.connected_cubes == {}


class TestBrainWiring:
    """The two halves joined: a moved cube really does leave the map."""

    def test_moved_cube_loses_its_pose_end_to_end(self) -> None:
        cozmo = Cozmo()
        brain = cozmo._brain
        tracking = brain.cube_tracking
        tracking.cube_factory_id_map[2] = FACTORY_ID
        brain.world_map.update_cube_measurement(to_world_index(2), np.array([0.0, 0.0, 200.0]), np.eye(3), 0.9, 0.0)
        assert cozmo.world_map.get_cube(2) is not None

        cli = FakeCli({7: {"factory_id": FACTORY_ID, "object_type": 1}})
        brain.cubes.on_object_moved(cli, _moved(7))

        assert cozmo.world_map.get_cube(2) is None

    def test_a_cube_that_was_never_seen_stays_unknown(self) -> None:
        """The event is harmless for a cube the map has no pose for."""
        cozmo = Cozmo()
        brain = cozmo._brain
        brain.cube_tracking.cube_factory_id_map[2] = FACTORY_ID

        cli = FakeCli({7: {"factory_id": FACTORY_ID, "object_type": 1}})
        brain.cubes.on_object_moved(cli, _moved(7))

        assert cozmo.world_map.get_cube(2) is None
