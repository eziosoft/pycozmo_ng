"""Tests for the public Cozmo facade (Cozmo + VisionAPI/CubesAPI/NavigatorAPI/...).

The facade's contract is that every documented call reaches a method that
*really exists* on the underlying subsystem. All tests below run against
the real classes with no robot connected, so a renamed or removed
subsystem method fails here instead of raising AttributeError in front of
a user (the bug this suite guards against).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

import pycozmo_ng
from pycozmo_ng import Cozmo, cubes, navigation


@pytest.fixture
def cozmo() -> Cozmo:
    """A Cozmo instance with the full Brain wiring, but no robot."""
    return Cozmo()


def _seed_cube(brain, cube_idx: int = 0) -> None:
    """Put one cube (0-indexed world-map key) straight ahead of the robot."""
    wm = brain.world_map
    wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
    wm.update_cube_measurement(cube_idx, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)


class TestConstruction:
    def test_constructs_without_robot(self, cozmo):
        assert cozmo.is_connected is False
        assert cozmo.cli is None

    @pytest.mark.parametrize("name", ["vision", "cubes", "navigator", "world_map", "sensors", "web_viewer"])
    def test_subsystems_available(self, cozmo, name):
        assert getattr(cozmo, name) is not None

    def test_phantom_protocol_modules_removed(self):
        # Interfaces with no second implementation were deleted (COZMO-DOC-13).
        for name in (
            "NavigatorInterface",
            "CubeApproachInterface",
            "WorldMapInterface",
            "DriveToPointFn",
            "MoveForwardFn",
        ):
            assert not hasattr(navigation, name)
        for name in (
            "CubeManagerInterface",
            "CubeInterface",
            "CubeCyclerInterface",
        ):
            assert not hasattr(cubes, name)


class TestNavigatorFacade:
    def test_drive_to_without_robot_returns_false(self, cozmo):
        assert cozmo.navigator.drive_to(100.0, 0.0) is False

    def test_drive_to_forwards_keyword_args(self, cozmo):
        # Real signature: drive_to_point(tx, ty, *, stop_mm, timeout_s, ...).
        with pytest.raises(TypeError):
            cozmo.navigator.drive_to(100.0, 0.0, nonsense_kwarg=1)

    def test_dock_without_robot_does_not_raise(self, cozmo):
        cozmo.navigator.dock()

    def test_approach_unknown_cube_returns_false(self, cozmo):
        assert cozmo.navigator.approach_cube(1) is False

    def test_approach_without_robot_returns_false(self, cozmo):
        _seed_cube(cozmo._brain)  # cube IS in the map, but no cli
        assert cozmo.navigator.approach_cube(1) is False

    def test_approach_resolves_cube_id_to_world_map_key(self, cozmo):
        """Facade speaks cube IDs 1-3; the world map keys 0-2."""
        nav = cozmo._brain.navigator
        nav.cli = MagicMock()  # pretend connection so resolution proceeds
        _seed_cube(cozmo._brain, cube_idx=0)

        calls: list[tuple[int, float, float]] = []

        def _fake_visit(idx: int, x: float, y: float) -> bool:
            # Stands in for the routing seam: records the visit and
            # reports that it started.
            calls.append((idx, x, y))
            return True

        nav.approach_cube_at = _fake_visit

        assert cozmo.navigator.approach_cube(1) is True  # ID 1 -> world key 0
        assert cozmo.navigator.approach_cube(2) is False  # not detected

        assert len(calls) == 1
        idx, x, y = calls[0]
        assert idx == 0
        assert isinstance(x, float) and isinstance(y, float)
        assert x > 0  # cube straight ahead -> world +X

    def test_astar_toggle_roundtrip(self, cozmo):
        assert cozmo.navigator.is_astar_enabled() is False
        cozmo.navigator.set_astar_enabled(True)
        assert cozmo.navigator.is_astar_enabled() is True
        cozmo.navigator.set_astar_enabled(False)
        assert cozmo.navigator.is_astar_enabled() is False


class TestWorldMapFacade:
    def test_empty_map(self, cozmo):
        assert cozmo.world_map.cubes == {}
        assert cozmo.world_map.path == []
        assert cozmo.world_map.get_cube(1) is None

    def test_robot_pose_defaults(self, cozmo):
        position, yaw = cozmo.world_map.robot_pose
        np.testing.assert_allclose(position, [0.0, 0.0, 0.0])
        assert yaw == 0.0

    def test_cubes_keyed_by_cube_id(self, cozmo):
        _seed_cube(cozmo._brain, cube_idx=0)

        cubes_by_id = cozmo.world_map.cubes
        assert list(cubes_by_id) == [1]  # world key 0 -> cube ID 1
        assert cozmo.world_map.get_cube(1) is not None
        assert cozmo.world_map.get_cube(2) is None

    def test_path_trail(self, cozmo):
        wm = cozmo._brain.world_map
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_robot_pose([5.0, 6.0, 0.0], 0.0, origin_id=1, record_path=True)

        path = cozmo.world_map.path
        assert len(path) >= 2


class TestCubesFacade:
    def test_connected_empty(self, cozmo):
        assert cozmo.cubes.connected == []

    def test_connected_lists_cube_ids(self, cozmo):
        ct = cozmo._brain.cubes.ct
        ct.object_id_map[1] = 42
        ct.object_id_map[3] = 7
        assert sorted(cozmo.cubes.connected) == [1, 3]

    def test_discover_requires_connection(self, cozmo):
        with pytest.raises(RuntimeError):
            cozmo.cubes.discover()

    def test_start_lights_requires_connection(self, cozmo):
        with pytest.raises(RuntimeError):
            cozmo.cubes.start_lights()

    def test_stop_lights_without_connection_is_safe(self, cozmo):
        cozmo.cubes.stop_lights()

    def test_flat_cozmo_methods_delegate(self, cozmo):
        # demo.py uses the flat Cozmo methods; they share the CubesAPI path.
        with pytest.raises(RuntimeError):
            cozmo.discover_cubes()
        with pytest.raises(RuntimeError):
            cozmo.start_cube_lights()
        cozmo.stop_cube_lights()


class TestSensorsFacade:
    def test_defaults_without_robot(self, cozmo):
        assert cozmo.sensors.battery_voltage == 0.0
        assert cozmo.sensors.imu == {"accel": (0, 0, 0), "gyro": (0, 0, 0)}
        assert cozmo.sensors.cliff is None
        assert cozmo.sensors.wheel_speeds == {"left": 0, "right": 0}
        assert cozmo.sensors.head_angle == 0.0
        assert cozmo.sensors.lift_height == 0.0


class TestVisionFacade:
    def test_frames_empty_before_start(self, cozmo):
        assert cozmo.vision.latest_frame is None
        assert cozmo.vision.latest_annotated_frame is None
        assert cozmo.vision.latest_detections == []

    def test_pipeline_not_created_before_setup(self, cozmo):
        assert cozmo.vision.detection_pipeline is None
        assert cozmo.vision.camera_calibration is None
        assert cozmo.vision.annotation is None

    def test_start_requires_connection(self, cozmo):
        with pytest.raises(RuntimeError):
            cozmo.vision.start()

    def test_stop_without_connection_is_safe(self, cozmo):
        cozmo.vision.stop()

    def test_setup_vision_requires_connection(self, cozmo):
        # Raises before attempting calibration retrieval from the robot.
        with pytest.raises(RuntimeError):
            cozmo.setup_vision()


class TestWebViewerFacade:
    def test_is_running_false_initially(self, cozmo):
        assert cozmo.web_viewer.is_running is False

    def test_stop_before_start_is_safe(self, cozmo):
        cozmo.web_viewer.stop()


class TestDocumentedExamplesStayTrue:
    """Guard against docstring drift: the examples must call real API."""

    def test_module_docstring_example(self, cozmo):
        assert "cozmo.vision.start()" in pycozmo_ng.__doc__
        assert callable(cozmo.vision.start)
        assert "cozmo.navigator.drive_to(" in pycozmo_ng.__doc__
        assert callable(cozmo.navigator.drive_to)
        assert "cozmo.sensors.battery_voltage" in pycozmo_ng.__doc__
        assert isinstance(cozmo.sensors.battery_voltage, float)

    def test_class_docstring_example(self, cozmo):
        assert "cozmo.vision.start()" in Cozmo.__doc__
        assert callable(cozmo.vision.start)
        assert "cozmo.navigator.drive_to(x=100, y=0)" in Cozmo.__doc__
        assert callable(cozmo.navigator.drive_to)
