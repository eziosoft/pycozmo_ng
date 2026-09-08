"""Tests for the world-map coordinate transforms and fusion (pycozmo/coords.py,
pycozmo/world_map.py)."""

import numpy as np
import pytest

from pycozmo import coords
from pycozmo.world_map import WorldMap


class TestCoords:
    def test_camera_to_world_straight_ahead(self):
        # Cube 200 mm straight ahead, level head, robot at origin facing +Y.
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(p_cam, np.eye(3), 0.0, np.zeros(3), 0.0)
        # Body forward is +Y; camera sits 40 mm above ground.
        np.testing.assert_allclose(pw, [0.0, 200.0, 40.0], atol=1e-9)

    def test_camera_to_world_robot_translation(self):
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(
            p_cam, np.eye(3), 0.0, np.array([100.0, 0.0, 0.0]), 0.0
        )
        np.testing.assert_allclose(pw, [100.0, 200.0, 40.0], atol=1e-9)

    def test_camera_to_world_robot_yaw(self):
        # Robot turns left 90 deg; the cube now lies to its left (-X).
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(p_cam, np.eye(3), 0.0, np.zeros(3), np.pi / 2)
        np.testing.assert_allclose(pw, [-200.0, 0.0, 40.0], atol=1e-9)

    def test_world_to_scene_point(self):
        # Path on the ground stays at scene y=0; no +40 lift hack.
        scene = coords.world_to_scene_point([10.0, 20.0, 0.0])
        np.testing.assert_allclose(scene, [-10.0, 0.0, 20.0], atol=1e-9)

    def test_world_yaw_mirrors_to_scene(self):
        assert coords.world_yaw_to_scene(0.0) == pytest.approx(0.0)
        assert coords.world_yaw_to_scene(np.pi / 2) == pytest.approx(-np.pi / 2)


class TestWorldMapFusion:
    def test_cube_placed_in_world_frame(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
        pos = wm.get_cubes()[0].position
        np.testing.assert_allclose(pos, [0.0, 200.0, 40.0], atol=1e-6)

    def test_confidence_below_floor_dropped(self):
        wm = WorldMap(confidence_floor=0.5)
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.2, 0.0)
        assert 0 not in wm.get_cubes()

    def test_forget_cube_removes_it(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
        wm.forget_cube(0)
        assert 0 not in wm.get_cubes()

    def test_origin_change_resets_map(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
        wm.update_robot_pose([5.0, 6.0, 0.0], 0.0, origin_id=2, record_path=True)
        # Stale cubes are dropped; the stale trail is wiped (only the new
        # origin's current point remains).
        assert wm.get_cubes() == {}
        assert len(wm.get_path_points()) == 1
        np.testing.assert_allclose(wm.get_path_points()[0], [5.0, 6.0, 0.0])

    def test_path_recording_and_reset(self):
        wm = WorldMap()
        wm.update_robot_pose([1.0, 2.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_robot_pose([3.0, 4.0, 0.0], 0.0, origin_id=1, record_path=True)
        assert len(wm.get_path_points()) == 2
        wm.reset_path()
        assert wm.get_path_points() == []
