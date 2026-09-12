"""Tests for the world-map coordinate transforms and fusion (pycozmo/coords.py,
pycozmo/world_map.py)."""

import numpy as np
import pytest

from pycozmo import coords
from pycozmo.navigation.world_map import WorldMap


class TestCoords:
    def test_camera_to_world_straight_ahead(self):
        # Cube 200 mm straight ahead, level head, robot at origin facing +X.
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(p_cam, np.eye(3), 0.0, np.zeros(3), 0.0)
        # Body forward is +X (Anki), camera sits 38 mm above ground, 30 mm back.
        # Cube at 200+30=230 mm forward.
        np.testing.assert_allclose(pw, [230.0, 0.0, 38.0], atol=1e-9)

    def test_camera_to_world_robot_translation(self):
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(
            p_cam, np.eye(3), 0.0, np.array([100.0, 0.0, 0.0]), 0.0
        )
        np.testing.assert_allclose(pw, [330.0, 0.0, 38.0], atol=1e-9)

    def test_camera_to_world_robot_yaw(self):
        # Robot turns left 90 deg (CCW, +yaw); the cube ahead is now to its left.
        p_cam = np.array([0.0, 0.0, 200.0])
        pw, _ = coords.camera_to_world(p_cam, np.eye(3), 0.0, np.zeros(3), np.pi / 2)
        # Left is +Y world, so cube at +230 Y (200+30 camera setback).
        np.testing.assert_allclose(pw, [0.0, 230.0, 38.0], atol=1e-9)

    def test_world_to_scene_point(self):
        # World (X fwd, Y left, Z up) -> scene (X right, Y up, Z fwd).
        scene = coords.world_to_scene_point([10.0, 20.0, 30.0])
        np.testing.assert_allclose(scene, [20.0, 30.0, 10.0], atol=1e-9)

    def test_world_to_scene_preserves_handedness(self):
        # The world->scene map is a proper rotation (det=+1), so chirality is
        # preserved: an Anki left turn (+) must also be a scene left turn — a
        # clockwise drive-out in real life stays clockwise in the top-down 3D
        # view (regression: an earlier det=-1 reflection mirrored circles).
        for yaw in (0.0, np.pi / 4, np.pi / 2, -2.0):
            fwd_world = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            fwd_scene = coords.world_to_scene_point(fwd_world)
            fwd_scene = fwd_scene / np.linalg.norm(fwd_scene)
            # Reference: what a Three.js rotation.y=yaw turns +Z into.
            ref_scene = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
            np.testing.assert_allclose(fwd_scene, ref_scene, atol=1e-9)

    def test_world_to_scene_forward_is_scene_forward(self):
        # Robot driving forward (+X world) must render along scene +Z.
        fwd_world = np.array([1.0, 0.0, 0.0])
        fwd_scene = coords.world_to_scene_point(fwd_world)
        np.testing.assert_allclose(fwd_scene, [0.0, 0.0, 1.0], atol=1e-9)

    def test_world_yaw_to_scene_passes_through(self):
        # Yaw about world Z maps to an equal scene rotation.y: the basis change
        # is orientation-preserving (det=+1), so the yaw is NOT negated.
        assert coords.world_yaw_to_scene(0.0) == pytest.approx(0.0)
        assert coords.world_yaw_to_scene(np.pi / 2) == pytest.approx(np.pi / 2)
        assert coords.world_yaw_to_scene(-np.pi / 2) == pytest.approx(-np.pi / 2)


class TestWorldMapFusion:
    def test_cube_placed_in_world_frame(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
        pos = wm.get_cubes()[0].position
        np.testing.assert_allclose(pos, [230.0, 0.0, 38.0], atol=1e-6)

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

    def test_cube_relative_to_robot_stays_in_front_after_turn(self):
        # A cube observed straight ahead must render directly in front of the
        # robot box for ANY heading (camera sees only the front, COZMO complaint).
        for yaw in (0.0, np.pi / 4, np.pi / 2, -2.0):
            wm = WorldMap()
            wm.update_robot_pose([0.0, 0.0, 0.0], yaw, origin_id=1, record_path=False)
            wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
            cube_world = np.array(wm.get_cubes()[0].position)
            # Unit vector from robot to cube, in world frame.
            rel = cube_world[:2] / np.linalg.norm(cube_world[:2])
            # Robot forward in world (Anki) = (cos yaw, sin yaw) — +X is forward.
            fwd = np.array([np.cos(yaw), np.sin(yaw)])
            # The cube must be substantially "ahead" (dot positive, near 1).
            dot = float(np.dot(rel, fwd))
            assert dot > 0.9, f"yaw={yaw}: cube not in front (dot={dot:.2f})"

    def test_path_recording_and_reset(self):
        wm = WorldMap()
        wm.update_robot_pose([1.0, 2.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_robot_pose([3.0, 4.0, 0.0], 0.0, origin_id=1, record_path=True)
        assert len(wm.get_path_points()) == 2
        wm.reset_path()
        assert wm.get_path_points() == []
