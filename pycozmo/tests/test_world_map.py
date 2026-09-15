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

    def test_head_tilt_matrix_is_proper_rotation(self):
        # The camera->body pitch matrix must be orthonormal with det=+1.
        # Regression: the Yc column had the wrong sign (-sin instead of +sin),
        # making the matrix a shear (det=cos(2*theta)) for theta != 0 — a
        # level cube then came out tilted by roughly the head angle.
        for theta in (0.0, np.pi / 6, -np.pi / 6, np.pi / 3):
            M = coords._head_tilt_matrix(theta)
            np.testing.assert_allclose(
                M @ M.T, np.eye(3), atol=1e-9, err_msg=f"theta={theta}"
            )
            np.testing.assert_allclose(np.linalg.det(M), 1.0, atol=1e-9)

    def test_cube_orientation_invariant_to_head_tilt(self):
        # A level cube (yaw 20 deg) must recover that exact world orientation
        # regardless of the head pitch. Regression: the shear in the head-tilt
        # matrix left a residual tilt proportional to head angle, so the cube
        # appeared to rock with the head.
        yaw = np.deg2rad(20.0)
        c, s = np.cos(yaw), np.sin(yaw)
        r_world = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        p_cam = np.array([0.0, 40.0, 300.0])
        for head in (0.0, np.deg2rad(10), np.deg2rad(-20), np.deg2rad(30)):
            # Inverse of the pipeline, all the way back to camera frame.
            r_cam = np.linalg.inv(coords._head_tilt_matrix(head)) @ r_world
            _, rot_world = coords.camera_to_world(p_cam, r_cam, head, np.zeros(3), 0.0)
            np.testing.assert_allclose(
                rot_world, r_world, atol=1e-9, err_msg=f"head={head}"
            )


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


class TestWorldMapRotationAlpha:
    """Tests for separate rotation smoothing (rotation_alpha)."""

    def _rotation_angle_error(self, r1: np.ndarray, r2: np.ndarray) -> float:
        """Angle (degrees) between two rotation matrices."""
        from scipy.spatial.transform import Rotation as R

        return float(
            np.degrees((R.from_matrix(r1) * R.from_matrix(r2).inv()).magnitude())
        )

    def test_first_measurement_no_smoothing(self):
        """First measurement is stored as-is (after world transform)."""
        wm = WorldMap(smoothing_alpha=0.3, rotation_alpha=0.12)
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        r = np.eye(3)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], r, 0.9, 0.0)
        # Expected world-frame rotation for identity camera rotation at
        # head=0, robot at origin, yaw=0.
        _, rot_world = coords.camera_to_world(
            [0.0, 0.0, 200.0], r, 0.0, np.zeros(3), 0.0
        )
        np.testing.assert_allclose(
            wm.get_cubes()[0].rotation_matrix, rot_world, atol=1e-6
        )

    def test_rotation_uses_rotation_alpha(self):
        """Rotation blend uses rotation_alpha, not smoothing_alpha."""
        wm_high_pos = WorldMap(smoothing_alpha=0.9, rotation_alpha=0.12)
        wm_low_rot = WorldMap(smoothing_alpha=0.9, rotation_alpha=0.12)
        # Both have same rotation_alpha, so rotation convergence should match
        # even if smoothing_alpha differs (position alpha is irrelevant to rotation).

        # Initial measurement: identity rotation
        r_init = np.eye(3)
        for wm in (wm_high_pos, wm_low_rot):
            wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
            wm.update_cube_measurement(0, [0.0, 0.0, 200.0], r_init, 0.9, 0.0)

        # Second measurement: 30° yaw
        yaw = np.deg2rad(30)
        r_new = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
            dtype=np.float64,
        )

        wm_high_pos.update_cube_measurement(0, [0.0, 0.0, 200.0], r_new, 0.9, 0.0)
        wm_low_rot.update_cube_measurement(0, [0.0, 0.0, 200.0], r_new, 0.9, 0.0)

        # Both should have identical rotation (same rotation_alpha)
        err = self._rotation_angle_error(
            wm_high_pos.get_cubes()[0].rotation_matrix,
            wm_low_rot.get_cubes()[0].rotation_matrix,
        )
        assert err < 0.01, (
            f"rotation mismatch with different smoothing_alpha: {err:.3f}°"
        )

    def test_lower_rotation_alpha_slower_convergence(self):
        """Lower rotation_alpha produces slower rotation convergence."""
        wm_fast = WorldMap(smoothing_alpha=0.3, rotation_alpha=0.20)
        wm_slow = WorldMap(smoothing_alpha=0.3, rotation_alpha=0.10)

        yaw = np.deg2rad(30)
        r_new = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
            dtype=np.float64,
        )

        for wm in (wm_fast, wm_slow):
            wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
            wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)

        # Apply 5 identical measurements
        for _ in range(5):
            wm_fast.update_cube_measurement(0, [0.0, 0.0, 200.0], r_new, 0.9, 0.0)
            wm_slow.update_cube_measurement(0, [0.0, 0.0, 200.0], r_new, 0.9, 0.0)

        err_fast = self._rotation_angle_error(
            wm_fast.get_cubes()[0].rotation_matrix, r_new
        )
        err_slow = self._rotation_angle_error(
            wm_slow.get_cubes()[0].rotation_matrix, r_new
        )
        assert err_fast < err_slow, (
            f"higher alpha should converge faster: fast_err={err_fast:.2f}° "
            f">= slow_err={err_slow:.2f}°"
        )

    def test_position_still_uses_smoothing_alpha(self):
        """Position blending still uses smoothing_alpha (not rotation_alpha)."""
        wm = WorldMap(smoothing_alpha=0.5, rotation_alpha=0.10)
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)

        # Second measurement: cube deeper (camera z: 200 -> 300).
        wm.update_cube_measurement(0, [0.0, 0.0, 300.0], np.eye(3), 0.9, 0.0)

        pos = wm.get_cubes()[0].position
        # Camera depth maps to world X (forward); Y stays 0, Z stays 38 (camera
        # height). World X: 200+30=230 (first), 300+30=330 (second).
        # blend = 0.5 * 0.9 = 0.45 -> x = 0.45*330 + 0.55*230 = 275.0
        expected = np.array([275.0, 0.0, 38.0])
        np.testing.assert_allclose(pos, expected, atol=0.1)

    def test_default_rotation_alpha(self):
        """Default rotation_alpha is 0.12."""
        wm = WorldMap()
        assert wm.rotation_alpha == pytest.approx(0.12)

    def test_custom_rotation_alpha(self):
        """Custom rotation_alpha is stored."""
        wm = WorldMap(rotation_alpha=0.05)
        assert wm.rotation_alpha == pytest.approx(0.05)
