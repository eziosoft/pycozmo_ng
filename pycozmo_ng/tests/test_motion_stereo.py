"""Tests for MotionStereo — 3-D point cloud from the robot's own motion.

The module's contract: features tracked between two views whose cameras are
known (from the world-map pose) are triangulated into world-frame points,
voxel-deduped, and bounded in size. Tests cross the public seam
(``update``/``get_points``/``reset``) with synthetic image pairs whose
ground-truth geometry is known — no robot, no live camera.
"""

import cv2
import numpy as np
import pytest

from pycozmo_ng.util.coords import CAMERA_TO_BODY_OFFSET, R_yaw, _head_tilt_matrix
from pycozmo_ng.vision.motion_stereo import MotionStereo

# Intrinsic-ish matrix matching the real camera (320x240 frames).
K = np.array([[305.0, 0.0, 160.0], [0.0, 305.0, 126.0], [0.0, 0.0, 1.0]])


def camera_pose(head_angle_rad: float, robot_position, robot_yaw_rad: float):
    """World camera centre + camera->world rotation (same math as MotionStereo)."""
    r_yaw = R_yaw(robot_yaw_rad)
    rotation = r_yaw @ _head_tilt_matrix(head_angle_rad)
    center = r_yaw @ CAMERA_TO_BODY_OFFSET + np.asarray(robot_position, dtype=np.float64)
    return center, rotation


def project(points_world, center, rotation):
    """Project world points (N,3) into pixels from the given camera pose."""
    cam = (np.asarray(points_world) - center) @ rotation  # camera frame
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    v = K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]
    return np.stack([u, v], axis=1)


def render_dots(pixels, shape=(240, 320), seed=7):
    """Draw distinguishable dots (varying size + brightness) at the pixels.

    Uniform dots on a regular grid are ambiguous for optical flow — LK
    locks onto the wrong neighbour. Jittered positions and per-dot
    appearance give each local patch a unique signature, like real texture.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.uint8)
    for u, v in pixels:
        radius = int(rng.integers(2, 5))
        brightness = int(rng.integers(120, 256))
        cv2.circle(img, (round(u), round(v)), radius, brightness, -1)
    return img


def make_scene_points():
    """A grid of 3-D points on a fronto-parallel wall ~450 mm ahead.

    The wall spans a wide lateral range so dots project well apart in the
    image (optical flow needs distinctive, well-separated texture — real
    rooms have it; a dense synthetic grid does not). Depth and baseline
    are chosen so image disparity stays in the range LK tracking handles
    on 320x240 frames (~40-70 px), mirroring a real room seen from a
    robot that moves in 40-80 mm steps.
    """
    ys = np.linspace(-200.0, -60.0, 8)
    zs = np.linspace(-15.0, 70.0, 7)
    grid = np.stack(np.meshgrid(ys, zs, indexing="ij"), axis=-1).reshape(-1, 2)
    return np.column_stack([np.full(len(grid), 450.0), grid[:, 0], grid[:, 1]])


def two_view_scene(baseline_mm=60.0, head_angle_rad=0.0):
    """Rendered frame pair + poses for the robot translated along -Y (right)."""
    points = make_scene_points()
    pose_a = camera_pose(head_angle_rad, np.array([0.0, 0.0, 0.0]), 0.0)
    pose_b = camera_pose(head_angle_rad, np.array([0.0, -baseline_mm, 0.0]), 0.0)
    img_a = render_dots(project(points, *pose_a))
    img_b = render_dots(project(points, *pose_b))
    return points, img_a, img_b, pose_a, pose_b


class TestTwoViewTriangulation:
    def test_recovers_wall_points(self):
        """Points triangulated from the two views land at the true 3-D positions."""
        points, img_a, img_b, _, _ = two_view_scene()
        stereo = MotionStereo(K, min_baseline_mm=40.0)

        stereo.update(img_a, 0.0, np.array([0.0, 0.0, 0.0]), 0.0)
        stereo.update(img_b, 0.0, np.array([0.0, -60.0, 0.0]), 0.0)

        cloud = stereo.get_points()
        assert len(cloud) > 10

        # Every true point should be within ~25 mm of some cloud point
        # (voxel size is 25 mm, so exact equality is not the contract).
        distances = np.linalg.norm(cloud[:, None, :] - points[None, :, :], axis=2)
        assert np.median(distances.min(axis=1)) < 20.0

    def test_cloud_points_are_on_the_wall_plane(self):
        """Recovered points cluster around the wall's world X (depth) position."""
        _, img_a, img_b, _, _ = two_view_scene()
        stereo = MotionStereo(K)
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)
        stereo.update(img_b, 0.0, np.array([0.0, -60.0, 0.0]), 0.0)

        cloud = stereo.get_points()
        # Wall is at x = 450; depth noise spreads points along the ray
        # (mostly world X at yaw 0), so the median must still sit close.
        assert np.abs(np.median(cloud[:, 0]) - 450.0) < 40.0

    def test_head_tilt_is_applied(self):
        """A tilted camera triangulates to the same world points."""
        points, img_a, img_b, _, _ = two_view_scene(head_angle_rad=np.radians(10.0))
        stereo = MotionStereo(K)
        stereo.update(img_a, np.radians(10.0), np.zeros(3), 0.0)
        stereo.update(img_b, np.radians(10.0), np.array([0.0, -60.0, 0.0]), 0.0)

        cloud = stereo.get_points()
        assert len(cloud) > 5
        distances = np.linalg.norm(cloud[:, None, :] - points[None, :, :], axis=2)
        assert np.median(distances.min(axis=1)) < 25.0

    def test_robot_yaw_is_applied(self):
        """The same scene seen after a yaw rotation lands in the same world place."""
        points = make_scene_points()
        stereo = MotionStereo(K)

        # Robot at origin, yaw 0: render the wall ahead.
        pose_a = camera_pose(0.0, np.zeros(3), 0.0)
        img_a = render_dots(project(points, pose_a[0], pose_a[1]))
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)

        # Robot turned 15 deg left and shifted right: same wall, new pose.
        yaw = np.radians(15.0)
        pose_b = camera_pose(0.0, np.array([20.0, -60.0, 0.0]), yaw)
        img_b = render_dots(project(points, pose_b[0], pose_b[1]))
        stereo.update(img_b, 0.0, np.array([20.0, -60.0, 0.0]), yaw)

        cloud = stereo.get_points()
        assert len(cloud) > 5
        distances = np.linalg.norm(cloud[:, None, :] - points[None, :, :], axis=2)
        assert np.median(distances.min(axis=1)) < 30.0


class TestBaselineGating:
    def test_no_points_below_baseline(self):
        """Below min_baseline_mm the keyframe waits — no triangulation."""
        _, img_a, _, _, _ = two_view_scene()
        stereo = MotionStereo(K, min_baseline_mm=40.0)
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)

        # Robot moved only 10 mm — no points yet.
        changed = stereo.update(img_a, 0.0, np.array([0.0, -10.0, 0.0]), 0.0)
        assert changed is False
        assert stereo.point_count == 0

    def test_baseline_accumulates_across_frames(self):
        """The keyframe is kept while the baseline is insufficient."""
        _, img_a, img_b, _, _ = two_view_scene()
        stereo = MotionStereo(K, min_baseline_mm=40.0)
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)
        assert stereo.update(img_b, 0.0, np.array([0.0, -20.0, 0.0]), 0.0) is False
        # Same keyframe + same target frame now has a 80 mm baseline.
        assert stereo.update(img_b, 0.0, np.array([0.0, -60.0, 0.0]), 0.0) is True

    def test_first_frame_never_triangulates(self):
        stereo = MotionStereo(K)
        img = np.zeros((240, 320), dtype=np.uint8)
        cv2.circle(img, (160, 120), 4, 255, -1)
        assert stereo.update(img, 0.0, np.zeros(3), 0.0) is False
        assert stereo.point_count == 0


class TestCloudManagement:
    def _fill(self, stereo: MotionStereo, n_poses=3):
        _, img_a, img_b, _, _ = two_view_scene()
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)
        stereo.update(img_b, 0.0, np.array([0.0, -60.0, 0.0]), 0.0)
        return stereo.point_count

    def test_voxel_dedup_bounds_growth(self):
        """Re-triangulating the same scene does not grow the cloud unboundedly."""
        stereo = MotionStereo(K, voxel_mm=25.0)
        first = self._fill(stereo)
        # Repeat the same two views several times from fresh keyframes.
        for _ in range(3):
            _, img_a, img_b, _, _ = two_view_scene()
            stereo.update(img_a, 0.0, np.zeros(3), 0.0)
            stereo.update(img_b, 0.0, np.array([0.0, -60.0, 0.0]), 0.0)
        assert stereo.point_count <= first * 1.5

    def test_max_points_cap(self):
        """The cloud never exceeds max_points."""
        stereo = MotionStereo(K, voxel_mm=5.0, max_points=50)
        self._fill(stereo)
        assert stereo.point_count <= 50

    def test_reset_clears_everything(self):
        stereo = MotionStereo(K)
        self._fill(stereo)
        assert stereo.point_count > 0
        stereo.reset()
        assert stereo.point_count == 0
        assert stereo.get_points().shape == (0, 3)

    def test_get_points_returns_copy(self):
        stereo = MotionStereo(K)
        self._fill(stereo)
        points = stereo.get_points()
        points[:] = 0.0
        assert np.any(stereo.get_points() != 0.0)


class TestPoseMath:
    def test_camera_pose_matches_coords_convention(self):
        """Camera centre = robot pos + R_yaw @ offset; rotation = yaw @ head."""
        stereo = MotionStereo(K)
        center, rotation = stereo._camera_pose(0.3, np.array([100.0, 50.0, 0.0]), 1.2)

        expected_center = R_yaw(1.2) @ CAMERA_TO_BODY_OFFSET + np.array([100.0, 50.0, 0.0])
        np.testing.assert_allclose(center, expected_center, atol=1e-9)
        np.testing.assert_allclose(rotation, R_yaw(1.2) @ _head_tilt_matrix(0.3), atol=1e-9)

    def test_rotation_is_proper(self):
        stereo = MotionStereo(K)
        _, rotation = stereo._camera_pose(0.5, np.zeros(3), 0.8)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


class TestFeatureTracking:
    def test_featureless_frames_add_nothing(self):
        """Blank frames (no features) never produce points or crash."""
        stereo = MotionStereo(K)
        blank = np.zeros((240, 320), dtype=np.uint8)
        assert stereo.update(blank, 0.0, np.zeros(3), 0.0) is False
        assert stereo.update(blank, 0.0, np.array([0.0, -60.0, 0.0]), 0.0) is False
        assert stereo.point_count == 0

    def test_rotation_without_translation_restarts_keyframe(self):
        """A large rotation with no baseline restarts instead of tracking."""
        _, img_a, _, _, _ = two_view_scene()
        stereo = MotionStereo(K, min_baseline_mm=40.0, max_keyframe_rotation_rad=0.45)
        stereo.update(img_a, 0.0, np.zeros(3), 0.0)
        # 45+ deg yaw swing, no translation: keyframe must restart.
        big_rotation = np.radians(60.0)
        assert stereo.update(img_a, 0.0, np.zeros(3), big_rotation) is False
        assert stereo.point_count == 0
