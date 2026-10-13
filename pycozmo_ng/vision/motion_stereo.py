"""Motion stereo — a world-frame point cloud from the robot's own motion.

Turns Cozmo's single forward camera into a crude stereo rig by using the
robot's motion as the baseline: features detected in a keyframe image are
re-tracked in later frames with Lucas-Kanade optical flow, and once the
robot has translated far enough that the two views form a usable baseline,
each tracked feature is triangulated (mid-point of the two camera rays)
into a 3-D point in the world frame.

The cloud accumulates in a voxel grid (one running-mean point per voxel)
so it stays bounded while the robot explores, and is published to the web
viewer through the ``FrameSink`` seam (``push_point_cloud``), where it
renders as ``THREE.Points`` in the 3-D view.

Accuracy is deliberately modest: the baseline is odometry-scale (tens of
millimetres), depth noise grows quadratically with range, and the world
pose carries localization drift — this is a live "shape sketch" of the
room for the 3-D view, not a metric map. Points below the floor or behind
either camera are rejected; everything else is drawn as measured.

Frame conventions follow ``util/coords.py``: rays are built in the camera
frame (X right, Y down, Z forward), then rotated into the world frame
(X forward, Y left, Z up) with the head tilt and robot yaw applied.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from pycozmo_ng.util.coords import CAMERA_TO_BODY_OFFSET, R_yaw, _head_tilt_matrix

__all__ = ["MotionStereo"]


class MotionStereo:
    """Two-view 3-D reconstruction from the robot's own translation.

    Call :meth:`update` once per processed camera frame with the grayscale
    image and the robot's world pose; the module owns the keyframe, the
    feature tracking, and the accumulated cloud. :meth:`get_points` returns
    a copy of the cloud for display — safe to call from another thread.

    Args:
        camera_matrix: 3×3 intrinsic matrix K from CameraCalibration.
        min_baseline_mm: Camera translation (mm) required between the
            keyframe and the current frame before features are
            triangulated. Smaller baselines give unusable depth.
        min_disparity_px: Minimum image-space motion (px) of a tracked
            feature for its triangulation to be kept.
        max_separation_mm: Maximum distance (mm) between the two rays'
            closest approach for a triangulation to be kept — larger
            values mean a bad correspondence.
        min_depth_mm / max_depth_mm: Accepted distance (mm) of a point
            from the keyframe camera.
        min_height_mm: Accepted world Z (mm); points below the floor are
            triangulation noise and are dropped.
        max_features: Feature budget detected per keyframe.
        min_features: Tracking survivor count below which the keyframe is
            restarted instead of triangulated.
        max_keyframe_rotation_rad: Rotation (rad, camera-to-camera) that
            forces a keyframe restart even without enough translation —
            large rotations degrade LK tracking.
        voxel_mm: Cube edge (mm) of the dedup grid; one point per voxel.
        max_points: Cap on cloud size; the oldest voxels are evicted
            first (dict insertion order).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        *,
        min_baseline_mm: float = 40.0,
        min_disparity_px: float = 2.0,
        max_separation_mm: float = 40.0,
        min_depth_mm: float = 60.0,
        max_depth_mm: float = 1600.0,
        min_height_mm: float = -25.0,
        max_features: int = 100,
        min_features: int = 15,
        max_keyframe_rotation_rad: float = 0.45,
        voxel_mm: float = 25.0,
        max_points: int = 2500,
    ) -> None:
        k = np.asarray(camera_matrix, dtype=np.float64)
        self._fx = float(k[0, 0])
        self._fy = float(k[1, 1])
        self._cx = float(k[0, 2])
        self._cy = float(k[1, 2])

        self.min_baseline_mm = float(min_baseline_mm)
        self.min_disparity_px = float(min_disparity_px)
        self.max_separation_mm = float(max_separation_mm)
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.min_height_mm = float(min_height_mm)
        self.max_features = int(max_features)
        self.min_features = int(min_features)
        self.max_keyframe_rotation_rad = float(max_keyframe_rotation_rad)
        self.voxel_mm = float(voxel_mm)
        self.max_points = int(max_points)

        # Keyframe: grayscale image, tracked pixels (N,1,2) float32, pose.
        self._key_gray: np.ndarray | None = None
        self._key_pixels: np.ndarray | None = None
        self._key_center: np.ndarray | None = None
        self._key_rotation: np.ndarray | None = None

        # Accumulated cloud: voxel key -> world point (running mean).
        self._cloud: dict[tuple[int, int, int], np.ndarray] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop the keyframe and the cloud (e.g. after the world origin moved)."""
        with self._lock:
            self._cloud.clear()
        self._key_gray = None
        self._key_pixels = None
        self._key_center = None
        self._key_rotation = None

    def get_points(self) -> np.ndarray:
        """Copy of the accumulated cloud as an (N, 3) array of world points (mm)."""
        with self._lock:
            if not self._cloud:
                return np.zeros((0, 3), dtype=np.float64)
            return np.array(list(self._cloud.values()), dtype=np.float64)

    @property
    def point_count(self) -> int:
        with self._lock:
            return len(self._cloud)

    def update(
        self,
        gray: np.ndarray,
        head_angle_rad: float,
        robot_position,
        robot_yaw_rad: float,
    ) -> bool:
        """Process one camera frame; return True if the cloud changed.

        Args:
            gray: Grayscale uint8 camera frame.
            head_angle_rad: Current head tilt (positive = looking up).
            robot_position: Robot world position (mm), e.g. from
                ``WorldMap.get_robot_pose()``.
            robot_yaw_rad: Robot world yaw about Z (rad).
        """
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        center, rotation = self._camera_pose(head_angle_rad, robot_position, robot_yaw_rad)

        key_pixels = self._key_pixels
        key_center = self._key_center
        key_rotation = self._key_rotation
        if self._key_gray is None or key_pixels is None or key_center is None or key_rotation is None:
            self._reset_keyframe(gray, center, rotation)
            return False

        pixels_key, pixels_cur = self._track(gray)
        if pixels_key is None or pixels_cur is None:
            self._reset_keyframe(gray, center, rotation)
            return False

        baseline_mm = float(np.linalg.norm(center - key_center))
        rotation_angle = self._rotation_angle(rotation, key_rotation)
        if baseline_mm < self.min_baseline_mm:
            # Not enough parallax yet: keep the same keyframe so the
            # baseline can accumulate as the robot keeps moving — but
            # restart if the view has swung too far for LK to hold on.
            if rotation_angle > self.max_keyframe_rotation_rad:
                self._reset_keyframe(gray, center, rotation)
            return False

        added = self._triangulate_and_add(
            pixels_key,
            pixels_cur,
            key_center,
            key_rotation,
            center,
            rotation,
        )
        self._reset_keyframe(gray, center, rotation)
        return added

    # ------------------------------------------------------------------
    # Camera pose
    # ------------------------------------------------------------------

    def _camera_pose(
        self,
        head_angle_rad: float,
        robot_position,
        robot_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World-frame camera centre (mm) and camera→world rotation."""
        r_yaw = R_yaw(float(robot_yaw_rad))
        rotation = r_yaw @ _head_tilt_matrix(float(head_angle_rad))
        center = r_yaw @ CAMERA_TO_BODY_OFFSET + np.asarray(robot_position, dtype=np.float64)
        return center, rotation

    @staticmethod
    def _rotation_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        """Angle (rad) between two camera→world rotations."""
        cos_angle = (float(np.trace(rotation_a @ rotation_b.T)) - 1.0) / 2.0
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # ------------------------------------------------------------------
    # Feature tracking
    # ------------------------------------------------------------------

    def _reset_keyframe(self, gray: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> None:
        self._key_gray = gray
        self._key_center = center
        self._key_rotation = rotation
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=0.01,
            minDistance=8,
            blockSize=7,
        )
        self._key_pixels = points if points is not None else None

    def _track(self, gray: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Track keyframe features into ``gray``; (pixels_key, pixels_cur) or (None, None)."""
        key_gray = self._key_gray
        key_pixels = self._key_pixels
        if key_gray is None or key_pixels is None or len(key_pixels) < self.min_features:
            return None, None

        # nextPts is the LK output; without OPTFLOW_USE_INITIAL_FLOW the
        # passed array is overwritten (prevPts is copied in as the estimate).
        next_pixels, status, _ = cv2.calcOpticalFlowPyrLK(
            key_gray,
            gray,
            key_pixels,
            key_pixels.copy(),
            winSize=(21, 21),
            maxLevel=3,
        )
        if next_pixels is None or status is None:
            return None, None

        status = status.reshape(-1).astype(bool)
        key = key_pixels.reshape(-1, 2)
        cur = next_pixels.reshape(-1, 2)
        key, cur = key[status], cur[status]

        # Discard features that slid out of the frame.
        h, w = gray.shape[:2]
        inside = (cur[:, 0] >= 2) & (cur[:, 0] < w - 2) & (cur[:, 1] >= 2) & (cur[:, 1] < h - 2)
        key, cur = key[inside], cur[inside]

        if len(key) < self.min_features:
            return None, None
        return key, cur

    # ------------------------------------------------------------------
    # Triangulation
    # ------------------------------------------------------------------

    def _rays_world(
        self,
        pixels: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Unit ray directions (world frame) through the given pixels."""
        rays = np.stack(
            [
                (pixels[:, 0] - self._cx) / self._fx,
                (pixels[:, 1] - self._cy) / self._fy,
                np.ones(len(pixels)),
            ],
            axis=1,
        )
        rays = rays @ rotation.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def _triangulate_and_add(
        self,
        pixels_key: np.ndarray,
        pixels_cur: np.ndarray,
        center_key: np.ndarray,
        rotation_key: np.ndarray,
        center_cur: np.ndarray,
        rotation_cur: np.ndarray,
    ) -> bool:
        """Triangulate tracked features and merge survivors into the voxel cloud."""
        # Image-space motion gate: a feature that barely moved cannot
        # triangulate meaningfully at this baseline.
        disparity = np.linalg.norm(pixels_cur - pixels_key, axis=1)
        keep = disparity >= self.min_disparity_px
        if not np.any(keep):
            return False
        pixels_key, pixels_cur = pixels_key[keep], pixels_cur[keep]

        d_key = self._rays_world(pixels_key, rotation_key)
        d_cur = self._rays_world(pixels_cur, rotation_cur)

        # Mid-point triangulation: minimise |C_k + t_k d_k - (C_c + t_c d_c)|.
        # With unit rays: t_k = (b e - d) / (1 - b^2), t_c = e + b t_k,
        # where b = d_k·d_c, d = d_k·w0, e = d_c·w0, w0 = C_k - C_c.
        w0 = center_key - center_cur
        b = np.einsum("ij,ij->i", d_key, d_cur)
        d = d_key @ w0
        e = d_cur @ w0
        denom = 1.0 - b * b
        valid = np.abs(denom) > 1e-9
        t_key = np.where(valid, (b * e - d) / np.where(valid, denom, 1.0), -1.0)
        t_cur = e + b * t_key

        p_key = center_key + t_key[:, None] * d_key
        p_cur = center_cur + t_cur[:, None] * d_cur
        points = (p_key + p_cur) / 2.0
        separation = np.linalg.norm(p_key - p_cur, axis=1)
        depth = t_key  # unit rays: t is distance from the keyframe camera

        accepted = (
            valid
            & (t_key > 0)
            & (t_cur > 0)
            & (separation < self.max_separation_mm)
            & (depth >= self.min_depth_mm)
            & (depth <= self.max_depth_mm)
            & (points[:, 2] >= self.min_height_mm)
        )
        if not np.any(accepted):
            return False
        return self._add_points(points[accepted])

    def _add_points(self, points: np.ndarray) -> bool:
        """Merge points into the voxel cloud (running mean per voxel)."""
        keys = np.floor(points / self.voxel_mm).astype(np.int64)
        changed = False
        with self._lock:
            for point, key in zip(points, map(tuple, keys), strict=True):
                existing = self._cloud.get(key)
                if existing is None:
                    if len(self._cloud) >= self.max_points:
                        # Oldest voxel first — dict iteration order.
                        self._cloud.pop(next(iter(self._cloud)))
                    self._cloud[key] = point.copy()
                else:
                    self._cloud[key] = (existing + point) / 2.0
                changed = True
        return changed
