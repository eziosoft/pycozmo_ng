"""Detection interfaces — abstractions for swappable detectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CubeDetector(Protocol):
    """Protocol for cube detectors — any implementation can be plugged in."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        markers: list[np.ndarray],
        **kwargs,
    ) -> None: ...

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Detect marker faces in an image.

        Returns:
            List of dicts with keys: cube_index (world index 0-2),
            corners, pose, confidence, etc.
        """
        ...


class CubeFusionTracker(Protocol):
    """Protocol for cube fusion trackers."""

    def __init__(self, cube_size_mm: float = 44.0, **kwargs) -> None: ...

    def update(self, detections: list[dict]) -> dict[int, dict]:
        """Fuse detections into coherent cube poses.

        Returns:
            Dict mapping cube_index (world index 0-2) -> fused state with
            position, rotation_matrix, confidence, num_faces, rotation_source.
        """
        ...


class FaceDetector(Protocol):
    """Protocol for face detectors."""

    def __init__(self, camera_calibration) -> None: ...

    def detect(self, img_bgr: np.ndarray) -> list:
        """Detect faces in image. Returns list of FaceDetection objects."""
        ...


class RobotCenterDetector(Protocol):
    """Protocol for robot center calibration detectors."""

    head_angle_rad: float

    def detect(self, img_bgr: np.ndarray) -> object | None:
        """Detect robot center circles. Returns result with detected, lateral_offset_mm, circle_a_px, circle_b_px."""
        ...
