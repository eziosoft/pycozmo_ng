"""Detection interfaces — abstractions for swappable detectors."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pycozmo_ng.vision.calibration.robot_center import RobotCenterResult


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


class ChargerDetector(Protocol):
    """Protocol for charger marker detectors (COZMO-65)."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        markers: list[np.ndarray],
        **kwargs,
    ) -> None: ...

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """Detect the charger's rectangular marker in an image.

        Returns:
            List of dicts with keys: score, corners, position (camera frame
            mm), rotation_matrix, kind == "charger". No cube identity keys —
            the cube index space stays 0-2.
        """
        ...


class ArucoDetector(Protocol):
    """Protocol for standard 4x4 ArUco marker detectors."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        distortion_coefficients: np.ndarray,
        **kwargs,
    ) -> None: ...

    def detect(self, img_bgr: np.ndarray) -> list:
        """Detect ArUco markers in an image.

        Returns:
            List of ArucoDetection — marker_id, size_mm, position_mm
            (camera frame), rotation_matrix, corners. Multiple markers per
            frame; the marker ID is the identity (never a cube index).
        """
        ...


class RobotCenterDetector(Protocol):
    """Protocol for robot center calibration detectors."""

    head_angle_rad: float

    def detect(self, img_bgr: np.ndarray) -> RobotCenterResult | None:
        """Detect robot center circles. Returns result with detected, lateral_offset_mm, circle_a_px, circle_b_px."""
        ...
