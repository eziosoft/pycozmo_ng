"""ArucoDetection — one detected ArUco marker in a camera frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArucoDetection:
    """Represents a detected standard 4x4 ArUco marker in the camera image.

    Unlike cubes there is no cross-frame identity — the marker ID *is* the
    identity (it never changes for a printed marker). Multiple markers can be
    visible in one frame, so events diff the set of visible IDs.
    """

    #: ArUco marker ID (from the printed dictionary, e.g. 0-49 for 4X4_50).
    marker_id: int

    #: Physical side length (mm) used for this marker's PnP solve — the
    #: per-ID configured size (falls back to the config default).
    size_mm: float

    # Bounding box (x, y, width, height) in pixels
    x: int
    y: int
    width: int
    height: int

    # Decoder confidence (1.0 for ArUco — the bit pattern is either valid
    # or the marker is rejected; kept for call-site symmetry with cubes).
    confidence: float = 1.0

    # Estimated distance from camera (in meters)
    distance: float = 0.0

    # Marker centre in camera frame (mm) — fed to WorldMap.update_aruco_measurement
    position_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # 3x3 marker rotation in camera frame (marker plane basis from PnP,
    # z = marker normal) — fused into the world-map marker pose.
    rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Pixel corners (4, 2), TL/TR/BR/BL — for annotation overlays.
    corners: np.ndarray = field(default_factory=lambda: np.zeros((4, 2)))

    # Timestamp of detection
    timestamp: float = 0.0


__all__ = ["ArucoDetection"]
