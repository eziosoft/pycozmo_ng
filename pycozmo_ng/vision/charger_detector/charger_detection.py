"""ChargerDetection — one detected charger marker in a camera frame."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ChargerDetection:
    """Represents the detected charger marker in the camera image.

    The charger is a single static object, so unlike cubes there is no
    per-object index and no cross-frame identity — the dispatcher only
    tracks presence/absence (COZMO-65).
    """

    # Bounding box (x, y, width, height) in pixels
    x: int
    y: int
    width: int
    height: int

    # Matching score (0.0 to 1.0)
    confidence: float

    # Estimated distance from robot (in meters)
    distance: float

    # Marker centre in camera frame (mm) — fed to WorldMap.update_charger_measurement
    position_mm: np.ndarray

    # Timestamp of detection
    timestamp: float = 0.0


__all__ = ["ChargerDetection"]
