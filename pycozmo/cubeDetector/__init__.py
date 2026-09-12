"""
Cube detection and tracking module.

This module provides advanced 3D cube detection and tracking capabilities
using ArUco-like markers and sensor fusion.
"""

from .cubeDetector import CozmoCubeDetector, CubeFusionTracker

__all__ = [
    "CozmoCubeDetector",
    "CubeFusionTracker",
]
