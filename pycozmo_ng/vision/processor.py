"""Legacy vision processor module — deprecated.

This module is kept for backward compatibility. Use the new modules instead:

- :mod:`pycozmo_ng.vision.detection_pipeline` — core detection logic
- :mod:`pycozmo_ng.vision.annotation` — drawing/annotation functions
- :mod:`pycozmo_ng.vision.events` — event dispatching

"""

import warnings

from pycozmo_ng.vision.cube_detector.detection_pipeline import CubeDetection, DetectionPipeline

warnings.warn(
    "pycozmo_ng.vision.processor is deprecated. "
    "Use pycozmo_ng.vision.detection_pipeline, pycozmo_ng.vision.annotation, "
    "and pycozmo_ng.vision.events instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Alias for backward compatibility
VisionProcessor = DetectionPipeline

__all__ = [
    "CubeDetection",
    "DetectionPipeline",
    "VisionProcessor",
]