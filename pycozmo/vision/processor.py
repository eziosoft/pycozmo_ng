"""Legacy vision processor module — deprecated.

This module is kept for backward compatibility. Use the new modules instead:

- :mod:`pycozmo.vision.detection_pipeline` — core detection logic
- :mod:`pycozmo.vision.annotation` — drawing/annotation functions
- :mod:`pycozmo.vision.events` — event dispatching

"""

import warnings

from pycozmo.vision.detection_pipeline import CubeDetection, DetectionPipeline

warnings.warn(
    "pycozmo.vision.processor is deprecated. "
    "Use pycozmo.vision.detection_pipeline, pycozmo.vision.annotation, "
    "and pycozmo.vision.events instead.",
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