"""Vision subpackage — processor, display, calibration, and detection modules."""

from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.camera_calibration import (
    CameraCalibration,
    CameraCalibrationRetriever,
)
from pycozmo.vision.detection_pipeline import (
    CubeDetection,
    DetectionPipeline,
    create_default_detection_pipeline,
    load_marker_templates,
)
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.interfaces import (
    CubeDetector,
    CubeFusionTracker,
    FaceDetector,
)
from pycozmo.vision.robot_center import RobotCenterDetector, RobotCenterResult

__all__ = [
    "AnnotationEngine",
    "CameraCalibration",
    "CameraCalibrationRetriever",
    "CubeDetection",
    "CubeDetector",
    "CubeFusionTracker",
    "DetectionEventDispatcher",
    "DetectionPipeline",
    "FaceDetector",
    "RobotCenterDetector",
    "RobotCenterResult",
    "create_default_detection_pipeline",
    "load_marker_templates",
]
