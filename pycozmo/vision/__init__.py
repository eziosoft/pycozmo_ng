"""Vision subpackage — processor, display, calibration, and detection modules."""

from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.camera_calibration import (
    CameraCalibration,
    CameraCalibrationRetriever,
)
from pycozmo.vision.detection_pipeline import CubeDetection, DetectionPipeline
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.robot_center import RobotCenterDetector, RobotCenterResult

__all__ = [
    "AnnotationEngine",
    "CameraCalibration",
    "CameraCalibrationRetriever",
    "CubeDetection",
    "DetectionEventDispatcher",
    "DetectionPipeline",
    "RobotCenterDetector",
    "RobotCenterResult",
]
