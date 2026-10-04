"""Vision subpackage — processor, display, calibration, and detection modules."""

from pycozmo_ng.vision.annotation import AnnotationEngine
from pycozmo_ng.vision.calibration.camera_calibration import (
    CameraCalibration,
    CameraCalibrationRetriever,
)
from pycozmo_ng.vision.calibration.robot_center import (
    RobotCenterDetector,
    RobotCenterResult,
)
from pycozmo_ng.vision.cube_detector.detection_pipeline import (
    CubeDetection,
    DetectionPipeline,
    create_default_detection_pipeline,
    load_marker_templates,
)
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.face_identity import FaceFrame, FaceIdentity
from pycozmo_ng.vision.frame_stage import FrameStage
from pycozmo_ng.vision.frame_state import FrameState
from pycozmo_ng.vision.interfaces import (
    CubeDetector,
    CubeFusionTracker,
    FaceDetector,
)

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
    "FaceFrame",
    "FaceIdentity",
    "FrameStage",
    "FrameState",
    "RobotCenterDetector",
    "RobotCenterResult",
    "create_default_detection_pipeline",
    "load_marker_templates",
]
