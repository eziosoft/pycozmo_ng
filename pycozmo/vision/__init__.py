"""Vision subpackage — processor, display, calibration, and detection modules."""

from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.calibration.camera_calibration import (
    CameraCalibration,
    CameraCalibrationRetriever,
)
from pycozmo.vision.calibration.robot_center import (
    RobotCenterDetector,
    RobotCenterResult,
)
from pycozmo.vision.cube_detector.detection_pipeline import (
    CubeDetection,
    DetectionPipeline,
    create_default_detection_pipeline,
    load_marker_templates,
)
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.face_identity import FaceFrame, FaceIdentity
from pycozmo.vision.frame_stage import FrameStage
from pycozmo.vision.frame_state import FrameState
from pycozmo.vision.interfaces import (
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
