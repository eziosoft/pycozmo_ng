"""
Computer vision module for cube detection and tracking.

This module handles cube detection using the cube_detector implementation
and provides fused cube states for downstream consumers (WorldMap, etc.).
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from pycozmo import logger
from pycozmo.vision.camera_calibration import CameraCalibration
from pycozmo.vision.robot_center import RobotCenterDetector, RobotCenterResult

# Import CubeFusionTracker
try:
    from pycozmo.vision.cube_detector.cube_detector import (
        CozmoCubeDetector,
        CubeFusionTracker,
    )

    CUBE_DETECTOR_AVAILABLE = True
except ImportError:
    CUBE_DETECTOR_AVAILABLE = False
    logger.error("cube_detector module is required but not available")
    raise

# Import FaceDetector
try:
    from pycozmo.vision.face_detector import FaceDetection, FaceDetector

    FACE_DETECTOR_AVAILABLE = True
except ImportError:
    FACE_DETECTOR_AVAILABLE = False

__all__ = [
    "CubeDetection",
    "DetectionPipeline",
]


@dataclass
class CubeDetection:
    """Represents a detected cube in the camera image."""

    # Bounding box (x, y, width, height) in pixels
    x: int
    y: int
    width: int
    height: int

    # Confidence score (0.0 to 1.0)
    confidence: float

    # Estimated distance from robot (in meters)
    distance: float

    # Cube ID (1-indexed for display)
    cube_id: int

    # Timestamp of detection
    timestamp: float = 0.0


class DetectionPipeline:
    """
    Computer vision processor for detecting and tracking Cozmo cubes.

    Uses CozmoCubeDetector from cube_detector for marker-based 3D tracking.
    Returns fused cube states in camera frame; temporal fusion and world-frame
    transform live in WorldMap.
    """

    def __init__(self, camera_calibration: CameraCalibration):
        """
        Initialize the detection pipeline.

        Args:
            camera_calibration: Camera calibration from Cozmo (required)
        """
        self.camera_calibration = camera_calibration
        self.cube_detector = None
        self.cube_tracker = None
        self.fused_cubes = {}  # 3D fused cube states

        # Face detection
        self.face_detector = None
        self._face_measurements: list[dict] = []
        self.enable_face_detection = True

        # Robot center detection
        self.robot_center_detector = RobotCenterDetector(
            camera_calibration.camera_matrix,
            camera_calibration.distortion_coefficients,
        )
        self.robot_center_result: RobotCenterResult | None = None
        # One-time measurement: the offset is measured once during startup
        # calibration (head at max-down) and then frozen. After a successful
        # detection this flag is set and the detector is never run again.
        self._robot_center_measured: bool = False
        self._camera_lateral_offset_mm: float = 0.0

        # Initialize the detector
        self._init_detector()
        self._init_face_detector()

        logger.info("Detection pipeline initialized with cube_detector")

    def _init_detector(self):
        """Initialize the CozmoCubeDetector with marker templates."""
        import glob
        import os

        import cv2

        # Load marker templates
        target_markers = []
        marker_dir = os.path.join(
            os.path.dirname(__file__), "cube_detector", "cube_faces"
        )

        if not os.path.exists(marker_dir):
            raise FileNotFoundError(f"Marker directory not found: {marker_dir}")

        template_files = sorted(glob.glob(os.path.join(marker_dir, "*.jpg")))
        if not template_files:
            raise FileNotFoundError(f"No marker templates found in {marker_dir}")

        logger.info(f"Loading {len(template_files)} marker templates")
        for idx, img_path in enumerate(template_files):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to load {os.path.basename(img_path)}")
                continue
            # Process template: crop, resize, add border
            h, w = img.shape
            offset = 18
            img = img[offset : h - offset, offset : w - offset]
            img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_AREA)
            img_with_border = cv2.copyMakeBorder(
                img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255
            )
            target_markers.append(img_with_border)

        if not target_markers:
            raise RuntimeError("No valid marker templates could be loaded")

        # Create detector
        self.cube_detector = CozmoCubeDetector(
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coefficients,
            target_markers,
            use_clahe=True,
            max_detections=9,  # 3 cubes x up to 3 visible faces each (COZMO-36)
            denoise_strength=0,  # 0 disables noise reduction (denoise is off)
        )

        # Create fusion tracker
        # max_age_frames is set very high so cubes persist at their last
        # known position indefinitely after detection, rather than
        # vanishing when temporarily out of view. (COZMO-27)
        self.cube_tracker = CubeFusionTracker(
            cube_size_mm=44.0,
        )

        logger.info("Cube detector initialized successfully")

    def _init_face_detector(self):
        """Initialize the Haar cascade face detector."""
        if not FACE_DETECTOR_AVAILABLE:
            logger.warning("FaceDetector not available — face detection disabled")
            return
        try:
            self.face_detector = FaceDetector(self.camera_calibration)
            logger.info("Face detector initialized successfully")
        except OSError as e:
            logger.warning(f"Failed to initialize face detector: {e}")
            self.face_detector = None

    def get_fused_cube_states(self):
        """
        Get the fused per-cube states from the current frame.

        Returns:
            Dict of {cube_id: cube_state} containing position, rotation_matrix,
            confidence, etc. Poses are in **camera frame** — the temporal
            fusion and world-frame transform live in ``WorldMap``.
        """
        return self.fused_cubes.copy()

    def get_cube_measurements(self):
        """
        Get the camera-frame cube measurements from the current frame.

        These are the raw per-frame fused poses (one per cube) ready to be fed
        into ``WorldMap.update_cube_measurement``, which transforms them into
        world frame and fuses them there.

        Returns:
            List of dicts with keys: ``cube_id`` (0-indexed), ``position``
            (camera frame mm), ``rotation_matrix`` (camera frame),
            ``confidence``, ``rotation_source``.
        """
        measurements = []
        for cube_id, cube_state in self.fused_cubes.items():
            measurements.append(
                {
                    "cube_id": cube_id,
                    "position": cube_state["position"],
                    "rotation_matrix": cube_state["rotation_matrix"],
                    "confidence": cube_state["confidence"],
                    "num_faces": cube_state.get("num_faces", 1),
                    "rotation_source": cube_state.get("rotation_source"),
                }
            )
        return measurements

    def detect_faces(
        self, image: Image.Image, head_angle_rad: float = 0.0
    ) -> list[FaceDetection]:
        """Detect the closest face and return with 3D camera-frame position.

        Always returns at most one detection.  No multi-face tracking —
        we cannot re-identify faces across frames, so we just report the
        single best (largest / closest) face each frame.

        Returns:
            List with zero or one FaceDetection (position_mm in camera frame).
        """
        if self.face_detector is None:
            return []
        if not self.enable_face_detection:
            return []

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections = self.face_detector.detect(img_bgr)

        # Store measurements for downstream (WorldMap, events)
        self._face_measurements = [
            {
                "face_id": 0,
                "position": det.position_mm,
                "confidence": det.confidence,
            }
            for det in detections
        ]
        return detections

    def get_face_measurements(self) -> list[dict]:
        """Return the latest face measurements for feeding into WorldMap."""
        return list(self._face_measurements)

    def forget_cube(self, cube_id):
        """Remove a cube from the fusion tracker.

        Called when the cube's IMU detects movement — the old position is
        stale and the cube will be placed fresh on the next camera frame.
        (COZMO-27)

        CubeFusionTracker is stateless; temporal cube state lives in
        WorldMap, so there is nothing to do here.
        """

    def detect_cubes(
        self, image: Image.Image, head_angle_rad: float = 0.0
    ) -> list[CubeDetection]:
        """
        Detect cubes in the given camera image.

        Args:
            image: PIL Image from the robot's camera
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            List of detected cubes with 3D tracking
        """
        timestamp = time.time()
        img_array = np.array(image)

        detections = self._detect_cubes(img_array, head_angle_rad)

        # Add timestamps
        for det in detections:
            det.timestamp = timestamp

        return detections

    def _detect_cubes(
        self, img_array: np.ndarray, head_angle_rad: float = 0.0
    ) -> list[CubeDetection]:
        """
        Detect cubes using CozmoCubeDetector with 3D tracking.

        Also performs the one-time robot center measurement (head at
        max-down tilt) on the first successful detection.

        Args:
            img_array: Image as numpy array (RGB from PIL)
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            List of detected cubes with 3D information
        """
        try:
            import cv2

            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Robot center detection: measured exactly once during startup
            # calibration (head at max-down). Once a measurement succeeds the
            # value is remembered and detection never runs again.
            if not self._robot_center_measured:
                self.robot_center_detector.head_angle_rad = head_angle_rad
                self.robot_center_result = self.robot_center_detector.detect(img_bgr)
                if (
                    self.robot_center_result is not None
                    and self.robot_center_result.detected
                ):
                    self._camera_lateral_offset_mm = (
                        self.robot_center_result.lateral_offset_mm
                    )
                    self._robot_center_measured = True

            # Detect marker faces
            results = self.cube_detector.detect(img_bgr)

            # Fuse detections into coherent cube poses with head tilt compensation
            self.fused_cubes = self.cube_tracker.update(results)

            # Convert to CubeDetection format
            detections = []
            for cube_id, cube_state in self.fused_cubes.items():
                pos = cube_state["position"]
                distance = np.linalg.norm(pos) / 1000.0  # Convert mm to meters

                # Get bounding box from detections
                cube_results = [r for r in results if r["cube_id"] == cube_id]
                if cube_results:
                    corner_list = []
                    for r in cube_results:
                        corners = r["corners"]
                        if corners.ndim == 3:  # Handle OpenCV contour format (n, 1, 2)
                            corners = corners.reshape(-1, 2)
                        corner_list.append(corners)

                    if corner_list:
                        all_corners = np.vstack(corner_list)
                        x_min = int(np.min(all_corners[:, 0]))
                        y_min = int(np.min(all_corners[:, 1]))
                        x_max = int(np.max(all_corners[:, 0]))
                        y_max = int(np.max(all_corners[:, 1]))

                        det = CubeDetection(
                            x=x_min,
                            y=y_min,
                            width=max(1, x_max - x_min),
                            height=max(1, y_max - y_min),
                            confidence=cube_state["confidence"],
                            distance=distance,
                            cube_id=cube_id + 1,  # 1-indexed for display
                        )
                        detections.append(det)

            return detections
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error in cube detection: {e}", exc_info=True)
            return []

    def reset_tracking(self):
        """Reset the cube tracking state."""
        self.fused_cubes = {}
        logger.info("Cube tracking reset")

    @property
    def camera_lateral_offset_mm(self) -> float:
        """Lateral offset of camera from robot centre (mm), measured once.

        Measured once during startup calibration (head at max-down) and then
        frozen for the session.  Returns 0.0 until the one-time measurement
        succeeds.
        """
        return self._camera_lateral_offset_mm

    @camera_lateral_offset_mm.setter
    def camera_lateral_offset_mm(self, value: float) -> None:
        self._camera_lateral_offset_mm = value
