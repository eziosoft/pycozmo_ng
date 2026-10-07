"""
Computer vision module for cube detection and tracking.

This module handles cube detection using pluggable detector implementations
and provides fused cube states for downstream consumers (WorldMap, etc.).
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np

from pycozmo_ng.logger import logger
from pycozmo_ng.vision.calibration.camera_calibration import CameraCalibration
from pycozmo_ng.vision.calibration.robot_center_calibration import RobotCenterCalibration
from pycozmo_ng.vision.charger_detector.charger_detection import ChargerDetection
from pycozmo_ng.vision.charger_detector.charger_template import load_charger_template
from pycozmo_ng.vision.interfaces import (
    ChargerDetector,
    CubeDetector,
    CubeFusionTracker,
    FaceDetector,
)


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

    # Cube index (world-map index 0-2; public cube IDs are 1-3 — see
    # pycozmo_ng.cubes.identity). Same key as WorldMap and fused_cubes.
    cube_index: int

    # Timestamp of detection
    timestamp: float = 0.0


class DetectionPipeline:
    """
    Computer vision processor for detecting and tracking Cozmo cubes.

    Uses pluggable detector and tracker implementations. Returns fused cube
    states in camera frame; temporal fusion and world-frame transform live in
    ``WorldMap``.

    Dependencies are injected via constructor for testability and swappability.
    Use :func:`create_default_detection_pipeline` for the standard setup.
    """

    def __init__(
        self,
        camera_calibration: CameraCalibration,
        cube_detector: CubeDetector,
        cube_tracker: CubeFusionTracker,
        face_detector: FaceDetector | None = None,
        robot_center_calibration: RobotCenterCalibration | None = None,
        charger_detector: ChargerDetector | None = None,
    ):
        """
        Initialize the detection pipeline with injected dependencies.

        Args:
            camera_calibration: Camera calibration from Cozmo (required)
            cube_detector: Cube detector implementation (marker-based, probe-sampling, etc.)
            cube_tracker: Cube fusion tracker implementation
            face_detector: Optional face detector (Haar cascade, etc.)
            robot_center_calibration: Owner of the robot-center measurement;
                frames are handed to it once per processed frame
            charger_detector: Optional charger marker detector (COZMO-65)
        """
        self.camera_calibration = camera_calibration
        self.cube_detector = cube_detector
        self.cube_tracker = cube_tracker
        self.fused_cubes: dict[int, dict] = {}  # 3D fused cube states

        # Face detection
        self.face_detector = face_detector
        self.enable_face_detection = True

        # Charger marker detection (COZMO-65)
        self.charger_detector = charger_detector
        self.enable_charger_detection = True

        # Robot center calibration — observes frames; owns the session offset
        self.robot_center_calibration = robot_center_calibration

        logger.info("Detection pipeline initialized")

    def get_fused_cube_states(self):
        """
        Get the fused per-cube states from the current frame.

        Returns:
            Dict of {cube_index: cube_state} containing position, rotation_matrix,
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
            List of dicts with keys: ``cube_index`` (0-2), ``position``
            (camera frame mm), ``rotation_matrix`` (camera frame),
            ``confidence``, ``rotation_source``.
        """
        measurements = []
        for cube_index, cube_state in self.fused_cubes.items():
            measurements.append(
                {
                    "cube_index": cube_index,
                    "position": cube_state["position"],
                    "rotation_matrix": cube_state["rotation_matrix"],
                    "confidence": cube_state["confidence"],
                    "num_faces": cube_state.get("num_faces", 1),
                    "rotation_source": cube_state.get("rotation_source"),
                }
            )
        return measurements

    def detect_faces(self, img_bgr: np.ndarray, head_angle_rad: float = 0.0) -> list:
        """Detect the closest face and return with 3D camera-frame position.

        Always returns at most one detection.  No multi-face tracking —
        we cannot re-identify faces across frames, so we just report the
        single best (largest / closest) face each frame.  Identity across
        frames is assigned downstream by ``FaceIdentity`` (presence
        epochs); this method only detects.

        Args:
            img_bgr: Image as BGR numpy array (already converted by caller).
            head_angle_rad: Current head tilt angle in radians.

        Returns:
            List with zero or one FaceDetection (position_mm in camera frame).
        """
        if self.face_detector is None:
            return []
        if not self.enable_face_detection:
            return []

        return self.face_detector.detect(img_bgr)

    def detect_charger(self, img_bgr: np.ndarray, head_angle_rad: float = 0.0) -> list[ChargerDetection]:
        """Detect the charger's rectangular marker (COZMO-65).

        Separate detection path from the cubes — the marker never enters the
        cube identity space, and the cube detector's ``reject_markers`` guard
        keeps the two families from claiming the same quad.

        Args:
            img_bgr: Image as BGR numpy array (already converted by caller).
            head_angle_rad: Current head tilt angle in radians (unused —
                kept for call-site symmetry with detect_cubes/detect_faces).

        Returns:
            Zero or one ChargerDetection (position_mm in camera frame).
        """
        if self.charger_detector is None or not self.enable_charger_detection:
            return []

        try:
            results = self.charger_detector.detect(img_bgr)
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Error in charger detection: {e}", exc_info=True)
            return []

        detections = []
        for r in results:
            corners = np.asarray(r["corners"]).reshape(-1, 2)
            x_min = int(corners[:, 0].min())
            y_min = int(corners[:, 1].min())
            x_max = int(corners[:, 0].max())
            y_max = int(corners[:, 1].max())
            position = np.asarray(r["position"], dtype=np.float64)

            detections.append(
                ChargerDetection(
                    x=x_min,
                    y=y_min,
                    width=max(1, x_max - x_min),
                    height=max(1, y_max - y_min),
                    confidence=float(r["score"]),
                    distance=float(np.linalg.norm(position)) / 1000.0,
                    position_mm=position,
                    timestamp=time.time(),
                )
            )
        return detections

    def detect_cubes(self, img_bgr: np.ndarray, head_angle_rad: float = 0.0) -> list[CubeDetection]:
        """
        Detect cubes in the given camera image.

        Args:
            img_bgr: Image as BGR numpy array (already converted by caller).
            image: Original PIL Image (unused, kept for API compatibility).
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            List of detected cubes with 3D tracking
        """
        timestamp = time.time()

        detections = self._detect_cubes(img_bgr, head_angle_rad)

        # Add timestamps
        for det in detections:
            det.timestamp = timestamp

        return detections

    def _detect_cubes(self, img_bgr: np.ndarray, head_angle_rad: float = 0.0) -> list[CubeDetection]:
        """
        Detect cubes using the injected detector with 3D tracking.

        Also performs the one-time robot center measurement (head at
        max-down tilt) on the first successful detection.

        Args:
            img_bgr: Image as BGR numpy array (already converted by caller).
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            List of detected cubes with 3D information
        """
        try:
            # Robot center calibration: observes every processed frame, but
            # only does work while a calibration run is armed (see
            # RobotCenterCalibration — it owns the session value).
            if self.robot_center_calibration is not None:
                self.robot_center_calibration.observe(img_bgr, head_angle_rad)

            # Detect marker faces
            results = self.cube_detector.detect(img_bgr)

            # Fuse detections into coherent cube poses with head tilt compensation
            self.fused_cubes = self.cube_tracker.update(results)

            # Convert to CubeDetection format
            detections = []
            for cube_index, cube_state in self.fused_cubes.items():
                pos = cube_state["position"]
                distance = np.linalg.norm(pos) / 1000.0  # Convert mm to meters

                # Get bounding box from detections
                cube_results = [r for r in results if r["cube_index"] == cube_index]
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
                            cube_index=cube_index,
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


def load_marker_templates(marker_dir: str | None = None) -> list[np.ndarray]:
    """Load and preprocess marker templates from disk.

    Args:
        marker_dir: Directory containing marker template JPGs. Defaults to
            ``pycozmo_ng/vision/cube_detector/cube_faces``.

    Returns:
        List of preprocessed marker templates as numpy arrays.
    """
    import glob
    import os

    if marker_dir is None:
        marker_dir = os.path.join(os.path.dirname(__file__), "", "cube_faces")

    if not os.path.exists(marker_dir):
        raise FileNotFoundError(f"Marker directory not found: {marker_dir}")

    template_files = sorted(glob.glob(os.path.join(marker_dir, "*.jpg")))
    if not template_files:
        raise FileNotFoundError(f"No marker templates found in {marker_dir}")

    logger.info(f"Loading {len(template_files)} marker templates")
    target_markers = []
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
        img_with_border = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
        target_markers.append(img_with_border)

    if not target_markers:
        raise RuntimeError("No valid marker templates could be loaded")

    return target_markers


def create_default_detection_pipeline(
    camera_calibration: CameraCalibration,
    marker_dir: str | None = None,
    use_clahe: bool = True,
    max_detections: int = 9,
    denoise_strength: int = 0,
    cube_size_mm: float = 44.0,
) -> DetectionPipeline:
    """Create a DetectionPipeline with the default marker-based implementation.

    This is the standard setup using CozmoCubeDetector and CubeFusionTracker.
    For alternative detectors (e.g., probe-sampling), construct DetectionPipeline
    directly with custom implementations.

    Args:
        camera_calibration: Camera calibration from Cozmo
        marker_dir: Directory with marker template JPGs
        use_clahe: Enable CLAHE preprocessing in detector
        max_detections: Maximum detections per frame
        denoise_strength: Denoise strength (0 = disabled)
        cube_size_mm: Physical cube size for fusion tracker

    Returns:
        Configured DetectionPipeline instance
    """
    # Import concrete implementations
    from pycozmo_ng.vision.calibration.robot_center import RobotCenterDetector
    from pycozmo_ng.vision.cube_detector.cube_detector import (
        CozmoCubeDetector,
        CubeFusionTracker,
    )
    from pycozmo_ng.vision.face_detector import FaceDetector

    # Load markers
    target_markers = load_marker_templates(marker_dir)

    # Load the charger marker template (COZMO-65). It serves double duty:
    # negative templates for the cube detector and positives for the charger
    # detector. Missing asset degrades to "no charger detection" only.
    charger_template = None
    try:
        charger_template = load_charger_template()
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(f"Charger marker template unavailable, charger detection disabled: {e}")

    # Create detector
    cube_detector = CozmoCubeDetector(
        camera_calibration.camera_matrix,
        camera_calibration.distortion_coefficients,
        target_markers,
        use_clahe=use_clahe,
        max_detections=max_detections,
        denoise_strength=denoise_strength,
        reject_markers=[charger_template] if charger_template is not None else None,
    )

    # Create fusion tracker
    cube_tracker = CubeFusionTracker(cube_size_mm=cube_size_mm)

    # Create face detector
    face_detector = None
    try:
        face_detector = FaceDetector(camera_calibration)
        logger.info("Face detector initialized successfully")
    except OSError as e:
        logger.warning(f"Failed to initialize face detector: {e}")

    # Create the charger marker detector — rejects quads the cube templates
    # out-score, mirroring the cube detector's guard (COZMO-65)
    charger_detector = None
    if charger_template is not None:
        from pycozmo_ng.vision.charger_detector.charger_detector import ChargerDetector

        charger_detector = ChargerDetector(
            camera_calibration.camera_matrix,
            camera_calibration.distortion_coefficients,
            [charger_template],
            reject_markers=target_markers,
            use_clahe=use_clahe,
            denoise_strength=denoise_strength,
        )
        logger.info("Charger marker detector initialized successfully")

    # Create the owner of the robot-center measurement (armed by
    # Cozmo.calibrate_robot_center; the detector only runs during a run)
    robot_center_calibration = RobotCenterCalibration(
        RobotCenterDetector(
            camera_calibration.camera_matrix,
            camera_calibration.distortion_coefficients,
        )
    )

    return DetectionPipeline(
        camera_calibration=camera_calibration,
        cube_detector=cube_detector,
        cube_tracker=cube_tracker,
        face_detector=face_detector,
        robot_center_calibration=robot_center_calibration,
        charger_detector=charger_detector,
    )


__all__ = [
    "CubeDetection",
    "DetectionPipeline",
    "create_default_detection_pipeline",
    "load_marker_templates",
]
