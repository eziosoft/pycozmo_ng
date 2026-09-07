"""

Computer vision module for cube detection and tracking.

This module handles cube detection using the cubeDetector.py implementation
and provides annotated frames for visualization.

"""

from typing import List, Tuple
from dataclasses import dataclass
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import logger
from .camera_calibration import CameraCalibration

# Import CubeFusionTracker
try:
    from .cubeDetector.cubeDetector import CubeFusionTracker, CozmoCubeDetector

    CUBE_DETECTOR_AVAILABLE = True
except ImportError:
    CUBE_DETECTOR_AVAILABLE = False
    logger.error("cubeDetector module is required but not available")
    raise

__all__ = [
    "CubeDetection",
    "VisionProcessor",
    "AnnotatedVisionProcessor",
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


class VisionProcessor:
    """
    Computer vision processor for detecting and tracking Cozmo cubes.

    Uses CozmoCubeDetector from cubeDetector.py for marker-based 3D tracking.
    """

    def __init__(self, camera_calibration: CameraCalibration):
        """
        Initialize the vision processor.

        Args:
            camera_calibration: Camera calibration from Cozmo (required)
        """
        self.camera_calibration = camera_calibration
        self.cube_detector = None
        self.cube_tracker = None
        self.fused_cubes = {}  # 3D fused cube states

        # Initialize the detector
        self._init_detector()

        logger.info("Vision processor initialized with cubeDetector.py")

    def _init_detector(self):
        """Initialize the CozmoCubeDetector with marker templates."""
        import cv2
        import glob
        import os

        # Load marker templates
        target_markers = []
        marker_dir = os.path.join(os.path.dirname(__file__), 'cubeDetector', 'cube_faces')

        if not os.path.exists(marker_dir):
            raise FileNotFoundError(f"Marker directory not found: {marker_dir}")

        template_files = sorted(glob.glob(os.path.join(marker_dir, '*.jpg')))
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
            img = img[offset:h - offset, offset:w - offset]
            img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_AREA)
            img_with_border = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)
            target_markers.append(img_with_border)

        if not target_markers:
            raise RuntimeError("No valid marker templates could be loaded")

        # Create detector
        self.cube_detector = CozmoCubeDetector(
            self.camera_calibration.camera_matrix,
            self.camera_calibration.distortion_coefficients,
            target_markers,
            use_clahe=True,
            max_detections=3,  # Cozmo has 3 cubes
            denoise_strength=0  # 0 disables noise reduction (denoise is off)
        )

        # self.cube_detector.set_debug(True)

        # Create fusion tracker
        self.cube_tracker = CubeFusionTracker(
            cube_size_mm=44.0,
            smoothing_alpha=0.3,
            max_age_frames=10
        )

        logger.info("Cube detector initialized successfully")

    def get_fused_cube_states(self):
        """
        Get the fused 3D cube states from the fusion tracker.

        Returns:
            Dict of {cube_id: cube_state} containing position, rotation_matrix, etc.
        """
        return self.fused_cubes.copy()

    def detect_cubes(self, image: Image.Image, head_angle_rad: float = 0.0) -> List[CubeDetection]:
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

    def _detect_cubes(self, img_array: np.ndarray, head_angle_rad: float = 0.0) -> List[CubeDetection]:
        """
        Detect cubes using CozmoCubeDetector with 3D tracking.

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


            # Save raw frame for debug
            if self.cube_detector.debug:
                self.cube_detector.debug_images['original'] = img_bgr.copy()

            # Detect marker faces
            results = self.cube_detector.detect(img_bgr)

            # Fuse detections into coherent cube poses with head tilt compensation
            self.fused_cubes = self.cube_tracker.update(results, head_angle_rad)

            # Convert to CubeDetection format
            detections = []
            for cube_id, cube_state in self.fused_cubes.items():
                pos = cube_state['position']
                distance = np.linalg.norm(pos) / 1000.0  # Convert mm to meters

                # Get bounding box from detections
                cube_results = [r for r in results if r['cube_id'] == cube_id]
                if cube_results:
                    corner_list = []
                    for r in cube_results:
                        corners = r['corners']
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
                            confidence=cube_state['confidence'],
                            distance=distance,
                            cube_id=cube_id + 1  # 1-indexed for display
                        )
                        detections.append(det)

            return detections
        except Exception as e:
            logger.error(f"Error in cube detection: {e}", exc_info=True)
            return []

    def reset_tracking(self):
        """Reset the cube tracking state."""
        if self.cube_tracker:
            self.cube_tracker.cubes.clear()
            logger.info("Cube tracking reset")

    # Annotation settings
    annotation_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    box_thickness = 2
    show_cube_id = True
    show_confidence = True
    show_distance = True

    def draw_detections(self, image: Image.Image, detections: List[CubeDetection]) -> Image.Image:
        """
        Draw detection bounding boxes and labels on an image.

        Args:
            image: PIL Image to annotate
            detections: List of cube detections to draw

        Returns:
            Annotated PIL Image (a copy of the original)
        """
        # Create a copy to avoid modifying the original
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

        for i, det in enumerate(detections):
            # Choose color based on cube ID or cycle through colors
            if det.cube_id is not None:
                color = self.annotation_colors[det.cube_id % len(self.annotation_colors)]
            else:
                color = self.annotation_colors[i % len(self.annotation_colors)]

            # Draw bounding box
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height

            # Draw rectangle with specified thickness. Pillow's ``width``
            # parameter already supports arbitrary thickness, so a single
            # call is enough — the previous loop redrew 1-px rectangles that
            # all overlapped, leaving only a 1-px line regardless of
            # ``box_thickness``. (COZMO-12)
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=max(1, self.box_thickness),
            )

            # Build label text
            label_parts = []
            if self.show_cube_id and det.cube_id is not None:
                label_parts.append(f"ID:{det.cube_id}")
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if self.show_distance and det.distance is not None:
                label_parts.append(f"{det.distance:.1f}m")

            if label_parts:
                label = " | ".join(label_parts)

                # Draw label background
                try:
                    # For newer Pillow versions
                    bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    # For older Pillow versions
                    text_width, text_height = draw.textsize(label, font=font)
                    bbox = (x1, y1 - 15, x1 + text_width, y1 - 15 + text_height)

                # Draw background rectangle for text
                draw.rectangle(
                    [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)],
                    fill=(0, 0, 0, 180)
                )

                # Draw text
                draw.text((x1, y1 - 15), label, fill=color, font=font)

        return annotated_image

    def detect_and_annotate(self, image: Image.Image, head_angle_rad: float = 0.0) -> Tuple[
        List[CubeDetection], Image.Image]:
        """
        Detect cubes and return both detections and an annotated image.

        This is a convenience method that combines detect_cubes() and draw_detections().

        Args:
            image: PIL Image from the robot's camera
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            Tuple of (detections list, annotated image)
        """
        detections = self.detect_cubes(image, head_angle_rad)
        annotated_image = self.draw_detections(image, detections)
        return detections, annotated_image


class AnnotatedVisionProcessor(VisionProcessor):
    """
    Vision processor that automatically generates annotated frames.

    This class extends VisionProcessor and automatically dispatches annotated
    camera images with detection boxes drawn on them. Use this when you want
    to receive both raw camera frames and annotated versions with visualizations.

    Example:
        vision = AnnotatedVisionProcessor(client, enable_tracking=True)

        # This will automatically start dispatching EvtAnnotatedCameraImage events
        def on_annotated_image(cli, annotated_image, detections):
            annotated_image.save("annotated_frame.png")

        cli.add_handler(pycozmo.event.EvtAnnotatedCameraImage, on_annotated_image)
    """

    def __init__(self, client, camera_calibration: CameraCalibration,
                 process_every_n_frames: int = 1, undistort: bool = False):
        """
        Initialize the annotated vision processor.

        Args:
            client: PyCozmo Client instance to attach handlers to
            camera_calibration: Camera calibration from Cozmo (required)
            process_every_n_frames: Process every Nth frame (1 = all frames,
                                   2 = every other frame, etc.)
            undistort: Whether to undistort images using camera calibration
                       (default: False, as undistortion may not be needed or coefficients may be incorrect)
        """
        super().__init__(camera_calibration=camera_calibration)

        self.client = client
        self.process_every_n_frames = process_every_n_frames
        self.undistort = undistort
        self._frame_counter = 0

        # Import here to avoid circular dependency
        from . import event

        # Register camera image handler
        self.client.add_handler(event.EvtNewRawCameraImage, self._on_camera_image)

        logger.info(f"Annotated vision processor initialized (processing every "
                    f"{process_every_n_frames} frame{'s' if process_every_n_frames > 1 else ''})")

    def _on_camera_image(self, cli, image: Image.Image):
        """
        Internal handler for camera images.

        This processes the image, detects cubes, generates annotations,
        and dispatches events.
        """
        # Terminal stop / temporary pause — both short-circuit the pipeline so
        # the HUD's "Detection DISABLED" toggle matches what the user sees.
        # Without these checks, detection, bbox annotation, and web-viewer
        # broadcasts keep running even when the user thinks they disabled
        # detection. (COZMO-14)
        if getattr(self, "_stopped", False):
            return
        if getattr(self, "_paused", False):
            return

        # Frame skipping for performance
        self._frame_counter += 1
        if self._frame_counter % self.process_every_n_frames != 0:
            return

        try:
            # Import here to avoid circular dependency
            from . import event

            # Get current head angle from robot state
            head_angle_rad = cli.head_angle.radians if hasattr(cli, 'head_angle') else 0.0

            # Undistort the image using camera calibration (if enabled)
            if self.undistort:
                img_array = np.array(image)  # RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                undistorted_bgr = cv2.undistort(img_bgr, self.camera_calibration.camera_matrix, self.camera_calibration.distortion_coefficients)
                undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
                undistorted_image = Image.fromarray(undistorted_rgb)
            else:
                undistorted_image = image  # Use original image without undistortion

            # Detect cubes and annotate
            detections, annotated_image = self.detect_and_annotate(undistorted_image, head_angle_rad)

            # Dispatch annotated image event
            self.client.dispatch(event.EvtAnnotatedCameraImage, cli, annotated_image, detections)

            # Dispatch debug images event
            debug_images = self.cube_detector.get_debug_images()
            self.client.dispatch(event.EvtCubeDetectorDebugImages, cli, debug_images, detections)

            # Also handle cube detection events
            self._dispatch_cube_events(cli, detections)

            # Periodic cleanup every 30 frames to prevent memory buildup
            if self._frame_counter % 30 == 0:
                import gc
                gc.collect()

        except Exception as e:
            logger.error(f"Error processing camera image: {e}", exc_info=True)

    def _dispatch_cube_events(self, cli, detections: List[CubeDetection]):
        """
        Dispatch cube detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: List of current detections
        """
        # Import here to avoid circular dependency
        from . import event

        # Track which cubes we've seen
        if not hasattr(self, '_last_cube_ids'):
            self._last_cube_ids = set()

        current_ids = {det.cube_id for det in detections if det.cube_id is not None}

        # Only dispatch events if something changed
        if current_ids != self._last_cube_ids:
            # Detect new cubes
            new_ids = current_ids - self._last_cube_ids
            if new_ids:
                new_detections = [d for d in detections if d.cube_id in new_ids]
                cli.dispatch(event.EvtCubeDetected, cli, new_detections)

            # Detect lost cubes
            lost_ids = self._last_cube_ids - current_ids
            for cube_id in lost_ids:
                cli.dispatch(event.EvtCubeLost, cli, cube_id)

            # Update state
            self._last_cube_ids = current_ids

        # Throttle observation events - only dispatch every 5th frame. The
        # counter resets to zero whenever no cubes are detected so the
        # throttle measures "frames since the last observation dispatch
        # *with detections*". Without the reset, a long gap between
        # detections could either delay or skip the first re-detection
        # observation, depending on where the counter happened to stop.
        # (COZMO-15)
        if not hasattr(self, "_obs_frame_counter"):
            self._obs_frame_counter = 0

        if not detections:
            self._obs_frame_counter = 0
            return

        self._obs_frame_counter += 1
        if self._obs_frame_counter % 5 == 0:
            cli.dispatch(event.EvtCubeObserved, cli, detections)

    def stop(self):
        """
        Stop the annotated vision processor and remove handlers.
        """
        from . import event

        # Note: There's no built-in way to remove a specific handler in the current
        # implementation, but we can stop processing by setting a flag
        self._stopped = True
        logger.info("Annotated vision processor stopped")

    def pause(self):
        """
        Pause the detection pipeline.

        While paused, ``_on_camera_image`` short-circuits before any detection,
        annotation, event-dispatch, or web-viewer broadcast runs. Use this to
        truly stop the pipeline (in contrast to ``stop()``, which is terminal
        and also removes handlers). Pair with :meth:`resume` to re-enable.

        (COZMO-14)
        """
        self._paused = True

    def resume(self):
        """Resume the detection pipeline after :meth:`pause`."""
        self._paused = False
