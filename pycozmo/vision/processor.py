"""

Computer vision module for cube detection and tracking.

This module handles cube detection using the cube_detector implementation
and provides annotated frames for visualization.

"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pycozmo import logger
from pycozmo.vision.camera_calibration import CameraCalibration
from pycozmo.vision.floor_pipeline import FloorPipeline
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
    "AnnotatedVisionProcessor",
    "CubeDetection",
    "VisionProcessor",
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

    Uses CozmoCubeDetector from cube_detector for marker-based 3D tracking.
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

        # Face detection
        self.face_detector = None
        self._face_measurements: list[dict] = []

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

        logger.info("Vision processor initialized with cube_detector")

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
        except Exception as e:
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
        if not getattr(self, "enable_face_detection", True):
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
        except Exception as e:
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

    # Annotation settings
    annotation_colors = [
        (0, 255, 255),  # Cyan
    ]
    box_thickness = 2
    show_cube_id = True
    show_confidence = True
    show_distance = True

    def draw_detections(
        self, image: Image.Image, detections: list[CubeDetection]
    ) -> Image.Image:
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
                color = self.annotation_colors[
                    det.cube_id % len(self.annotation_colors)
                ]
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
                    fill=(0, 0, 0, 180),
                )

                # Draw text
                draw.text((x1, y1 - 15), label, fill=color, font=font)

        return annotated_image

    def detect_and_annotate(
        self, image: Image.Image, head_angle_rad: float = 0.0
    ) -> tuple[list[CubeDetection], list[FaceDetection], Image.Image]:
        """
        Detect cubes and faces, return both detections and an annotated image.

        Args:
            image: PIL Image from the robot's camera
            head_angle_rad: Current head tilt angle in radians (positive = looking up)

        Returns:
            Tuple of (cube_detections, face_detections, annotated image)
        """
        cube_detections = self.detect_cubes(image, head_angle_rad)
        face_detections = self.detect_faces(image, head_angle_rad)
        annotated_image = self.draw_detections(image, cube_detections)
        annotated_image = self.draw_face_detections(annotated_image, face_detections)
        return cube_detections, face_detections, annotated_image

    def draw_face_detections(
        self, image: Image.Image, detections: list[FaceDetection]
    ) -> Image.Image:
        """Draw face bounding boxes on the image.

        Args:
            image: PIL Image to annotate.
            detections: List of face detections to draw.

        Returns:
            Annotated PIL Image (a copy of the original).
        """
        if not detections:
            return image

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except Exception:
                font = ImageFont.load_default()

        color = (255, 160, 0)  # Orange for faces

        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            face_id = getattr(det, "_face_id", "?")
            z_mm = det.position_mm[2]
            label = f"Face {face_id} | {z_mm:.0f}mm"
            draw.text((x1, y1 - 13), label, fill=color, font=font)

        return annotated

    def draw_robot_center(
        self, image: Image.Image, result: RobotCenterResult | None = None
    ) -> Image.Image:
        """Draw robot center detection overlay on the image.

        Shows the two detected circles, the midpoint (robot center), and
        the lateral offset value.

        Args:
            image: PIL Image to annotate.
            result: RobotCenterResult to draw. If None, uses the latest result.

        Returns:
            Annotated PIL Image (a copy of the original).
        """
        if result is None:
            result = self.robot_center_result
        if result is None or not result.detected:
            return image

        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except Exception:
            try:
                font = ImageFont.truetype("arial.ttf", 11)
            except Exception:
                font = ImageFont.load_default()

        # Draw circle centres
        if result.circle_a_px is not None:
            cx, cy = result.circle_a_px
            r = 6
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2
            )

        if result.circle_b_px is not None:
            cx, cy = result.circle_b_px
            r = 6
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2
            )

        # Draw midpoint (robot center)
        if result.circle_a_px is not None and result.circle_b_px is not None:
            mid_x = (result.circle_a_px[0] + result.circle_b_px[0]) / 2.0
            mid_y = (result.circle_a_px[1] + result.circle_b_px[1]) / 2.0

            img_h = image.size[1]

            # Image centre reference line (grey dashed)
            img_center_x = image.size[0] / 2.0
            draw.line(
                [(img_center_x, 0), (img_center_x, img_h)],
                fill=(0, 255, 255),
                width=1,
            )

            # Vertical line at detected robot center (cyan)
            draw.line([(mid_x, 0), (mid_x, img_h)], fill=(0, 255, 255), width=1)

            # Center dot
            draw.ellipse(
                [(mid_x - 4, mid_y - 4), (mid_x + 4, mid_y + 4)],
                fill=(0, 255, 255),
            )

        # Label
        label = f"center: {result.lateral_offset_mm:+.1f}mm"
        draw.text((5, 5), label, fill=(0, 255, 255), font=font)

        return annotated


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

    def __init__(
        self,
        client,
        camera_calibration: CameraCalibration,
        process_every_n_frames: int = 1,
        undistort: bool = False,
        enable_floor_projection: bool = True,
        floor_center_mm: tuple[float, float] = (200.0, 0.0),
        floor_size_mm: tuple[float, float] = (600.0, 600.0),
        mm_per_pixel: float = 5.0,
        enable_occupancy_map: bool = True,
        occupancy_grid_size_mm: float = 2000.0,
        occupancy_cell_size_mm: float = 5.0,
        occupancy_inflation_radius: int = 2,
        occupancy_vote_cap: int = 10,
        occupancy_valid_min_brightness: int = 8,
        occupancy_edge_margin_px: int = 3,
        floor_edge_margin_px: int = 4,
        enable_face_detection: bool = True,
    ):
        """
        Initialize the annotated vision processor.

        Args:
            client: PyCozmo Client instance to attach handlers to
            camera_calibration: Camera calibration from Cozmo (required)
            process_every_n_frames: Process every Nth frame (1 = all frames,
                                   2 = every other frame, etc.)
            undistort: Whether to undistort images using camera calibration
                       (default: False, as undistortion may not be needed or coefficients may be incorrect)
            enable_floor_projection: Enable floor-projected (bird's-eye) view.
            floor_center_mm: World-frame (x, y) centre of the floor grid in mm.
            floor_size_mm: (width, height) of the floor grid in mm.
            mm_per_pixel: Millimetres per output pixel for the floor view.
            enable_occupancy_map: Enable world-frame occupancy map accumulation.
            occupancy_grid_size_mm: Physical extent of the occupancy grid (mm).
            occupancy_cell_size_mm: Millimetres per occupancy grid cell.
            occupancy_inflation_radius: Dilate edge mask by this many iterations.
            floor_edge_margin_px: Cut this many pixels off the camera-frame
                edges before projecting (drops stretched edge noise).
            occupancy_vote_cap: Saturation cap on each cell's vote accumulator.
                Votes never decay (map persists); the cap bounds how fast a
                cell flips between free/occupied.
            occupancy_valid_min_brightness: Minimum grey value for a floor pixel
                to count as seen (drops the black border + dim edge ring).
            occupancy_edge_margin_px: Suppress Canny edges within this many
                pixels of the projection silhouette (must be > inflation radius
                so the boundary ring isn't voted as occupied; interior edges
                survive).
            enable_face_detection: Enable Haar cascade face detection and 3D
                position estimation (default: True).
        """
        super().__init__(camera_calibration=camera_calibration)

        self.client = client
        self.process_every_n_frames = process_every_n_frames
        self.undistort = undistort
        self._frame_counter = 0
        self.enable_face_detection = enable_face_detection
        self.latest_occupancy_map_image: np.ndarray | None = None

        # Floor pipeline (optional)
        self.floor_pipeline: FloorPipeline | None = None
        if enable_floor_projection:
            self.floor_pipeline = FloorPipeline(
                camera_calibration=camera_calibration,
                enable_floor_projection=True,
                floor_center_mm=floor_center_mm,
                floor_size_mm=floor_size_mm,
                mm_per_pixel=mm_per_pixel,
                enable_occupancy_map=enable_occupancy_map,
                occupancy_grid_size_mm=occupancy_grid_size_mm,
                occupancy_cell_size_mm=occupancy_cell_size_mm,
                occupancy_inflation_radius=occupancy_inflation_radius,
                occupancy_vote_cap=occupancy_vote_cap,
                occupancy_valid_min_brightness=occupancy_valid_min_brightness,
                occupancy_edge_margin_px=occupancy_edge_margin_px,
                floor_edge_margin_px=floor_edge_margin_px,
            )

        # Import here to avoid circular dependency
        from pycozmo import event

        # Register camera image handler
        self.client.add_handler(event.EvtNewRawCameraImage, self._on_camera_image)

        logger.info(
            f"Annotated vision processor initialized (processing every "
            f"{process_every_n_frames} frame{'s' if process_every_n_frames > 1 else ''})"
        )

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
            from pycozmo import event

            # Get current head angle from robot state
            head_angle_rad = (
                cli.head_angle.radians if hasattr(cli, "head_angle") else 0.0
            )

            # Undistort the image using camera calibration (if enabled)
            if self.undistort:
                img_array = np.array(image)  # RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                undistorted_bgr = cv2.undistort(
                    img_bgr,
                    self.camera_calibration.camera_matrix,
                    self.camera_calibration.distortion_coefficients,
                )
                undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
                undistorted_image = Image.fromarray(undistorted_rgb)
            else:
                undistorted_image = image  # Use original image without undistortion

            # Detect cubes, faces, and annotate
            cube_detections, face_detections, annotated_image = (
                self.detect_and_annotate(undistorted_image, head_angle_rad)
            )

            # Draw robot center overlay if detected
            if (
                self.robot_center_result is not None
                and self.robot_center_result.detected
            ):
                annotated_image = self.draw_robot_center(
                    annotated_image, self.robot_center_result
                )

            # Dispatch annotated image event (pass cube detections for backwards compat)
            self.client.dispatch(
                event.EvtAnnotatedCameraImage, cli, annotated_image, cube_detections
            )

            # Also handle cube detection events
            self._dispatch_cube_events(cli, cube_detections)

            # Handle face detection events
            self._dispatch_face_events(cli, face_detections)

            # Floor projection (optional)
            if self.floor_pipeline is not None:
                robot_pose = cli.pose if hasattr(cli, "pose") else None
                floor_img, occ_grid, occ_map_image = self.floor_pipeline.process(
                    undistorted_image, head_angle_rad, robot_pose
                )
                if floor_img is not None:
                    self.client.dispatch(
                        event.EvtFloorProjectedImage, cli, floor_img, occ_grid
                    )
                # Store occupancy map image for web viewer
                self.latest_occupancy_map_image = occ_map_image

            # NOTE: do NOT call ``gc.collect()`` here. Periodic forced GC
            # runs synchronously on the pycozmo event-dispatcher thread,
            # which is the same thread that processes incoming camera
            # chunks. A full collection walks every tracked object in the
            # process; pause time grows with heap size and, after ~20 s
            # of cube detection (hundreds of PIL/numpy allocations per
            # second), became long enough that the dispatcher stalled
            # and both the web preview and detection fell to ~1 fps.
            # CPython's generational collector already handles short-lived
            # numpy/PIL objects; reference counting frees everything
            # promptly without it. (COZMO-21)

        except Exception as e:
            logger.error(f"Error processing camera image: {e}", exc_info=True)

    def _dispatch_cube_events(self, cli, detections: list[CubeDetection]):
        """
        Dispatch cube detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: List of current detections
        """
        # Import here to avoid circular dependency
        from pycozmo import event

        # Track which cubes we've seen
        if not hasattr(self, "_last_cube_ids"):
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

    def _dispatch_face_events(self, cli, detections: list[FaceDetection]):
        """Dispatch face detection events (detected, observed, lost)."""
        from pycozmo import event

        if not hasattr(self, "_last_face_ids"):
            self._last_face_ids = set()

        current_ids = {getattr(d, "_face_id", None) for d in detections}
        current_ids.discard(None)

        if current_ids != self._last_face_ids:
            new_ids = current_ids - self._last_face_ids
            if new_ids:
                new_dets = [
                    d for d in detections if getattr(d, "_face_id", None) in new_ids
                ]
                cli.dispatch(event.EvtFaceDetected, cli, new_dets)

            lost_ids = self._last_face_ids - current_ids
            for fid in lost_ids:
                cli.dispatch(event.EvtFaceLost, cli, fid)

            self._last_face_ids = current_ids

        if not hasattr(self, "_face_obs_counter"):
            self._face_obs_counter = 0

        if not detections:
            self._face_obs_counter = 0
            return

        self._face_obs_counter += 1
        if self._face_obs_counter % 5 == 0:
            cli.dispatch(event.EvtFaceObserved, cli, detections)

    def stop(self):
        """
        Stop the annotated vision processor and remove handlers.
        """

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
