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
from pycozmo.vision.floor_projector import FloorProjector
from pycozmo.vision.occupancy_map import OccupancyMap

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

        # Initialize the detector
        self._init_detector()

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

        # self.cube_detector.set_debug(True)

        # Create fusion tracker
        # max_age_frames is set very high so cubes persist at their last
        # known position indefinitely after detection, rather than
        # vanishing when temporarily out of view. (COZMO-27)
        self.cube_tracker = CubeFusionTracker(
            cube_size_mm=44.0, smoothing_alpha=0.3, max_age_frames=999999
        )

        logger.info("Cube detector initialized successfully")

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
            ``confidence``.
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
                }
            )
        return measurements

    def forget_cube(self, cube_id):
        """Remove a cube from the fusion tracker.

        Called when the cube's IMU detects movement — the old position is
        stale and the cube will be placed fresh on the next camera frame.
        (COZMO-27)
        """
        self.cube_tracker.forget_cube(cube_id)

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
                self.cube_detector.debug_images["original"] = img_bgr.copy()

            # Detect marker faces
            results = self.cube_detector.detect(img_bgr)

            # Fuse detections into coherent cube poses with head tilt compensation
            self.fused_cubes = self.cube_tracker.update(results, head_angle_rad)

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
    ) -> tuple[list[CubeDetection], Image.Image]:
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
        floor_edge_margin_px: int = 4,
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
        """
        super().__init__(camera_calibration=camera_calibration)

        self.client = client
        self.process_every_n_frames = process_every_n_frames
        self.undistort = undistort
        self._frame_counter = 0

        # Floor projector (optional)
        self.floor_projector: FloorProjector | None = None
        self.latest_floor_image = None
        self.latest_occupancy_grid = None
        self.latest_occupancy_map_image: np.ndarray | None = None
        if enable_floor_projection:
            self.floor_projector = FloorProjector(
                camera_calibration.camera_matrix,
                camera_calibration.image_width,
                camera_calibration.image_height,
                floor_center_mm=floor_center_mm,
                floor_size_mm=floor_size_mm,
                mm_per_pixel=mm_per_pixel,
                edge_margin_px=floor_edge_margin_px,
            )
            logger.info(
                f"Floor projection enabled: "
                f"{floor_size_mm[0]}x{floor_size_mm[1]}mm @ {mm_per_pixel} mm/px"
            )

        # Occupancy map (optional, requires floor projector)
        self.occupancy_map: OccupancyMap | None = None
        if enable_floor_projection and enable_occupancy_map:
            self.occupancy_map = OccupancyMap(
                grid_size_mm=occupancy_grid_size_mm,
                cell_size_mm=occupancy_cell_size_mm,
                inflation_radius=occupancy_inflation_radius,
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

            # Detect cubes and annotate
            detections, annotated_image = self.detect_and_annotate(
                undistorted_image, head_angle_rad
            )

            # Dispatch annotated image event
            self.client.dispatch(
                event.EvtAnnotatedCameraImage, cli, annotated_image, detections
            )

            # Dispatch debug images event
            debug_images = self.cube_detector.get_debug_images()
            self.client.dispatch(
                event.EvtCubeDetectorDebugImages, cli, debug_images, detections
            )

            # Also handle cube detection events
            self._dispatch_cube_events(cli, detections)

            # Floor projection (optional)
            if self.floor_projector is not None:
                img_array_for_floor = np.array(undistorted_image)
                img_bgr_for_floor = cv2.cvtColor(img_array_for_floor, cv2.COLOR_RGB2BGR)
                floor_img = self.floor_projector.project(
                    img_bgr_for_floor, head_angle_rad
                )
                occ_grid = self.floor_projector.compute_occupancy_grid(floor_img)
                self.latest_floor_image = floor_img
                self.latest_occupancy_grid = occ_grid
                cli.dispatch(event.EvtFloorProjectedImage, cli, floor_img, occ_grid)

                # Accumulate into world-frame occupancy map
                if self.occupancy_map is not None and hasattr(cli, "pose"):
                    pose = cli.pose
                    self.occupancy_map.update(
                        floor_img,
                        self.floor_projector._floor_x,
                        self.floor_projector._floor_y,
                        pose.position.x,
                        pose.position.y,
                        pose.rotation.angle_z.radians,
                    )
                    self.latest_occupancy_map_image = (
                        self.occupancy_map.to_color_image()
                    )

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
