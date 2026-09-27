"""Frame stage — the per-frame vision sequence behind one small interface.

Owns everything that happens to a camera frame, end to end:

    convert → detect → annotate → publish → dispatch events → feed world map

``Brain`` (the composition root) only constructs and wires the stage; the
stable dependencies (state containers, world map, web sink) arrive via the
constructor, and the per-session vision components created by
``Brain.setup_vision`` arrive through :meth:`configure`.

Replaces the old three-module relay chain ``Brain._on_camera_image`` →
``FrameHandler`` + ``WebStreamer`` + ``WebBridge.update_web_viewer``.
"""

import time

import cv2
import numpy as np
from PIL import Image

from pycozmo import event
from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.display.expressions import Happiness
from pycozmo.logger import logger, logger_cubes
from pycozmo.navigation.world_map import WorldMap
from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.cube_detector.detection_pipeline import DetectionPipeline
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.frame_state import FrameState
from pycozmo.vision.occupancy_map.floor_pipeline import FloorPipeline
from pycozmo.web.viewer_server import WebViewerServer


class FrameStage:
    """Deep module: one camera frame in, world map and browser updated.

    The interface is ``configure()`` (once per vision setup) and
    :meth:`on_camera_image` (once per frame). Detection, annotation,
    event dispatch, face expression, floor projection, occupancy push,
    and world-map feeding are all implementation details behind it.
    """

    def __init__(
        self,
        frame_state: FrameState,
        cube_tracking: CubeTrackingState,
        world_map: WorldMap,
        web_viewer: WebViewerServer,
    ):
        self.fs = frame_state
        self.ct = cube_tracking
        self.world_map = world_map
        self.web_viewer = web_viewer
        # Set by the composition root when the client connects
        self.cli = None

        # Per-session vision components — None until configure()
        self.detection_pipeline: DetectionPipeline | None = None
        self.annotation_engine: AnnotationEngine | None = None
        self.event_dispatcher: DetectionEventDispatcher | None = None
        self.floor_pipeline: FloorPipeline | None = None

        # Frame gating — configure() overrides
        self.process_every_n_frames = 1
        self.undistort = False
        self.face_detect_every_n_frames = 1
        self._vision_frame_counter = 0
        self._face_frame_counter = 0

    def configure(
        self,
        *,
        detection_pipeline: DetectionPipeline,
        annotation_engine: AnnotationEngine,
        event_dispatcher: DetectionEventDispatcher,
        floor_pipeline: FloorPipeline | None = None,
        process_every_n_frames: int = 1,
        undistort: bool = False,
        face_detect_every_n_frames: int = 1,
    ) -> None:
        """Inject the per-session vision components (from Brain.setup_vision).

        Resets the frame gates so a re-setup starts counting from scratch.
        """
        self.detection_pipeline = detection_pipeline
        self.annotation_engine = annotation_engine
        self.event_dispatcher = event_dispatcher
        self.floor_pipeline = floor_pipeline
        self.process_every_n_frames = process_every_n_frames
        self.undistort = undistort
        self.face_detect_every_n_frames = face_detect_every_n_frames
        self._vision_frame_counter = 0
        self._face_frame_counter = 0

    # ------------------------------------------------------------------
    # Per-frame sequence
    # ------------------------------------------------------------------

    def on_camera_image(self, cli, image) -> None:
        """Handle EvtNewRawCameraImage — run the full vision sequence."""
        self._stamp_raw_frame(image)
        self._update_fps()

        # Frame skipping for performance
        self._vision_frame_counter += 1
        if self._vision_frame_counter % self.process_every_n_frames != 0:
            return
        if self.detection_pipeline is None or self.annotation_engine is None:
            return

        try:
            # Get current head angle from robot state
            head_angle_rad = (
                cli.head_angle.radians if hasattr(cli, "head_angle") else 0.0
            )

            # Single PIL→numpy→BGR conversion (reused by all downstream)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Undistort the image using camera calibration (if enabled)
            if self.undistort:
                calib = self.detection_pipeline.camera_calibration
                img_bgr = cv2.undistort(
                    img_bgr,
                    calib.camera_matrix,
                    calib.distortion_coefficients,
                )

            # Convert to PIL for annotation (only one conversion needed)
            undistorted_image = Image.fromarray(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            )

            # Detect cubes — pass BGR numpy directly (avoids re-conversion)
            cube_detections = self.detection_pipeline.detect_cubes(
                img_bgr, head_angle_rad
            )

            # Detect faces (every Nth processed frame)
            self._face_frame_counter += 1
            face_due = self._face_frame_counter % self.face_detect_every_n_frames == 0
            if face_due:
                face_detections = self.detection_pipeline.detect_faces(
                    img_bgr, head_angle_rad
                )
            else:
                face_detections = []

            # Annotate image (single copy inside)
            annotated_image = self.annotation_engine.detect_and_annotate(
                undistorted_image,
                cube_detections,
                face_detections,
                self.detection_pipeline.robot_center_result,
            )

            # Store frame + detections, stamp them, push to the MJPEG sink
            self._publish_annotated(cli, annotated_image, cube_detections)

            # Dispatch cube/face events
            if self.event_dispatcher is not None:
                self.event_dispatcher.dispatch_cube_events(cli, cube_detections)
                self.event_dispatcher.dispatch_face_events(cli, face_detections)

            # Set face expression based on detection
            self._update_face_expression(cli, face_detections, face_due)

            # Feed measurements into the world map, then wake the browser
            self._feed_world_map(cli)

            # Floor projection + occupancy (optional)
            self._process_floor(cli, img_bgr, head_angle_rad)

        except (OSError, ValueError, RuntimeError) as e:
            logger_cubes.error(f"Error processing camera image: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    def _stamp_raw_frame(self, image) -> None:
        """Record the raw frame (runs for every frame, even skipped ones)."""
        self.fs.latest_raw_frame = image
        self.fs.frame_count += 1
        self.fs.last_frame_time = time.time()

    def _update_fps(self) -> None:
        """Update the FPS counter in frame state (once per raw frame)."""
        elapsed = time.time() - self.fs.start_time
        if elapsed > 1.0:
            self.fs.fps = self.fs.frame_count / elapsed
            self.fs.frame_count = 0
            self.fs.start_time = time.time()

    def _publish_annotated(self, cli, annotated_image, cube_detections) -> None:
        """Store the annotated frame + detections and push it to the browser."""
        self.fs.latest_annotated_frame = annotated_image
        with self.fs.detections_lock:
            self.fs.latest_detections = cube_detections

        # Record the current frame index for every detected cube
        detected_cube_ids = {det.cube_id for det in cube_detections}
        for cube_id in detected_cube_ids:
            self.ct.last_seen_frame[cube_id] = self.fs.frame_count

        if annotated_image is not None:
            self.web_viewer.push_annotated_frame(annotated_image)

    def _update_face_expression(self, cli, face_detections, face_due: bool) -> None:
        """Mirror face detection onto the procedural face (Happiness)."""
        anim_ctrl = getattr(cli, "anim_controller", None)
        if anim_ctrl is None:
            return
        if face_detections:
            anim_ctrl.set_face_expression(Happiness())
        elif face_due:
            # Only clear when we actually ran detection and found nothing
            anim_ctrl.clear_face_expression()

    def _feed_world_map(self, cli) -> None:
        """Feed cube, face, and procedural-face measurements into the world map.

        Cubes that have been invalidated (moved, forget_cube) are
        suppressed until they are re-detected stably. (COZMO-34)
        """
        if not self.detection_pipeline or self.world_map is None:
            return

        head_angle = cli.head_angle.radians if cli is not None else 0.0

        # Read persistent lateral offset
        lateral_offset = self.detection_pipeline.camera_lateral_offset_mm

        # Synchronized access to suppress state
        with self.ct.suppress_lock:
            suppressed = set(self.ct.suppress.get("ids", set()))
            fresh_frames = dict(self.ct.suppress.get("fresh_frames", {}))
            fresh_frames_needed = self.ct.suppress.get("fresh_frames_needed", 5)
            new_fresh_frames = {}

        for m in self.detection_pipeline.get_cube_measurements():
            cube_id = m["cube_id"]
            if cube_id in suppressed:
                n = fresh_frames.get(cube_id, 0) + 1
                new_fresh_frames[cube_id] = n
                if n < fresh_frames_needed:
                    continue
                suppressed.discard(cube_id)
                logger.info(
                    "Cube %d re-settled - re-enabling measurements", cube_id + 1
                )

            self.world_map.update_cube_measurement(
                cube_id,
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
                m.get("num_faces"),
                camera_lateral_offset_mm=lateral_offset,
                rotation_source=m.get("rotation_source"),
            )

        with self.ct.suppress_lock:
            self.ct.suppress["ids"] = suppressed
            self.ct.suppress["fresh_frames"] = new_fresh_frames

        # Feed face measurements into world map
        for fm in self.detection_pipeline.get_face_measurements():
            self.world_map.update_face_measurement(
                fm["face_id"],
                fm["position"],
                fm["confidence"],
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
                label=f"Face {fm['face_id']}",
            )

        # Push procedural face image to the browser
        anim_ctrl = getattr(cli, "anim_controller", None)
        if anim_ctrl is not None:
            face_im = getattr(anim_ctrl, "latest_face_image", None)
            if face_im is not None:
                self.web_viewer.push_face_image(face_im)

        self.web_viewer.broadcast()

    def _process_floor(self, cli, img_bgr, head_angle_rad: float) -> None:
        """Project the floor, dispatch the event, publish image + occupancy."""
        if self.floor_pipeline is None:
            return

        robot_pose = cli.pose if hasattr(cli, "pose") else None
        floor_img, occ_grid, occ_map_image = self.floor_pipeline.process(
            img_bgr, head_angle_rad, robot_pose
        )
        if floor_img is None:
            return

        cli.dispatch(event.EvtFloorProjectedImage, cli, floor_img, occ_grid)
        self.web_viewer.push_floor_frame(floor_img)

        # Publish the occupancy map image for the browser overlay
        occ = self.floor_pipeline.occupancy_map
        if (
            occ_map_image is not None
            and self.floor_pipeline.occupancy_enabled
            and occ is not None
        ):
            self.web_viewer.push_occupancy_grid(
                occ_map_image, occ.origin[0], occ.origin[1]
            )

        # Sync floor grid dimensions to browser on first frame
        fp = self.floor_pipeline.floor_projector
        if fp is not None:
            self.web_viewer.set_floor_size(fp.floor_size[0], fp.floor_size[1])
            self.web_viewer.set_floor_center(fp.floor_center[0], fp.floor_center[1])
