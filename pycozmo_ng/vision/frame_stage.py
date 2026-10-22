"""Frame stage — the per-frame vision sequence behind one small interface.

Owns everything that happens to a camera frame, end to end:

    convert → detect → identify → annotate → publish → dispatch events →
    feed world map

``Brain`` (the composition root) only constructs and wires the stage; the
stable dependencies (frame state, world map, web sink) arrive via the
constructor, and the per-session vision components created by
``Brain.setup_vision`` arrive through :meth:`configure`.

Replaces the old three-module relay chain ``Brain._on_camera_image`` →
``FrameHandler`` + ``WebStreamer`` + ``WebBridge.update_web_viewer``.
"""

import time

import cv2
import numpy as np
from PIL import Image

from pycozmo_ng import event
from pycozmo_ng.display.expressions import Happiness
from pycozmo_ng.logger import logger_cubes
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.util import robot
from pycozmo_ng.vision.annotation import AnnotationEngine
from pycozmo_ng.vision.cube_detector.detection_pipeline import DetectionPipeline
from pycozmo_ng.vision.distance_rings import DistanceRings
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.face_identity import FaceFrame, FaceIdentity
from pycozmo_ng.vision.frame_sink import FrameSink
from pycozmo_ng.vision.frame_state import FrameState
from pycozmo_ng.vision.occupancy_map.floor_pipeline import FloorPipeline
from pycozmo_ng.vision.rolling_shutter import RollingShutterCorrector


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
        world_map: WorldMap,
        sink: FrameSink,
    ):
        self.fs = frame_state
        self.world_map = world_map
        self.sink = sink
        # Set by the composition root when the client connects
        self.cli = None

        # Per-session vision components — None until configure()
        self.detection_pipeline: DetectionPipeline | None = None
        self.annotation_engine: AnnotationEngine | None = None
        self.distance_rings: DistanceRings | None = None
        self.event_dispatcher: DetectionEventDispatcher | None = None
        self.floor_pipeline: FloorPipeline | None = None
        # Rolling-shutter corrector (fed by imageGyro packets) — None until
        # configure() injects it; None = correction disabled.
        self.rolling_shutter: RollingShutterCorrector | None = None
        # Face presence identity; replaced by the composition root's
        # instance on configure() (fresh ID numbering per vision session).
        self.face_identity = FaceIdentity()

        # Frame gating — configure() overrides
        self.process_every_n_frames = 1
        self.undistort = False
        self.face_detect_every_n_frames = 1
        self._vision_frame_counter = 0
        self._face_frame_counter = 0

        # Cube motion gate (see ``_cube_motion_gate_active``): while the robot
        # is turning, motion blur makes blurred blobs match cube templates, so
        # only cubes already in the world map may be updated.
        self.cube_body_rate_gate_radps = 0.5236  # 30 deg/s — the firmware's own limit
        self.cube_head_rate_gate_radps = 0.1745  # 10 deg/s — likewise
        self._last_head_angle: float | None = None
        self._last_head_time: float | None = None
        self._head_rate_radps = 0.0

    def configure(
        self,
        *,
        detection_pipeline: DetectionPipeline,
        annotation_engine: AnnotationEngine,
        distance_rings: DistanceRings | None = None,
        event_dispatcher: DetectionEventDispatcher,
        face_identity: FaceIdentity,
        floor_pipeline: FloorPipeline | None = None,
        rolling_shutter: RollingShutterCorrector | None = None,
        process_every_n_frames: int = 1,
        undistort: bool = False,
        face_detect_every_n_frames: int = 1,
        cube_body_rate_gate_radps: float = 0.5236,
        cube_head_rate_gate_radps: float = 0.1745,
    ) -> None:
        """Inject the per-session vision components (from Brain.setup_vision).

        Resets the frame gates so a re-setup starts counting from scratch.
        """
        self.detection_pipeline = detection_pipeline
        self.annotation_engine = annotation_engine
        self.distance_rings = distance_rings
        self.event_dispatcher = event_dispatcher
        self.face_identity = face_identity
        self.floor_pipeline = floor_pipeline
        self.rolling_shutter = rolling_shutter
        self.process_every_n_frames = process_every_n_frames
        self.undistort = undistort
        self.face_detect_every_n_frames = face_detect_every_n_frames
        self.cube_body_rate_gate_radps = cube_body_rate_gate_radps
        self.cube_head_rate_gate_radps = cube_head_rate_gate_radps
        self._vision_frame_counter = 0
        self._face_frame_counter = 0
        self._last_head_angle = None
        self._last_head_time = None
        self._head_rate_radps = 0.0

    # ------------------------------------------------------------------
    # Per-frame sequence
    # ------------------------------------------------------------------

    def on_camera_image(self, cli, image) -> None:
        """Handle EvtNewRawCameraImage — run the full vision sequence."""
        self._stamp_raw_frame(image)
        self._update_fps()
        self._update_head_rate(cli)

        # Frame skipping for performance
        self._vision_frame_counter += 1
        if self._vision_frame_counter % self.process_every_n_frames != 0:
            return
        if self.detection_pipeline is None or self.annotation_engine is None:
            return

        try:
            # Get current head angle from robot state
            head_angle_rad = cli.head_angle.radians if hasattr(cli, "head_angle") else 0.0

            # Single PIL→numpy→BGR conversion (reused by all downstream)
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Rolling-shutter correction (engine RollingShutterCorrector):
            # each 2-row band was exposed at a different time during the
            # ~65 ms sensor read-out; in-frame IMU samples give the angular
            # rate over that window, integrated per band and warped
            # horizontally so detectors see a re-aligned image.
            if self.rolling_shutter is not None:
                img_bgr = self.rolling_shutter.correct(img_bgr, cli.last_image_timestamp, cli.last_image_id)

            # Undistort the image using camera calibration (if enabled)
            if self.undistort:
                calib = self.detection_pipeline.camera_calibration
                img_bgr = cv2.undistort(
                    img_bgr,
                    calib.camera_matrix,
                    calib.distortion_coefficients,
                )

            # Convert to PIL for annotation (only one conversion needed)
            undistorted_image = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            # Ground distance rings first — detection overlays land on top
            if self.distance_rings is not None:
                self.distance_rings.draw(undistorted_image, head_angle_rad)

            # Detect cubes — pass BGR numpy directly (avoids re-conversion)
            cube_detections = self.detection_pipeline.detect_cubes(img_bgr, head_angle_rad)

            # While the robot turns, motion blur turns blurred blobs into cube
            # template matches. Trust only cubes the world map already knows;
            # a phantom must not enter just because the camera is sweeping.
            if self._cube_motion_gate_active(cli):
                known = set(self.world_map.get_cubes()) if self.world_map is not None else set()
                cube_detections = [d for d in cube_detections if d.cube_index in known]
                self.detection_pipeline.restrict_cubes(known)

            # Detect the charger marker — separate path, never a cube ID
            charger_detections = self.detection_pipeline.detect_charger(img_bgr, head_angle_rad)

            # Detect ArUco markers — separate path, marker ID is the identity
            aruco_detections = self.detection_pipeline.detect_aruco(img_bgr, head_angle_rad)

            # Detect faces (every Nth processed frame)
            self._face_frame_counter += 1
            face_due = self._face_frame_counter % self.face_detect_every_n_frames == 0
            if face_due:
                face_detections = self.detection_pipeline.detect_faces(img_bgr, head_angle_rad)
            else:
                face_detections = []

            # Assign face identity BEFORE annotation reads it. A skipped
            # frame (face_due False) is "not measured", not "face gone".
            face_frame = self.face_identity.update(face_detections, measured=face_due)

            # Annotate image (single copy inside)
            robot_center = self.detection_pipeline.robot_center_calibration
            annotated_image = self.annotation_engine.detect_and_annotate(
                undistorted_image,
                cube_detections,
                face_detections,
                robot_center.last_result if robot_center is not None else None,
                charger_detections,
                aruco_detections,
            )

            # Store frame + detections, stamp them, push to the MJPEG sink
            self._publish_annotated(cli, annotated_image, cube_detections)

            # Dispatch cube/face/charger/aruco events
            if self.event_dispatcher is not None:
                self.event_dispatcher.dispatch_cube_events(cli, cube_detections)
                self.event_dispatcher.dispatch_charger_events(cli, charger_detections)
                self.event_dispatcher.dispatch_aruco_events(cli, aruco_detections)
                self.event_dispatcher.dispatch_face_events(cli, face_detections, face_frame)

            # Set face expression based on detection
            self._update_face_expression(cli, face_detections, face_due)

            # Feed measurements into the world map, then wake the browser
            self._feed_world_map(cli, face_detections, face_frame, charger_detections, aruco_detections)

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
        # The raw MJPEG stream mirrors the camera exactly — it runs for
        # every frame, independent of the vision gates below.
        self.sink.push_raw_frame(image)

    def _update_fps(self) -> None:
        """Update the FPS counter in frame state (once per raw frame)."""
        elapsed = time.time() - self.fs.start_time
        if elapsed > 1.0:
            self.fs.fps = self.fs.frame_count / elapsed
            self.fs.frame_count = 0
            self.fs.start_time = time.time()

    def _update_head_rate(self, cli) -> None:
        """Measure the head's angular rate from successive frames.

        Must run before the gate is consulted: it compares against the previous
        frame's angle and then stores the current one.
        """
        head_angle = getattr(cli, "head_angle", None)
        if head_angle is None:
            self._head_rate_radps = 0.0
            return
        now = time.monotonic()
        angle = float(head_angle.radians)
        rate = 0.0
        if self._last_head_angle is not None and self._last_head_time is not None:
            elapsed = now - self._last_head_time
            # Ignore gaps that are not a frame apart (stalls, reconnects).
            if 0.0 < elapsed <= 0.25:
                rate = abs(angle - self._last_head_angle) / elapsed
        self._head_rate_radps = rate
        self._last_head_angle = angle
        self._last_head_time = now

    def _cube_motion_gate_active(self, cli) -> bool:
        """True while the robot turns fast enough to blur cube markers.

        The firmware's own gate (``VisionComponent::WasRotatingTooFast``)
        discards marker observations above 30 deg/s of body yaw or 10 deg/s of
        head motion; the same thresholds are used here. Body yaw comes from the
        wheel speeds (``TRACK_WIDTH`` apart, so differential speed / track width
        is the yaw rate) — the gyro axes' units are not verified, while wheel
        speeds are millimetres per second by protocol.
        """
        left = getattr(cli, "left_wheel_speed", None)
        right = getattr(cli, "right_wheel_speed", None)
        if left is not None and right is not None:
            body_rate = abs(float(right.mmps) - float(left.mmps)) / robot.TRACK_WIDTH.mm
            if body_rate > self.cube_body_rate_gate_radps:
                return True
        return self._head_rate_radps > self.cube_head_rate_gate_radps

    def _publish_annotated(self, cli, annotated_image, cube_detections) -> None:
        """Store the annotated frame + detections and push it to the browser."""
        self.fs.latest_annotated_frame = annotated_image
        with self.fs.detections_lock:
            self.fs.latest_detections = cube_detections

        if annotated_image is not None:
            self.sink.push_annotated_frame(annotated_image)

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

    def _feed_world_map(
        self,
        cli,
        face_detections,
        face_frame: FaceFrame,
        charger_detections=None,
        aruco_detections=None,
    ) -> None:
        """Feed cube, face, charger, ArUco, and procedural-face measurements into the world map.

        Face identities that ended this frame are dropped from the map.
        The charger and ArUco markers are stationary — their poses are
        fused but never forgotten.
        """
        if not self.detection_pipeline or self.world_map is None:
            return

        head_angle = cli.head_angle.radians if cli is not None else 0.0

        # Read the session camera lateral offset (frozen by robot-center calibration)
        robot_center = self.detection_pipeline.robot_center_calibration
        lateral_offset = robot_center.offset_mm if robot_center is not None else 0.0

        for m in self.detection_pipeline.get_cube_measurements():
            self.world_map.update_cube_measurement(
                m["cube_index"],
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
                m.get("num_faces"),
                camera_lateral_offset_mm=lateral_offset,
                rotation_source=m.get("rotation_source"),
            )

        # Feed face measurements (identity assigned earlier this frame),
        # then drop presence epochs that ended.
        for det in face_detections:
            self.world_map.update_face_measurement(
                det.face_id,
                det.position_mm,
                det.confidence,
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
                label=f"Face {det.face_id}",
            )
        if face_frame.ended is not None:
            self.world_map.forget_face(face_frame.ended)

        # Feed charger marker measurements (single static landmark — no
        # forgetting on visibility loss, COZMO-65; pose carries the dock
        # facing for alignment, COZMO-67).
        for det in charger_detections or []:
            self.world_map.update_charger_measurement(
                det.position_mm,
                det.rotation_matrix,
                det.confidence,
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
            )

        # Feed ArUco marker measurements (static printed landmarks keyed
        # by ID — no forgetting on visibility loss, like the charger).
        for det in aruco_detections or []:
            self.world_map.update_aruco_measurement(
                det.marker_id,
                det.position_mm,
                det.rotation_matrix,
                det.confidence,
                head_angle,
                det.size_mm,
                camera_lateral_offset_mm=lateral_offset,
            )

        # Push procedural face image to the browser
        anim_ctrl = getattr(cli, "anim_controller", None)
        if anim_ctrl is not None:
            face_im = getattr(anim_ctrl, "latest_face_image", None)
            if face_im is not None:
                self.sink.push_face_image(face_im)

        self.sink.broadcast()

    def _process_floor(self, cli, img_bgr, head_angle_rad: float) -> None:
        """Project the floor, dispatch the event, publish image + occupancy."""
        if self.floor_pipeline is None:
            return

        robot_pose = cli.pose if hasattr(cli, "pose") else None
        floor_img, occ_grid, occ_map_image = self.floor_pipeline.process(img_bgr, head_angle_rad, robot_pose)
        if floor_img is None:
            return

        cli.dispatch(event.EvtFloorProjectedImage, cli, floor_img, occ_grid)
        self.sink.push_floor_frame(floor_img)

        # Publish the occupancy map image for the browser overlay
        occ = self.floor_pipeline.occupancy_map
        if occ_map_image is not None and self.floor_pipeline.occupancy_enabled and occ is not None:
            self.sink.push_occupancy_frame(occ_map_image, occ.origin[0], occ.origin[1])

        # Sync floor grid dimensions to browser on first frame
        fp = self.floor_pipeline.floor_projector
        if fp is not None:
            size, center = fp.floor_size, fp.floor_center
            self.sink.set_floor_grid((float(size[0]), float(size[1])), (float(center[0]), float(center[1])))
