"""Brain — internal orchestrator for the Cozmo robot.

Creates typed state containers and wires them to domain classes.
Domain classes receive the specific state container they need, not
a reference to Brain itself. This is an INTERNAL class - use Cozmo
as the public API.
"""

import time

import pycozmo_ng
from pycozmo_ng import event
from pycozmo_ng.camera.service import CameraService
from pycozmo_ng.cubes.carry_state import CarryDetector
from pycozmo_ng.cubes.cube_tracking_state import CubeTrackingState
from pycozmo_ng.cubes.identity import to_world_index
from pycozmo_ng.cubes.indicator import CubeIndicator
from pycozmo_ng.cubes.manager import CubeManager
from pycozmo_ng.logger import logger, logger_cubes
from pycozmo_ng.navigation.astar import AStarPlanner
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.motion_controller import MotionController
from pycozmo_ng.navigation.navigator import Navigator
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.protocol import protocol_encoder
from pycozmo_ng.util.robot import RobotStatusFlag
from pycozmo_ng.vision.annotation import AnnotationEngine
from pycozmo_ng.vision.cube_detector.detection_pipeline import (
    DetectionPipeline,
    create_default_detection_pipeline,
)
from pycozmo_ng.vision.distance_rings import DistanceRings
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.face_identity import FaceIdentity
from pycozmo_ng.vision.frame_stage import FrameStage
from pycozmo_ng.vision.frame_state import FrameState
from pycozmo_ng.vision.occupancy_map.floor_pipeline import FloorPipeline
from pycozmo_ng.vision.vision_setup import VisionSetup
from pycozmo_ng.web.viewer_server import WebViewerServer


class Brain:
    """Internal brain — orchestrates vision, navigation, cubes, and web viewer.

    Creates typed state containers and passes them to domain classes.
    Brain is an internal implementation detail; use Cozmo class as public API.
    """

    def __init__(self):
        # Robot connection (backing field for property)
        self._cli = None

        # Typed state containers
        self.frame_state = FrameState()
        self.cube_tracking = CubeTrackingState()
        self.dock_state = DockState()

        # World map — the single source of truth. Owned here (the
        # composition root), not by the web adapter, and injected into
        # every module that reads or writes it.
        self.world_map = WorldMap()

        # Camera lifecycle — single owner of enable/disable, the settings
        # vocabulary, and what is actually in force (COZMO-22, COZMO-30).
        self.camera = CameraService()

        # Web viewer (thin adapter over the world map + camera service)
        self.web_viewer = WebViewerServer(world_map=self.world_map, camera=self.camera)

        # Vision components (composition root)
        self.detection_pipeline: DetectionPipeline | None = None
        self.annotation_engine: AnnotationEngine | None = None
        self.distance_rings: DistanceRings | None = None
        self.event_dispatcher: DetectionEventDispatcher | None = None
        self.face_identity: FaceIdentity | None = None
        self.floor_pipeline: FloorPipeline | None = None
        self.camera_calibration = None  # Set by setup_vision()

        # Remote control (COZMO-25)
        self._remote_control = None

        # Per-frame vision stage — owns the whole camera-frame sequence
        # (convert → detect → annotate → publish → events → world map).
        # Configured by setup_vision() once the detectors exist.
        self.frame_stage = FrameStage(self.frame_state, self.world_map, self.web_viewer)

        # Domain sub-components — each receives the state it needs.
        # One motion seam (cancellation + driver lease) shared by the
        # navigation jobs and the browser remote control, so "who may
        # command the wheels" lives in exactly one object.
        self.motion = MotionController()
        # Cube carry state — after a grab, wait for the fork to finish
        # rising, then let visibility decide (see cubes/carry_state.py).
        self.carry = CarryDetector(
            get_cube_state=self.world_map.get_cube,
            get_lift_height=lambda: self.cli.lift_position.height.mm if self.cli is not None else 0.0,
            get_lift_in_pos=lambda: self.cli is None or bool(self.cli.robot_status & RobotStatusFlag.LIFT_IN_POS),
        )
        self.navigator = Navigator(
            self.dock_state,
            world_map=self.world_map,
            motion=self.motion,
            carry=self.carry,
        )
        self.cubes = CubeManager(self.cube_tracking)
        # Steady-state cube LEDs: dark while idle (battery), two blinks on
        # connect, dim blue when the camera sees a cube, red while a grab or
        # stack is working on one. Reads state the collaborators already hold.
        self.cube_indicator = CubeIndicator(
            self.cubes,
            world_map=self.world_map,
            visit=self.navigator.cube_visit,
        )
        # A cube whose accelerometer reports motion is no longer where the
        # world map last saw it — drop the stale pose.
        self.cubes.set_moved_handler(self._on_cube_moved)

        # Wire cross-domain references
        self.navigator.cli = self.cli

        # A* pathfinding
        self.astar_planner: AStarPlanner | None = None
        self.navigator.set_astar_planner(self.astar_planner)
        self.navigator.set_astar_enabled(False)

    @property
    def cli(self):
        return self._cli

    @cli.setter
    def cli(self, value):
        self._cli = value
        self.navigator.cli = value
        self.frame_stage.cli = value
        self.camera.cli = value

    def setup_vision(self, setup: VisionSetup | None = None) -> None:
        """Set up the vision pipeline components and register event handlers.

        This wires together DetectionPipeline, AnnotationEngine,
        DetectionEventDispatcher, and FloorPipeline, then registers all
        event handlers on the client: the camera image handler plus the
        subsystem handlers (charger state, cube connection, robot state
        trail, pick-up).

        Every option is declared once, in :class:`~pycozmo_ng.vision.vision_setup.VisionSetup`
        (frame gating, floor projection, occupancy map, face detection);
        the defaults apply when ``setup`` is None.

        Requires a connection: if the calibration is omitted it is
        retrieved from the robot.

        Args:
            setup: Vision options. Defaults to ``VisionSetup()``.
        """
        setup = setup or VisionSetup()

        if self.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() before setup_vision")

        camera_calibration = setup.camera_calibration
        if camera_calibration is None:
            from pycozmo_ng.vision.calibration.camera_calibration import (
                CameraCalibrationRetriever,
            )

            retriever = CameraCalibrationRetriever(self.cli)
            camera_calibration = retriever.get_calibration(timeout=setup.calibration_timeout)
            if camera_calibration is None:
                raise RuntimeError(
                    f"Camera calibration not retrieved from the robot within {setup.calibration_timeout}s"
                )

        self.camera_calibration = camera_calibration

        # Detection pipeline (core detection logic) - uses factory for DI
        self.detection_pipeline = create_default_detection_pipeline(
            camera_calibration=camera_calibration,
        )
        self.detection_pipeline.enable_face_detection = setup.enable_face_detection
        self.detection_pipeline.enable_charger_detection = setup.enable_charger_detection
        self.detection_pipeline.enable_aruco_detection = setup.enable_aruco_detection

        # Annotation engine (pure drawing)
        self.annotation_engine = AnnotationEngine()

        # Ground distance rings — assigned unconditionally so a re-setup
        # with the option off cannot leave the previous session's rings.
        self.distance_rings = None
        if setup.enable_distance_rings and setup.distance_ring_radii_mm:
            self.distance_rings = DistanceRings(
                camera_calibration.camera_matrix,
                radii_mm=setup.distance_ring_radii_mm,
            )

        # Event dispatcher (cube/face events)
        self.event_dispatcher = DetectionEventDispatcher()

        # Face identity (presence epochs) — fresh ID numbering per session
        self.face_identity = FaceIdentity()

        # Floor pipeline (optional)
        if setup.enable_floor_projection:
            self.floor_pipeline = FloorPipeline(
                camera_calibration=camera_calibration,
                enable_floor_projection=True,
                floor_center_mm=setup.floor_center_mm,
                floor_size_mm=setup.floor_size_mm,
                mm_per_pixel=setup.mm_per_pixel,
                enable_occupancy_map=setup.enable_occupancy_map,
                occupancy_grid_size_mm=setup.occupancy_grid_size_mm,
                occupancy_cell_size_mm=setup.occupancy_cell_size_mm,
                occupancy_inflation_radius=setup.occupancy_inflation_radius,
                occupancy_logodds_hit=setup.occupancy_logodds_hit,
                occupancy_logodds_miss=setup.occupancy_logodds_miss,
                occupancy_logodds_max=setup.occupancy_logodds_max,
                occupancy_logodds_decay=setup.occupancy_logodds_decay,
                occupancy_floor_distance_threshold=setup.occupancy_floor_distance_threshold,
                occupancy_floor_light_weight=setup.occupancy_floor_light_weight,
                occupancy_floor_model_ema=setup.occupancy_floor_model_ema,
                occupancy_valid_min_brightness=setup.occupancy_valid_min_brightness,
                occupancy_silhouette_margin_px=setup.occupancy_silhouette_margin_px,
                floor_edge_margin_px=setup.floor_edge_margin_px,
            )

        # Hand every per-session vision component to the frame stage in
        # one call — it owns the whole camera-frame sequence from here.
        self.frame_stage.configure(
            detection_pipeline=self.detection_pipeline,
            annotation_engine=self.annotation_engine,
            distance_rings=self.distance_rings,
            event_dispatcher=self.event_dispatcher,
            face_identity=self.face_identity,
            floor_pipeline=self.floor_pipeline,
            process_every_n_frames=setup.process_every_n_frames,
            undistort=setup.undistort,
            face_detect_every_n_frames=setup.face_detect_every_n_frames,
            cube_body_rate_gate_radps=setup.cube_body_rate_gate_radps,
            cube_head_rate_gate_radps=setup.cube_head_rate_gate_radps,
        )

        # Plan A* paths over the occupancy grid (Brain is the composition
        # root — the web server never holds the map)
        if self.floor_pipeline and self.floor_pipeline.occupancy_map is not None:
            self.astar_planner = AStarPlanner(self.floor_pipeline.occupancy_map, allow_unknown=True)
            self.navigator.set_astar_planner(self.astar_planner)
            self.navigator.set_astar_enabled(False)

        # Subsystem event handlers (vision, navigation, cubes, web trail,
        # pick-up)
        self.cli.add_handler(event.EvtNewRawCameraImage, self.frame_stage.on_camera_image)
        self.cli.add_handler(event.EvtRobotOnChargerChange, self.navigator.on_charger_change)
        self.cli.add_handler(event.EvtRobotChargingChange, self.navigator.on_charging_change)
        self.cli.add_handler(
            protocol_encoder.ObjectConnectionState,
            self.cubes.on_object_connection_state,
        )
        self.cli.add_handler(
            protocol_encoder.ObjectMoved,
            self.cubes.on_object_moved,
        )
        self.cli.add_handler(event.EvtRobotStateUpdated, self._on_robot_state)
        self.cli.add_handler(event.EvtRobotPickedUpChange, self._on_picked_up)

        logger_cubes.info(
            f"Vision pipeline initialized (processing every "
            f"{setup.process_every_n_frames} frame{'s' if setup.process_every_n_frames > 1 else ''})"
        )

    # ------------------------------------------------------------------
    # Remote control (COZMO-25)
    # ------------------------------------------------------------------

    def install_remote_control(self, cli) -> None:
        """Wire browser keyboard input to Cozmo's motors."""
        from pycozmo_ng.navigation.remote_control import RemoteControl

        self._remote_control = RemoteControl(cli, motion=self.motion)
        self._remote_control.start()
        self.web_viewer.router.set_actions(
            control=self._remote_control.update_keys,
            navigate=self.navigator.navigate_to,
            approach=self.navigator.approach_cube_at,
            # Z docks on the ArUco charger marker (id 0) — two-step
            # standoff + reverse (COZMO-66).
            dock=self.navigator.dock_on_aruco,
            camera_settings=self.camera.apply_settings,
            say=self._on_say,
            test_mode=self.navigator.set_test_mode,
            astar_toggle=self.navigator.set_astar_enabled,
        )
        if self.cli is not None:
            self.dock_state.dock_pose = self.cli.pose
        logger.info("Browser keyboard control enabled: WASD drive, R/F head, T/G lift")
        logger.info("Click-to-navigate: click ground to drive there")
        logger.info("Click cube to approach and grab it")
        logger.info("Click another cube while carrying one to stack them")
        logger.info("Z to dock on the charger (ArUco marker 0)")
        logger.info("A* pathfinding: toggle in the web viewer")

    # ------------------------------------------------------------------
    # TTS say (browser)
    # ------------------------------------------------------------------

    def _on_say(self, text: str) -> None:
        """Handle TTS say request from the browser."""
        cli = self.cli
        if cli is not None and text:
            try:
                cli.say(text)
            except Exception as e:  # noqa: BLE001
                logger.warning("say failed: %s", e)

    def stop_remote_control(self) -> None:
        if self._remote_control is not None:
            self._remote_control.stop()
            self._remote_control = None

    # ------------------------------------------------------------------
    # Cube motion handler (moved cube -> position unknown)
    # ------------------------------------------------------------------

    def _on_cube_moved(self, cube_id: int) -> None:
        """A cube's accelerometer saw motion: its world pose is now unknown.

        Registered on ``CubeManager.set_moved_handler``. The cube keeps its
        identity and its connection — only the fused position is dropped, and
        the next camera measurement starts a fresh pose from where the cube
        actually is now.
        """
        if self.world_map.forget_cube(to_world_index(cube_id)):
            logger_cubes.info("Cube %d moved - position reset to unknown", cube_id)
        else:
            logger_cubes.debug("Cube %d moved - no stored position to reset", cube_id)

    # ------------------------------------------------------------------
    # Robot state handler (trail + telemetry)
    # ------------------------------------------------------------------

    def _on_robot_state(self, c) -> None:
        """Push robot pose into the world map for path tracking (COZMO-27)
        and the latest telemetry into the browser HUD."""
        try:
            pose = c.pose
            pos = pose.position.x_y_z
            yaw = pose.rotation.angle_z.radians
            self.world_map.update_robot_pose(pos, yaw, origin_id=pose.origin_id, record_path=True)
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to update robot pose: {e}")

        try:
            with self.dock_state.lock:
                on_charger = self.dock_state.on_charger
                charging = self.dock_state.charging
            self.web_viewer.update_telemetry(
                charging=charging,
                on_charger=on_charger,
                battery_voltage=round(c.battery_voltage, 3),
                left_wheel_speed=round(c.left_wheel_speed.mmps, 1),
                right_wheel_speed=round(c.right_wheel_speed.mmps, 1),
                head_angle=round(c.head_angle.degrees, 1),
                lift_height=round(c.lift_position.height.mm, 1),
                cliff_data=c.cliff_data,
                # Carry status for the HUD — evaluated on demand at push
                # time (see cubes/carry_state.py).
                carry=self.carry.state.value,
                carry_cube=self.carry.cube_id,
            )
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to update telemetry: {e}")

        self.web_viewer.broadcast()

    # ------------------------------------------------------------------
    # Pick-up handler
    # ------------------------------------------------------------------

    def _on_picked_up(self, c, state):
        if state:
            self.world_map.reset_path()

    # ------------------------------------------------------------------
    # Web streamer lifecycle
    # ------------------------------------------------------------------

    def start_web_viewer(self, port: int = 8765) -> None:
        """Start the web viewer server."""
        self.web_viewer.port = port
        self.web_viewer.start()

    def stop_web_viewer(self) -> None:
        """Stop the web viewer server."""
        self.web_viewer.close()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self, cli):
        """Main loop, headless — all UI lives in the web viewer."""
        _CONNECTED = pycozmo_ng.conn.Connection.CONNECTED
        fs = self.frame_state

        while fs.running:
            if cli.conn.state != _CONNECTED:
                logger_cubes.warning(
                    "Connection state is %s - exiting main loop",
                    cli.conn.state,
                )
                fs.running = False
                break

            if self.camera.active and self.camera.enabled_at is not None and time.time() - self.camera.enabled_at > 5.0:
                stale_for = time.time() - fs.last_frame_time
                if stale_for > 5.0:
                    logger_cubes.warning(
                        "No camera frames for %.1fs - assuming Wi-Fi drop, exiting main loop",
                        stale_for,
                    )
                    fs.running = False
                    break

            time.sleep(0.05)

        self.stop_remote_control()
        self.stop_web_viewer()
