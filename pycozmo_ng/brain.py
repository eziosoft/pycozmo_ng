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
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.face_identity import FaceIdentity
from pycozmo_ng.vision.frame_stage import FrameStage
from pycozmo_ng.vision.frame_state import FrameState
from pycozmo_ng.vision.occupancy_map.floor_pipeline import FloorPipeline
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

    def setup_vision(
        self,
        camera_calibration=None,
        calibration_timeout: float = 10.0,
        process_every_n_frames: int = 1,
        undistort: bool = False,
        enable_floor_projection: bool = True,
        floor_center_mm: tuple[float, float] = (200.0, 0.0),
        floor_size_mm: tuple[float, float] = (600.0, 600.0),
        mm_per_pixel: float = 5.0,
        enable_occupancy_map: bool = True,
        occupancy_grid_size_mm: float = 2000.0,
        occupancy_cell_size_mm: float = 50.0,
        occupancy_inflation_radius: int = 2,
        occupancy_vote_cap: int = 10,
        occupancy_valid_min_brightness: int = 8,
        occupancy_edge_margin_px: int = 3,
        floor_edge_margin_px: int = 4,
        enable_face_detection: bool = True,
        face_detect_every_n_frames: int = 1,
    ):
        """Set up the vision pipeline components and register event handlers.

        This wires together DetectionPipeline, AnnotationEngine,
        DetectionEventDispatcher, and FloorPipeline, then registers all
        event handlers on the client: the camera image handler plus the
        subsystem handlers (charger state, cube connection, robot state
        trail, pick-up).

        Requires a connection: if ``camera_calibration`` is None it is
        retrieved from the robot.

        Args:
            camera_calibration: CameraCalibration instance. If None,
                retrieves it from the robot (needs ``self.cli``).
            calibration_timeout: Seconds to wait for calibration retrieval.
            face_detect_every_n_frames: Run face detection every Nth
                processed frame (1 = every frame). Higher values reduce
                CPU load — faces are static enough that every 3rd frame
                is usually sufficient.
        """
        if self.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() before setup_vision")

        if camera_calibration is None:
            from pycozmo_ng.vision.calibration.camera_calibration import (
                CameraCalibrationRetriever,
            )

            retriever = CameraCalibrationRetriever(self.cli)
            camera_calibration = retriever.get_calibration(timeout=calibration_timeout)

        self.camera_calibration = camera_calibration

        # Detection pipeline (core detection logic) - uses factory for DI
        self.detection_pipeline = create_default_detection_pipeline(
            camera_calibration=camera_calibration,
        )
        self.detection_pipeline.enable_face_detection = enable_face_detection

        # Annotation engine (pure drawing)
        self.annotation_engine = AnnotationEngine()

        # Event dispatcher (cube/face events)
        self.event_dispatcher = DetectionEventDispatcher()

        # Face identity (presence epochs) — fresh ID numbering per session
        self.face_identity = FaceIdentity()

        # Floor pipeline (optional)
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

        # Hand every per-session vision component to the frame stage in
        # one call — it owns the whole camera-frame sequence from here.
        self.frame_stage.configure(
            detection_pipeline=self.detection_pipeline,
            annotation_engine=self.annotation_engine,
            event_dispatcher=self.event_dispatcher,
            face_identity=self.face_identity,
            floor_pipeline=self.floor_pipeline,
            process_every_n_frames=process_every_n_frames,
            undistort=undistort,
            face_detect_every_n_frames=face_detect_every_n_frames,
        )

        # Expose occupancy map to web viewer for A* pathfinding
        if self.floor_pipeline and self.floor_pipeline.occupancy_map is not None:
            self.web_viewer.occupancy_map = self.floor_pipeline.occupancy_map
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
        self.cli.add_handler(event.EvtRobotStateUpdated, self._on_robot_state)
        self.cli.add_handler(event.EvtRobotPickedUpChange, self._on_picked_up)

        logger_cubes.info(
            f"Vision pipeline initialized (processing every "
            f"{process_every_n_frames} frame{'s' if process_every_n_frames > 1 else ''})"
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
            dock=self.navigator.dock,
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
        logger.info("Z to dock: drive back to start position")
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
