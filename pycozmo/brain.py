"""Brain — internal orchestrator for the Cozmo robot.

Creates typed state containers and wires them to domain classes.
Domain classes receive the specific state container they need, not
a reference to Brain itself. This is an INTERNAL class - use Cozmo
as the public API.
"""

import time
from typing import ClassVar

import pycozmo
from pycozmo import event
from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.cubes.manager import CubeManager
from pycozmo.logger import logger, logger_cubes
from pycozmo.navigation.astar import AStarPlanner
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.navigator import Navigator
from pycozmo.navigation.world_map import WorldMap
from pycozmo.protocol import protocol_encoder
from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.cube_detector.detection_pipeline import (
    DetectionPipeline,
    create_default_detection_pipeline,
)
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.frame_stage import FrameStage
from pycozmo.vision.frame_state import FrameState
from pycozmo.vision.occupancy_map.floor_pipeline import FloorPipeline
from pycozmo.web.viewer_server import WebViewerServer


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

        # Web viewer (thin adapter over the world map)
        self.web_viewer = WebViewerServer(world_map=self.world_map)

        # Vision components (composition root)
        self.detection_pipeline: DetectionPipeline | None = None
        self.annotation_engine: AnnotationEngine | None = None
        self.event_dispatcher: DetectionEventDispatcher | None = None
        self.floor_pipeline: FloorPipeline | None = None
        self.camera_calibration = None  # Set by setup_vision()

        # Remote control (COZMO-25)
        self._remote_control = None

        # Per-frame vision stage — owns the whole camera-frame sequence
        # (convert → detect → annotate → publish → events → world map).
        # Configured by setup_vision() once the detectors exist.
        self.frame_stage = FrameStage(
            self.frame_state, self.cube_tracking, self.world_map, self.web_viewer
        )

        # Domain sub-components — each receives the state it needs
        self.navigator = Navigator(self.dock_state, world_map=self.world_map)
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
            raise RuntimeError(
                "Not connected - call Cozmo.connect() before setup_vision"
            )

        if camera_calibration is None:
            from pycozmo.vision.calibration.camera_calibration import (
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
            floor_pipeline=self.floor_pipeline,
            process_every_n_frames=process_every_n_frames,
            undistort=undistort,
            face_detect_every_n_frames=face_detect_every_n_frames,
        )

        # Expose occupancy map to web viewer for A* pathfinding
        if self.floor_pipeline and self.floor_pipeline.occupancy_map is not None:
            self.web_viewer.occupancy_map = self.floor_pipeline.occupancy_map
            self.astar_planner = AStarPlanner(
                self.floor_pipeline.occupancy_map, allow_unknown=True
            )
            self.navigator.set_astar_planner(self.astar_planner)
            self.navigator.set_astar_enabled(False)

        # Subsystem event handlers (vision, navigation, cubes, web trail,
        # pick-up)
        self.cli.add_handler(
            event.EvtNewRawCameraImage, self.frame_stage.on_camera_image
        )
        self.cli.add_handler(
            event.EvtRobotOnChargerChange, self.navigator.on_charger_change
        )
        self.cli.add_handler(
            event.EvtRobotChargingChange, self.navigator.on_charging_change
        )
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
    # Camera lifecycle (COZMO-22)
    # ------------------------------------------------------------------

    def enable_camera_persistent(self, cli, resolution):
        """Enable the Cozmo camera at startup and keep it on regardless of viewers.

        Uses Stream mode (Cozmo's native ~30 fps). The camera stays on for
        the whole session; this is the always-on counterpart to the lazy
        camera lifecycle (COZMO-22), which toggles a second stream on/off
        based on browser-tab activity.
        """
        fs = self.frame_state
        fs.camera_resolution = resolution
        cli.enable_camera(
            enable=True,
            color=False,
            resolution=resolution,
        )
        cli.set_camera_params(
            gain=0.0,
            exposure_ms=20,
            auto_exposure_enabled=True,
        )
        fs.camera_active = True
        fs.camera_enabled_at = time.time()
        logger.info("Cozmo camera enabled at startup")

    def post_calibration(self, cli, resolution) -> None:
        """Return the camera to live auto-exposure after a manual calibration.

        The camera stays enabled (always-on by default); only the temporary
        manual-exposure settings used during calibration are undone.
        """
        fs = self.frame_state
        fs.camera_resolution = resolution
        cli.enable_camera(
            enable=True,
            color=False,
            resolution=resolution,
        )
        cli.set_camera_params(
            gain=0.0,
            exposure_ms=20,
            auto_exposure_enabled=True,
        )
        fs.camera_active = True
        fs.camera_enabled_at = time.time()

    # ------------------------------------------------------------------
    # Remote control (COZMO-25)
    # ------------------------------------------------------------------

    def install_remote_control(self, cli) -> None:
        """Wire browser keyboard input to Cozmo's motors."""
        from pycozmo.navigation.remote_control import RemoteControl

        self._remote_control = RemoteControl(cli)
        self._remote_control.start()
        self.web_viewer.router.set_actions(
            control=self._remote_control.update_keys,
            navigate=self.navigator.navigate_to,
            approach=self.navigator.approach_cube_at,
            dock=self.navigator.dock,
            camera_settings=self._on_camera_settings,
            say=self._on_say,
            test_mode=self.navigator.set_test_mode,
            astar_toggle=self.navigator.set_astar_enabled,
        )
        if self.cli is not None:
            self.dock_state.dock_pose = self.cli.pose
        logger.info("Browser keyboard control enabled: WASD drive, R/F head, T/G lift")
        logger.info("Click-to-navigate: click ground to drive there")
        logger.info("Click cube to approach and align")
        logger.info("Z to dock: drive back to start position")
        logger.info("A* pathfinding: toggle in the web viewer")

    # ------------------------------------------------------------------
    # Camera settings from browser (COZMO-30)
    # ------------------------------------------------------------------

    # Map resolution name strings to protocol enum values.
    _RESOLUTION_MAP: ClassVar[dict[str, int]] = {
        "QQQQVGA": protocol_encoder.ImageResolution.QQQQVGA,
        "QQQVGA": protocol_encoder.ImageResolution.QQQVGA,
        "QQVGA": protocol_encoder.ImageResolution.QQVGA,
        "QVGA": protocol_encoder.ImageResolution.QVGA,
        "CVGA": protocol_encoder.ImageResolution.CVGA,
        "VGA": protocol_encoder.ImageResolution.VGA,
        "SVGA": protocol_encoder.ImageResolution.SVGA,
        "XGA": protocol_encoder.ImageResolution.XGA,
        "SXGA": protocol_encoder.ImageResolution.SXGA,
        "UXGA": protocol_encoder.ImageResolution.UXGA,
        "QXGA": protocol_encoder.ImageResolution.QXGA,
        "QUXGA": protocol_encoder.ImageResolution.QUXGA,
    }

    def _on_camera_settings(self, settings: dict) -> None:
        """Apply camera settings received from the browser."""
        cli = self.cli
        if cli is None:
            return

        # Apply sensor parameters (gain, exposure, auto-exposure)
        cli.set_camera_params(
            gain=settings.get("gain", 0.0),
            exposure_ms=settings.get("exposure_ms", 100),
            auto_exposure_enabled=settings.get("auto_exposure_enabled", True),
        )

        # Apply head light (IR illuminator)
        cli.set_head_light(settings.get("head_light", False))

        # Apply color mode and resolution (requires re-enabling camera)
        color = settings.get("color", False)
        res_name = settings.get("resolution", "QVGA")
        resolution = self._RESOLUTION_MAP.get(res_name)
        if resolution is None:
            resolution = protocol_encoder.ImageResolution.QVGA

        cli.enable_camera(
            enable=True,
            color=color,
            resolution=resolution,
        )

        # Store resolution in frame state
        self.frame_state.camera_resolution = resolution

        # Echo settings back so the broadcast includes the latest values
        self.web_viewer.router.set_camera_settings(settings)

        logger.info(
            "Camera settings updated: auto_exposure=%s, gain=%s, exposure=%sms, "
            "color=%s, resolution=%s, head_light=%s",
            settings.get("auto_exposure_enabled"),
            settings.get("gain"),
            settings.get("exposure_ms"),
            "color" if color else "gray",
            res_name,
            "on" if settings.get("head_light") else "off",
        )

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
            self.world_map.update_robot_pose(
                pos, yaw, origin_id=pose.origin_id, record_path=True
            )
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
        _CONNECTED = pycozmo.conn.Connection.CONNECTED
        fs = self.frame_state

        while fs.running:
            if cli.conn.state != _CONNECTED:
                logger_cubes.warning(
                    "Connection state is %s - exiting main loop",
                    cli.conn.state,
                )
                fs.running = False
                break

            if (
                fs.camera_active
                and fs.camera_enabled_at is not None
                and time.time() - fs.camera_enabled_at > 5.0
            ):
                stale_for = time.time() - fs.last_frame_time
                if stale_for > 5.0:
                    logger_cubes.warning(
                        "No camera frames for %.1fs - assuming Wi-Fi drop, "
                        "exiting main loop",
                        stale_for,
                    )
                    fs.running = False
                    break

            time.sleep(0.05)

        self.stop_remote_control()
        self.stop_web_viewer()
