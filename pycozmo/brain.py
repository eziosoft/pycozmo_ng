"""Brain — internal orchestrator for the Cozmo robot.

Creates typed state containers and wires them to domain classes.
Domain classes receive the specific state container they need, not
a reference to Brain itself. This is an INTERNAL class - use Cozmo
as the public API.
"""

import time
from typing import ClassVar

import cv2
import numpy as np
from PIL import Image

import pycozmo
from pycozmo import event
from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.cubes.manager import CubeManager
from pycozmo.display.expressions import Happiness
from pycozmo.logger import logger_cubes
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.navigator import Navigator
from pycozmo.protocol import protocol_encoder
from pycozmo.vision.annotation import AnnotationEngine
from pycozmo.vision.detection_pipeline import (
    DetectionPipeline,
    create_default_detection_pipeline,
)
from pycozmo.vision.events import DetectionEventDispatcher
from pycozmo.vision.floor_pipeline import FloorPipeline
from pycozmo.vision.frame_handler import FrameHandler
from pycozmo.vision.frame_state import FrameState
from pycozmo.web.streamer import WebStreamer
from pycozmo.web.viewer import WebBridge
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

        # Web viewer
        self.web_viewer = WebViewerServer()
        self.web_streamer = WebStreamer(self.web_viewer, self.frame_state)

        # Vision components (composition root)
        self.detection_pipeline: DetectionPipeline | None = None
        self.annotation_engine: AnnotationEngine | None = None
        self.event_dispatcher: DetectionEventDispatcher | None = None
        self.floor_pipeline: FloorPipeline | None = None

        # Remote control (COZMO-25)
        self._remote_control = None

        # Domain sub-components — each receives the state it needs
        self.frame_handler = FrameHandler(
            self.frame_state, self.cube_tracking, self.detection_pipeline
        )
        self.navigator = Navigator(self.dock_state)
        self.cubes = CubeManager(self.cube_tracking)
        self.web = WebBridge(
            self.frame_state,
            self.cube_tracking,
            self.dock_state,
            self.detection_pipeline,
        )

        # Wire cross-domain references
        self.frame_handler.floor_pipeline = None  # Set after setup_vision
        self.navigator.cli = self.cli
        self.navigator.web_viewer = self.web_viewer
        self.web.web_viewer = self.web_viewer
        self.web_streamer.web_viewer = self.web_viewer  # For remote control listeners

    @property
    def cli(self):
        return self._cli

    @cli.setter
    def cli(self, value):
        self._cli = value
        self.navigator.cli = value
        self.web.cli = value

    def setup_vision(
        self,
        cli,
        camera_calibration,
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
        face_detect_every_n_frames: int = 1,
    ):
        """Set up the vision pipeline components.

        This wires together DetectionPipeline, AnnotationEngine,
        DetectionEventDispatcher, and FloorPipeline. Event handlers
        are registered on the client.

        Args:
            face_detect_every_n_frames: Run face detection every Nth
                processed frame (1 = every frame). Higher values reduce
                CPU load — faces are static enough that every 3rd frame
                is usually sufficient.
        """

        self._face_detect_every_n_frames = face_detect_every_n_frames

        # Detection pipeline (core detection logic) - uses factory for DI
        self.detection_pipeline = create_default_detection_pipeline(
            camera_calibration=camera_calibration,
        )
        self.detection_pipeline.enable_face_detection = enable_face_detection

        # Update frame handler with the new detection pipeline
        self.frame_handler.detection_pipeline = self.detection_pipeline

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

        # Wire floor pipeline to FrameHandler
        self.frame_handler.floor_pipeline = self.floor_pipeline

        # Wire web bridge to use detection pipeline
        self.web.detection_pipeline = self.detection_pipeline

        # Register camera image handler
        def on_camera_image(cli_arg, image):
            self._on_camera_image(
                cli_arg,
                image,
                process_every_n_frames,
                undistort,
            )

        cli.add_handler(event.EvtNewRawCameraImage, on_camera_image)

        logger_cubes.info(
            f"Vision pipeline initialized (processing every "
            f"{process_every_n_frames} frame{'s' if process_every_n_frames > 1 else ''})"
        )

    def _on_camera_image(
        self,
        cli,
        image,
        process_every_n_frames: int,
        undistort: bool,
    ):
        """Handler for camera images — runs the full vision pipeline."""
        self.frame_handler.on_raw_camera_image(cli, image)
        self.web_streamer.push_raw_frame(image)

        # Frame skipping for performance
        if not hasattr(self, "_vision_frame_counter"):
            self._vision_frame_counter = 0
        self._vision_frame_counter += 1
        if self._vision_frame_counter % process_every_n_frames != 0:
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
            if undistort:
                undistorted_bgr = cv2.undistort(
                    img_bgr,
                    self.detection_pipeline.camera_calibration.camera_matrix,
                    self.detection_pipeline.camera_calibration.distortion_coefficients,
                )
            else:
                undistorted_bgr = img_bgr

            # Convert to PIL for annotation (only one conversion needed)
            undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)
            undistorted_image = Image.fromarray(undistorted_rgb)

            # Detect cubes — pass BGR numpy directly (avoids re-conversion)
            cube_detections = self.detection_pipeline.detect_cubes(
                undistorted_bgr, head_angle_rad
            )

            # Detect faces (every Nth processed frame)
            if not hasattr(self, "_face_frame_counter"):
                self._face_frame_counter = 0
            self._face_frame_counter += 1
            face_every = getattr(self, "_face_detect_every_n_frames", 1)
            if self._face_frame_counter % face_every == 0:
                face_detections = self.detection_pipeline.detect_faces(
                    undistorted_bgr, head_angle_rad
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

            # Update frame state via frame handler
            self.frame_handler.on_annotated_camera_image(
                cli, annotated_image, cube_detections
            )

            # Push to web streamer
            self.web_streamer.push_annotated_frame(annotated_image, cube_detections)

            # Dispatch cube/face events
            self.event_dispatcher.dispatch_cube_events(cli, cube_detections)
            self.event_dispatcher.dispatch_face_events(cli, face_detections)

            # Set face expression based on detection
            anim_ctrl = getattr(cli, "anim_controller", None)
            if anim_ctrl is not None:
                if face_detections:
                    anim_ctrl.set_face_expression(Happiness())
                elif self._face_frame_counter % face_every == 0:
                    # Only clear when we actually ran detection and found nothing
                    anim_ctrl.clear_face_expression()

            # Feed cube measurements to web viewer for 3D display
            self.web.update_web_viewer()

            # Floor projection (optional) — pass BGR numpy directly
            if self.floor_pipeline is not None:
                robot_pose = cli.pose if hasattr(cli, "pose") else None
                floor_img, occ_grid, occ_map_image = self.floor_pipeline.process(
                    undistorted_bgr, head_angle_rad, robot_pose
                )
                if floor_img is not None:
                    cli.dispatch(event.EvtFloorProjectedImage, cli, floor_img, occ_grid)
                    # Store occupancy map image for web viewer
                    self.frame_state.latest_occupancy_map_image = occ_map_image

                    # Update frame handler with floor image
                    self.frame_handler.on_floor_projected_image(
                        cli, floor_img, occ_grid
                    )

                    # Push to web streamer
                    self.web_streamer.push_floor_frame(
                        floor_img, occ_grid, self.floor_pipeline
                    )

        except (OSError, ValueError, RuntimeError) as e:
            logger_cubes.error(f"Error processing camera image: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Camera lifecycle (COZMO-22)
    # ------------------------------------------------------------------

    def enable_camera_persistent(self, cli, resolution):
        """Enable the Cozmo camera at startup and keep it on regardless of viewers."""
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
        print("Cozmo camera enabled at startup")

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
        from pycozmo.remote_control import RemoteControl

        self._remote_control = RemoteControl(cli)
        self._remote_control.start()
        self.web_viewer.set_control_listener(self._remote_control.update_keys)
        self.web_viewer.set_navigate_listener(self.navigator._on_navigate_to)
        self.web_viewer.set_approach_listener(self.navigator._on_approach_cube)
        self.web_viewer.set_dock_listener(self.navigator._on_dock)
        self.web_viewer.set_camera_settings_listener(self._on_camera_settings)
        if self.cli is not None:
            self.dock_state.dock_pose = self.cli.pose
        print("Browser keyboard control enabled: WASD drive, R/F head, T/G lift")
        print("Click-to-navigate: click ground to drive there")
        print("Click cube to approach and align")
        print("Z to dock: drive back to start position")

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
        self.web_viewer.update_camera_settings(**settings)

        print(
            f"Camera settings updated: auto_exposure={settings.get('auto_exposure_enabled')}, "
            f"gain={settings.get('gain')}, exposure={settings.get('exposure_ms')}ms, "
            f"color={'color' if color else 'gray'}, resolution={res_name}, "
            f"head_light={'on' if settings.get('head_light') else 'off'}"
        )

    def stop_remote_control(self) -> None:
        if self._remote_control is not None:
            self._remote_control.stop()
            self._remote_control = None

    # ------------------------------------------------------------------
    # Pick-up handler
    # ------------------------------------------------------------------

    def _on_picked_up(self, c, state):
        if state:
            self.web_viewer.clear_path()

    # ------------------------------------------------------------------
    # Web streamer lifecycle
    # ------------------------------------------------------------------

    def start_web_viewer(self, port: int = 8765) -> None:
        """Start the web viewer server."""
        self.web_viewer.port = port
        self.web_streamer.start()

    def stop_web_viewer(self) -> None:
        """Stop the web viewer server."""
        self.web_streamer.stop()

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
