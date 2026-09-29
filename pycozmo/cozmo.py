"""
Cozmo — public API for controlling the Cozmo robot.

This is the main entry point for library users. It provides a clean,
high-level interface to all robot capabilities while hiding internal
implementation details.

Usage:
    from pycozmo import Cozmo

    cozmo = Cozmo()
    with cozmo.connect():
        cozmo.vision.start()
        cozmo.navigator.drive_to(x=100, y=0)
        cozmo.run()
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

import pycozmo
from pycozmo.brain import Brain
from pycozmo.cubes.identity import to_cube_id, to_world_index
from pycozmo.logger import logger, setup_basic_logging
from pycozmo.navigation.world_map import CubeState, WorldMap
from pycozmo.protocol import protocol_encoder
from pycozmo.vision.cube_detector.detection_pipeline import DetectionPipeline
from pycozmo.web.viewer_server import WebViewerServer


class VisionAPI:
    """High-level vision interface: start/stop, frame access, detections."""

    def __init__(self, brain: Brain):
        self._brain = brain

    def start(self, resolution=None, **kwargs) -> None:
        """Set up the vision pipeline and enable the camera.

        Convenience for the common case — equivalent to calling
        :meth:`Cozmo.setup_vision` followed by :meth:`Cozmo.enable_camera`.

        Args:
            resolution: ImageResolution enum (default: QVGA).
            **kwargs: Passed through to :meth:`Cozmo.setup_vision`.

        Raises:
            RuntimeError: If not connected to the robot.
        """
        brain = self._brain
        if brain.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() first")
        brain.setup_vision(**kwargs)
        brain.camera.enable(resolution)

    def stop(self) -> None:
        """Stop the camera stream (no-op when not connected)."""
        brain = self._brain
        if brain.cli is None:
            return
        brain.camera.disable()

    @property
    def annotation(self):
        """Access the annotation engine for configuring display options."""
        return self._brain.annotation_engine

    @property
    def latest_frame(self):
        """Get the latest raw camera frame."""
        return self._brain.frame_state.latest_raw_frame

    @property
    def latest_annotated_frame(self):
        """Get the latest annotated camera frame."""
        return self._brain.frame_state.latest_annotated_frame

    @property
    def latest_detections(self):
        """Get the latest cube detections."""
        with self._brain.frame_state.detections_lock:
            return list(self._brain.frame_state.latest_detections)

    @property
    def detection_pipeline(self) -> DetectionPipeline | None:
        """Access the underlying detection pipeline for advanced use."""
        return self._brain.detection_pipeline

    @property
    def camera_calibration(self):
        """Get the camera calibration (available after setup_vision)."""
        return self._brain.camera_calibration


class CubesAPI:
    """High-level cube interface (cube IDs are 1-3).

    Cube LED control (``set_lights``) is not implemented yet — there is
    no cube object model; :meth:`start_lights` runs the BLE keepalive
    cycler that holds the cubes' connections alive.
    """

    def __init__(self, brain: Brain):
        self._brain = brain

    @property
    def connected(self) -> list[int]:
        """Cube IDs (1-3) currently connected over BLE."""
        return list(self._brain.cubes.connected_cubes.keys())

    def discover(self) -> None:
        """Discover available cubes and build the factory_id mapping."""
        self._require_connection()
        self._brain.cubes.setup_cube_mapping(self._brain.cli)

    def start_lights(self) -> None:
        """Start the cube BLE keepalive cycler."""
        self._require_connection()
        self._brain.cubes.start_cube_light_worker(self._brain.cli)

    def stop_lights(self) -> None:
        """Stop the cube BLE keepalive cycler."""
        self._brain.cubes.stop_cube_light_worker()

    def _require_connection(self) -> None:
        if self._brain.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() first")


class NavigatorAPI:
    """High-level navigation interface (world positions in mm, cube IDs 1-3)."""

    def __init__(self, brain: Brain):
        self._brain = brain
        self._navigator = brain.navigator

    def drive_to(self, x: float, y: float, **kwargs) -> bool:
        """Drive to a world position (mm)."""
        return self._navigator.drive_to_point(x, y, **kwargs)

    def approach_cube(self, cube_id: int) -> bool:
        """Approach and align with the cube with the given ID (1-3).

        Returns True if the approach was started (the cube is in the
        world map), False otherwise.
        """
        return self._navigator.approach_cube(cube_id)

    def dock(self) -> None:
        """Drive back to dock/origin (in the background)."""
        self._navigator.dock()

    def stop(self) -> None:
        """Stop all movement."""
        self._navigator.stop()

    def set_astar_enabled(self, enabled: bool) -> None:
        """Enable or disable A* obstacle-avoiding pathfinding."""
        self._navigator.set_astar_enabled(enabled)

    def is_astar_enabled(self) -> bool:
        """Return whether A* pathfinding is currently enabled."""
        return self._navigator.astar.enabled


class WorldMapAPI:
    """Read-only access to the world map (world frame, mm).

    Cube-facing members use cube IDs 1-3 like the rest of the public
    API; conversion to the world-map index (0-2) happens here, through
    ``pycozmo.cubes.identity``.
    """

    def __init__(self, brain: Brain):
        self._world_map: WorldMap = brain.world_map

    @property
    def robot_pose(self):
        """Get current robot pose (position, yaw)."""
        return self._world_map.get_robot_pose()

    @property
    def cubes(self) -> dict[int, CubeState]:
        """Get all known cube poses, keyed by cube ID (1-3)."""
        return {
            to_cube_id(cube_index): state
            for cube_index, state in self._world_map.get_cubes().items()
        }

    @property
    def path(self):
        """Get the robot's path trail (world frame, mm)."""
        return self._world_map.get_path_points()

    def get_cube(self, cube_id: int) -> CubeState | None:
        """Get one cube's world state by cube ID (1-3), or None if unknown.

        Raises:
            ValueError: If ``cube_id`` is outside 1-3.
        """
        return self._world_map.get_cube(to_world_index(cube_id))


class SensorsAPI:
    """Read-only access to robot sensors."""

    def __init__(self, brain: Brain):
        self._brain = brain

    @property
    def battery_voltage(self) -> float:
        cli = self._brain.cli
        return cli.battery_voltage if cli else 0.0

    @property
    def imu(self):
        """Get IMU data (accel, gyro)."""
        cli = self._brain.cli
        if cli:
            return {
                "accel": cli.accel,
                "gyro": cli.gyro,
            }
        return {"accel": (0, 0, 0), "gyro": (0, 0, 0)}

    @property
    def cliff(self):
        """Get cliff sensor data."""
        cli = self._brain.cli
        return cli.cliff_data if cli else None

    @property
    def wheel_speeds(self):
        """Get wheel speeds (mm/s)."""
        cli = self._brain.cli
        if cli:
            return {
                "left": cli.left_wheel_speed.mmps,
                "right": cli.right_wheel_speed.mmps,
            }
        return {"left": 0, "right": 0}

    @property
    def head_angle(self) -> float:
        """Get head angle in degrees."""
        cli = self._brain.cli
        return cli.head_angle.degrees if cli else 0.0

    @property
    def lift_height(self) -> float:
        """Get lift height in mm."""
        cli = self._brain.cli
        return cli.lift_position.height.mm if cli else 0.0


class WebViewerAPI:
    """High-level web viewer interface."""

    def __init__(self, brain: Brain):
        self._brain = brain
        self._web_viewer: WebViewerServer = brain.web_viewer

    def start(self, port: int = 8765):
        """Start the web viewer server."""
        self._brain.start_web_viewer(port)

    def stop(self):
        """Stop the web viewer server."""
        self._brain.stop_web_viewer()

    @property
    def is_running(self) -> bool:
        """Whether the web viewer server is running."""
        return self._web_viewer.is_running


class Cozmo:
    """Main entry point for the PyCozmo library.

    Provides a clean, high-level interface to the Cozmo robot.
    All subsystems are accessible as properties.

    Example:
        cozmo = Cozmo()
        with cozmo.connect():
            cozmo.vision.start()
            print(cozmo.sensors.battery_voltage)
            cozmo.navigator.drive_to(x=100, y=0)
    """

    def __init__(self):
        self._brain = Brain()
        self._connected = False
        self._cli = None

        # Public subsystem APIs (lazy-initialized)
        self._vision: VisionAPI | None = None
        self._cubes: CubesAPI | None = None
        self._navigator: NavigatorAPI | None = None
        self._world_map: WorldMapAPI | None = None
        self._sensors: SensorsAPI | None = None
        self._web_viewer: WebViewerAPI | None = None

    @property
    def vision(self) -> VisionAPI:
        if self._vision is None:
            self._vision = VisionAPI(self._brain)
        return self._vision

    @property
    def cubes(self) -> CubesAPI:
        if self._cubes is None:
            self._cubes = CubesAPI(self._brain)
        return self._cubes

    @property
    def navigator(self) -> NavigatorAPI:
        if self._navigator is None:
            self._navigator = NavigatorAPI(self._brain)
        return self._navigator

    @property
    def world_map(self) -> WorldMapAPI:
        if self._world_map is None:
            self._world_map = WorldMapAPI(self._brain)
        return self._world_map

    @property
    def sensors(self) -> SensorsAPI:
        if self._sensors is None:
            self._sensors = SensorsAPI(self._brain)
        return self._sensors

    @property
    def web_viewer(self) -> WebViewerAPI:
        if self._web_viewer is None:
            self._web_viewer = WebViewerAPI(self._brain)
        return self._web_viewer

    def connect(
        self,
        host: str = "172.31.1.1",
        port: int = 5551,
        **kwargs,
    ) -> Cozmo:
        """Connect to the Cozmo robot.

        Args:
            host: Robot IP address (default: 172.31.1.1 on Cozmo AP)
            port: Robot port (default: 5551)
            **kwargs: Passed to Client constructor.

        Returns:
            Self for chaining.
        """
        if self._connected:
            return self

        setup_basic_logging()
        self._cli = pycozmo.Client(
            robot_addr=(host, port),
            **kwargs,
        )
        self._cli.start()
        self._cli.connect()
        self._cli.wait_for_robot()
        self._brain.cli = self._cli
        self._connected = True
        return self

    def disconnect(self):
        """Disconnect from the robot."""
        if self._connected and self._cli:
            self._cli.disconnect()
            self._cli.stop()
            self._connected = False

    def __enter__(self) -> Self:
        """Context manager entry - connects and returns self."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disconnects."""
        self.disconnect()

    @property
    def cli(self):
        """Access the underlying pycozmo Client for advanced use."""
        return self._cli

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Camera control (convenience methods)
    # ------------------------------------------------------------------

    def enable_camera(self, resolution=None, **kwargs):
        """Enable the camera with optional resolution.

        Args:
            resolution: ImageResolution enum or name (default: QVGA)
            **kwargs: Camera settings overrides (gain, exposure_ms, ...)
        """
        self._brain.camera.enable(resolution, **kwargs)

    def set_head_angle(self, angle_deg: float, **kwargs):
        """Set head angle in degrees."""
        self._cli.set_head_angle(
            pycozmo.util.Angle(degrees=angle_deg).radians, **kwargs
        )

    def set_lift_height(self, height_mm: float, **kwargs):
        """Set lift height in mm."""
        self._cli.set_lift_height(height_mm, **kwargs)

    def drive_wheels(self, left_mmps: float, right_mmps: float, **kwargs):
        """Drive wheels at specified speeds (mm/s)."""
        self._cli.drive_wheels(left_mmps, right_mmps, **kwargs)

    def stop_all_motors(self):
        """Stop all motors."""
        self._cli.stop_all_motors()

    def say(self, text: str) -> None:
        """Say text through the Cozmo robot using offline TTS.

        Synthesizes text to a WAV file (cached for repeated calls)
        and plays it through the robot's audio pipeline.

        Args:
            text: The text to speak.
        """
        self._cli.say(text)

    # ------------------------------------------------------------------
    # Vision setup
    # ------------------------------------------------------------------

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
        enable_face_detection: bool = True,
        face_detect_every_n_frames: int = 1,
    ):
        """Set up the vision pipeline and register all internal event handlers.

        Requires a connection: when ``camera_calibration`` is None it is
        retrieved from the robot.

        Args:
            camera_calibration: CameraCalibration instance. If None, retrieves
                it from the robot automatically.
            calibration_timeout: Seconds to wait for calibration retrieval.
            process_every_n_frames: Run detection every Nth frame (1 = every frame).
            undistort: Apply lens distortion correction.
            enable_floor_projection: Enable bird's-eye floor view.
            floor_center_mm: Floor projection center (x, y) in mm.
            floor_size_mm: Floor projection size (w, h) in mm.
            mm_per_pixel: Floor projection resolution.
            enable_face_detection: Enable Haar cascade face detection.
            face_detect_every_n_frames: Run face detection every Nth processed
                frame (1 = every frame). Higher values reduce CPU load.
        """
        self._brain.setup_vision(
            camera_calibration=camera_calibration,
            calibration_timeout=calibration_timeout,
            process_every_n_frames=process_every_n_frames,
            undistort=undistort,
            enable_floor_projection=enable_floor_projection,
            floor_center_mm=floor_center_mm,
            floor_size_mm=floor_size_mm,
            mm_per_pixel=mm_per_pixel,
            enable_face_detection=enable_face_detection,
            face_detect_every_n_frames=face_detect_every_n_frames,
        )

    # ------------------------------------------------------------------
    # Annotation options
    # ------------------------------------------------------------------

    def set_vision_annotation_options(
        self,
        show_confidence: bool = True,
        show_cube_id: bool = True,
        box_thickness: int = 2,
    ) -> None:
        """Configure the annotation engine display options."""
        if self._brain.annotation_engine is None:
            return
        self._brain.annotation_engine.show_confidence = show_confidence
        self._brain.annotation_engine.show_cube_id = show_cube_id
        self._brain.annotation_engine.box_thickness = box_thickness

    # ------------------------------------------------------------------
    # Robot center calibration
    # ------------------------------------------------------------------

    def calibrate_robot_center(
        self,
        timeout_s: float = 30.0,
        resolution=None,
    ) -> float | None:
        """Calibrate the robot center by tilting head down and detecting circles.

        Returns the lateral offset in mm, or None if detection failed.
        """
        logger.info("Calibrating robot center (head to max-down)...")

        # Temporarily switch to manual exposure for calibration
        self._brain.camera.apply_settings(
            {"gain": 8.0, "exposure_ms": 300, "auto_exposure_enabled": False}
        )
        time.sleep(2.0)

        self._cli.set_head_angle(
            pycozmo.robot.MIN_HEAD_ANGLE.radians, accel=30.0, max_speed=30.0
        )
        time.sleep(3.0)

        # Collect multiple detections and average them
        dp = self._brain.detection_pipeline
        dp._robot_center_measured = False
        offsets = []
        for _ in range(int(timeout_s * 10)):
            time.sleep(0.1)
            rc = dp.robot_center_result
            if rc is not None and rc.detected:
                offsets.append(rc.lateral_offset_mm)
                if len(offsets) >= 10:
                    break

        # Return head to normal and restore the standard auto-exposure config
        self._cli.set_head_angle(0.0, accel=30.0, max_speed=30.0)
        time.sleep(1.0)
        self._brain.camera.enable(resolution)

        if offsets:
            avg_offset = sum(offsets) / len(offsets)
            dp.camera_lateral_offset_mm = avg_offset
            dp._robot_center_measured = True
            logger.info(
                "Robot center detected: lateral_offset=%+.2f mm (avg of %d frames)",
                avg_offset,
                len(offsets),
            )
            return avg_offset

        logger.warning("Robot center NOT detected (circles not found)")
        return None

    # ------------------------------------------------------------------
    # Remote control
    # ------------------------------------------------------------------

    def enable_remote_control(self) -> None:
        """Wire browser keyboard input to Cozmo's motors."""
        self._brain.install_remote_control(self._cli)

    # ------------------------------------------------------------------
    # Cube discovery and lights
    # ------------------------------------------------------------------

    def discover_cubes(self) -> None:
        """Discover available cubes and build factory_id mapping."""
        self.cubes.discover()

    def start_cube_lights(self) -> None:
        """Start the cube BLE keepalive cycler."""
        self.cubes.start_lights()

    def stop_cube_lights(self) -> None:
        """Stop the cube BLE keepalive cycler."""
        self.cubes.stop_lights()

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def set_origin(self) -> None:
        """Set the localization origin to the current robot position."""
        self._cli.conn.send(protocol_encoder.SetOrigin())
        logger.info("Origin set to current position")

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run the headless main loop. Blocks until disconnected or interrupted."""
        self._brain.run(self._cli)
        self._brain.cubes.stop_cube_light_worker()
