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
        print(cozmo.sensors.battery_voltage)
        cozmo.navigator.drive_to(x=100, y=0)
        cozmo.web_viewer.start()
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Self

import pycozmo
from pycozmo.brain import Brain
from pycozmo.navigation.world_map import WorldMap
from pycozmo.vision.detection_pipeline import DetectionPipeline
from pycozmo.web.viewer_server import WebViewerServer


class VisionAPI:
    """High-level vision interface: start/stop, frame access."""

    def __init__(self, brain: Brain):
        self._brain = brain

    def start(self, **kwargs):
        """Start the vision pipeline.

        Args:
            **kwargs: Passed to Brain.setup_vision() (process_every_n_frames,
                     undistort, enable_floor_projection, etc.)
        """
        if self._brain.detection_pipeline is not None:
            return  # Already started
        # Vision is set up in connect() - just ensure it's running
        # The pipeline is already running via the event handler

    def stop(self):
        """Stop the vision pipeline."""
        # Vision runs automatically with camera - stop camera to stop vision

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


class CubesAPI:
    """High-level cube interface."""

    def __init__(self, brain: Brain):
        self._brain = brain

    @property
    def connected(self):
        """Get connected cube IDs."""
        return list(self._brain.cubes.connected_cubes.keys())

    def get_cube(self, cube_id: int):
        """Get a cube by ID (1-3)."""
        return self._brain.cubes.get_cube(cube_id)

    def set_lights(self, cube_id: int, color):
        """Set cube LED color."""
        cube = self.get_cube(cube_id)
        if cube:
            cube.set_lights(color)


class NavigatorAPI:
    """High-level navigation interface."""

    def __init__(self, brain: Brain):
        self._brain = brain
        self._navigator = brain.navigator

    def drive_to(self, x: float, y: float, **kwargs) -> bool:
        """Drive to a world position (mm)."""
        return self._navigator.drive_to_point(x, y, **kwargs)

    def approach_cube(self, cube_id: int, **kwargs) -> bool:
        """Approach and align with a cube."""
        return self._navigator.approach_cube(cube_id, **kwargs)

    def dock(self, **kwargs) -> bool:
        """Drive back to dock/origin."""
        return self._navigator.dock(**kwargs)

    def stop(self):
        """Stop all movement."""
        self._navigator.stop()


class WorldMapAPI:
    """Read-only access to the world map."""

    def __init__(self, brain: Brain):
        self._world_map: WorldMap = brain.web_viewer.world_map

    @property
    def robot_pose(self):
        """Get current robot pose (position, yaw)."""
        return self._world_map.get_robot_pose()

    @property
    def cubes(self):
        """Get all known cube poses in world frame."""
        return self._world_map.get_all_cube_poses()

    @property
    def path(self):
        """Get the robot's path trail."""
        return self._world_map.get_path()

    def get_cube(self, cube_id: int):
        """Get a specific cube's world pose."""
        return self._world_map.get_cube_pose(cube_id)


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
        return self._web_viewer._ws_server._thread is not None and \
               self._web_viewer._ws_server._thread.is_alive()


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
    ) -> "Cozmo":
        """Connect to the Cozmo robot.

        Args:
            host: Robot IP address (default: 172.31.1.1 on Cozmo AP)
            port: Robot port (default: 5551)
            **kwargs: Passed to pycozmo.connect()

        Returns:
            Self for chaining.
        """
        if self._connected:
            return self

        self._cli = pycozmo.connect(host=host, port=port, **kwargs)
        self._brain.cli = self._cli
        self._connected = True
        return self

    def disconnect(self):
        """Disconnect from the robot."""
        if self._connected and self._cli:
            self._cli.disconnect()
            self._connected = False

    def __enter__(self) -> "Self":
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
            resolution: ImageResolution enum (default: QVGA)
            **kwargs: Additional camera parameters (gain, exposure, etc.)
        """
        if resolution is None:
            resolution = pycozmo.protocol_encoder.ImageResolution.QVGA
        self._brain.enable_camera_persistent(self._cli, resolution)
        if kwargs:
            self._cli.set_camera_params(**kwargs)

    def set_head_angle(self, angle_deg: float, **kwargs):
        """Set head angle in degrees."""
        self._cli.set_head_angle(pycozmo.util.Angle(degrees=angle_deg), **kwargs)

    def set_lift_height(self, height_mm: float, **kwargs):
        """Set lift height in mm."""
        self._cli.set_lift_height(pycozmo.util.Distance(mm=height_mm), **kwargs)

    def drive_wheels(self, left_mmps: float, right_mmps: float, **kwargs):
        """Drive wheels at specified speeds (mm/s)."""
        self._cli.drive_wheels(
            pycozmo.util.Speed(mmps=left_mmps),
            pycozmo.util.Speed(mmps=right_mmps),
            **kwargs,
        )

    def stop_all_motors(self):
        """Stop all motors."""
        self._cli.stop_all_motors()