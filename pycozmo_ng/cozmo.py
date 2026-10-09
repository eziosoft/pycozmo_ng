"""
Cozmo — public API for controlling the Cozmo robot.

This is the main entry point for library users. It provides a clean,
high-level interface to all robot capabilities while hiding internal
implementation details.

Usage:
    from pycozmo_ng import Cozmo

    cozmo = Cozmo()
    with cozmo.connect():
        cozmo.vision.start()
        cozmo.navigator.drive_to(x=100, y=0)
        cozmo.run()
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

import pycozmo_ng
from pycozmo_ng.brain import Brain
from pycozmo_ng.cubes.carry_state import CarryState
from pycozmo_ng.cubes.identity import to_cube_id, to_world_index
from pycozmo_ng.logger import logger, setup_basic_logging
from pycozmo_ng.navigation.world_map import CubeState, WorldMap
from pycozmo_ng.protocol import protocol_encoder
from pycozmo_ng.vision.cube_detector.detection_pipeline import DetectionPipeline
from pycozmo_ng.vision.vision_setup import VisionSetup
from pycozmo_ng.web.viewer_server import WebViewerServer


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
            **kwargs: :class:`VisionSetup` fields — see
                :meth:`Cozmo.setup_vision`.

        Raises:
            RuntimeError: If not connected to the robot.
        """
        brain = self._brain
        if brain.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() first")
        brain.setup_vision(VisionSetup(**kwargs))
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

    @property
    def camera_settings(self) -> dict:
        """Camera settings currently in force (gain, exposure_ms, ...)."""
        return self._brain.camera.settings

    def apply_settings(self, settings: dict) -> dict:
        """Apply a partial camera-settings update and return the applied state.

        Omitted keys keep their current values; unknown keys are
        dropped. Equivalent to the browser's ``camera_settings`` action.

        Args:
            settings: Any subset of the camera settings vocabulary.

        Returns:
            The settings in force after the update.

        Raises:
            RuntimeError: If not connected to the robot.
        """
        if self._brain.cli is None:
            raise RuntimeError("Not connected - call Cozmo.connect() first")
        return self._brain.camera.apply_settings(settings)


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


class CarryAPI:
    """Whether the robot is carrying a cube on its lift — checked on demand.

    The answer comes from the post-grab visibility heuristic (see
    ``pycozmo_ng.cubes.carry_state``): after a grab nothing is judged until
    the fork has finished rising (lift in position), then with the forks
    up, a grabbed cube that is still in camera view is on the floor (grab
    missed), while one that left the frame is riding the forks. Seeing
    the cube resolves the answer to not carrying until the next grab
    (driving away afterwards cannot flip it back). Forks settled down
    reads as not carrying.

    Example:
        cozmo.navigator.approach_cube(1)
        ...
        if cozmo.carry.is_carrying:
            print(f"holding cube {cozmo.carry.cube_id}")
    """

    def __init__(self, brain: Brain):
        self._brain = brain

    @property
    def state(self) -> CarryState:
        """``CarryState.CARRYING`` / ``NOT_CARRYING`` / ``UNKNOWN``."""
        return self._brain.carry.state

    @property
    def is_carrying(self) -> bool | None:
        """``True`` / ``False``, or ``None`` when it cannot be determined."""
        return self._brain.carry.is_carrying

    @property
    def cube_id(self) -> int | None:
        """Cube ID (1-3) whose grab is being observed, or ``None``."""
        return self._brain.carry.cube_id

    @property
    def on_forks(self) -> bool:
        """Is a cube believed to be on the raised forks right now?

        ``True`` while carrying (or right after a grab, before the fork
        finished rising) — the predicate for stacking a cube on another.
        """
        return self._brain.carry.on_forks

    def clear(self) -> None:
        """Forget the observation (cube placed / forks lowered)."""
        self._brain.carry.clear()


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

        Returns True if a cube visit was actually started, False when
        refused: the cube is not in the world map, there is no robot
        connection, or another cube visit (approach/stack) is in flight.
        While a cube is on the raised forks, approaching a *different*
        cube stacks the held cube on it.
        """
        return self._navigator.approach_cube(cube_id)

    def stack_cube(self, cube_id: int) -> bool:
        """Carry the held cube to the given cube ID (1-3) and stack it.

        Requires a cube on the raised forks (see ``cozmo.carry``).
        Returns True if the stack was started, False when refused (no
        cube on the forks, the target is the held cube or unknown, or
        another cube visit is in flight).
        """
        return self._navigator.stack_cube(cube_id)

    def dock(self) -> None:
        """Drive back to dock/origin (in the background)."""
        self._navigator.dock()

    def dock_on_aruco(self, marker_id: int = 0) -> bool:
        """Drive to an ArUco marker and reverse into it until charging.

        The robot drives to 15 cm in front of the marker (id=0 by default
        — the charger marker) facing it, then reverses at low speed until
        the charger contacts are detected or a timeout fires.  Returns
        True if the dock attempt was started (marker known), False
        otherwise.
        """
        return self._navigator.dock_on_aruco(marker_id)

    def stop(self) -> None:
        """Stop all movement."""
        self._navigator.stop()

    def navigate_to(self, x: float, y: float) -> None:
        """Navigate to a world position (mm) in the background.

        Uses the A* path when enabled (see :meth:`set_astar_enabled`),
        otherwise the firmware path follower. Supersedes any navigation
        already in flight; observe it through :attr:`active_motions`.
        """
        self._navigator.navigate_to(x, y)

    def cancel_navigation(self) -> None:
        """Cancel background navigation and stop the motors.

        Leaves other motion (approach, dock) untouched.
        """
        self._navigator.navigate_to(None)

    @property
    def active_motions(self) -> tuple[str, ...]:
        """Names of motion jobs running right now (``navigate``, ``approach``, ...)."""
        return self._brain.motion.active_names

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
    ``pycozmo_ng.cubes.identity``.
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
        return {to_cube_id(cube_index): state for cube_index, state in self._world_map.get_cubes().items()}

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

    def get_world_state(self) -> dict:
        """Full world snapshot as plain JSON-able data (world frame, mm/rad).

        Keys: ``robot`` (position, yaw, origin_id), ``cubes`` (world-map
        index 0-2 -> pose), ``faces``, ``charger``, ``aruco_markers``,
        ``path_points``, ``astar_path``. Cube-facing members of this
        class use IDs 1-3; this raw snapshot keeps the map's own keys —
        use :meth:`get_cube` for the ID-based view.
        """
        return self._world_map.get_world_state()


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

    @property
    def is_charging(self) -> bool:
        """Whether the robot reports charging right now."""
        with self._brain.dock_state.lock:
            return self._brain.dock_state.charging

    @property
    def on_charger(self) -> bool:
        """Whether the robot believes it is on the charger."""
        with self._brain.dock_state.lock:
            return self._brain.dock_state.on_charger

    @property
    def picked_up(self) -> bool:
        """Whether the robot is currently picked up."""
        cli = self._brain.cli
        return bool(cli.robot_picked_up) if cli else False

    @property
    def moving(self) -> bool:
        """Whether the wheels are reported as moving."""
        cli = self._brain.cli
        return bool(cli.robot_moving) if cli else False


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


class McpAPI:
    """MCP server interface — let LLM clients drive Cozmo and read its state.

    The server speaks MCP streamable HTTP in-process (like the web
    viewer), default ``127.0.0.1:8767``. Requires the optional extra::

        pip install 'pycozmo-ng[mcp]'

    Example:
        cozmo.mcp.start()
        # ... point an MCP client at http://127.0.0.1:8767/mcp ...
        cozmo.mcp.stop()
    """

    def __init__(self, cozmo: Cozmo):
        self._cozmo = cozmo
        self._server: Any = None

    def start(self, host: str = "127.0.0.1", port: int = 8767) -> None:
        """Start the MCP server (no-op when already running).

        Args:
            host: Bind address (default localhost — keep it there unless
                you trust the LAN, the tool surface can move the robot).
            port: TCP port (default 8767).

        Raises:
            RuntimeError: If the ``mcp`` package is not installed.
        """
        if self._server is not None and self._server.is_running:
            return
        from pycozmo_ng.mcp.server import McPServer

        self._server = McPServer(self._cozmo, host=host, port=port)
        self._server.start()

    def stop(self) -> None:
        """Stop the MCP server (no-op when not running)."""
        if self._server is not None:
            self._server.stop()
            self._server = None

    @property
    def is_running(self) -> bool:
        """Whether the MCP server is serving."""
        return self._server is not None and self._server.is_running

    @property
    def url(self) -> str | None:
        """Endpoint URL for an MCP client, or None when not running."""
        if self._server is None or not self._server.is_running:
            return None
        return self._server.url


class Cozmo:
    """Main entry point for the PyCozmo-NG library.

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
        self._carry: CarryAPI | None = None
        self._navigator: NavigatorAPI | None = None
        self._world_map: WorldMapAPI | None = None
        self._sensors: SensorsAPI | None = None
        self._web_viewer: WebViewerAPI | None = None
        self._mcp: McpAPI | None = None

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
    def carry(self) -> CarryAPI:
        if self._carry is None:
            self._carry = CarryAPI(self._brain)
        return self._carry

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

    @property
    def mcp(self) -> McpAPI:
        if self._mcp is None:
            self._mcp = McpAPI(self)
        return self._mcp

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
        self._cli = pycozmo_ng.Client(
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
        """Access the underlying pycozmo_ng Client for advanced use."""
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
        self._cli.set_head_angle(pycozmo_ng.util.Angle(degrees=angle_deg).radians, **kwargs)

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

    def setup_vision(self, setup: VisionSetup | None = None, **overrides) -> None:
        """Set up the vision pipeline and register all internal event handlers.

        The options are declared exactly once — as the fields of
        :class:`~pycozmo_ng.vision.vision_setup.VisionSetup` — and this entry
        point accepts all of them, either as a pre-built instance or as
        individual keyword overrides. There is no second, shorter parameter
        list to keep in sync: anything the internal setup understands is
        accepted here too, and an unknown option raises ``TypeError``.

        Requires a connection: when the calibration is omitted it is
        retrieved from the robot.

        Args:
            setup: Options to apply (defaults to ``VisionSetup()`` defaults).
            **overrides: Individual ``VisionSetup`` fields to change, e.g.
                ``setup_vision(process_every_n_frames=3,
                enable_occupancy_map=True)``.

        Raises:
            RuntimeError: If not connected to the robot.
        """
        self._brain.setup_vision(replace(setup or VisionSetup(), **overrides))

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

        Arms a robot-center run, takes one reading per frame that sees the
        ring markers, and freezes their average as the session offset.

        Returns the lateral offset in mm, or None if detection failed.
        """
        dp = self._brain.detection_pipeline
        if dp is None or dp.robot_center_calibration is None:
            raise RuntimeError("Not set up - call Cozmo.setup_vision() before calibrate_robot_center")

        logger.info("Calibrating robot center (head to max-down)...")

        # Frames now feed fresh readings; the previous session value stays in
        # force until this run publishes its average.
        calib = dp.robot_center_calibration
        calib.arm()

        try:
            # Temporarily switch to manual exposure for calibration
            self._brain.camera.apply_settings({"gain": 8.0, "exposure_ms": 300, "auto_exposure_enabled": False})
            time.sleep(2.0)

            self._cli.set_head_angle(pycozmo_ng.robot.MIN_HEAD_ANGLE.radians, accel=30.0, max_speed=30.0)
            time.sleep(3.0)

            deadline = time.monotonic() + timeout_s
            while len(calib.readings) < 10 and time.monotonic() < deadline:
                time.sleep(0.1)
        finally:
            # Return head to normal and restore the standard auto-exposure config
            self._cli.set_head_angle(0.0, accel=30.0, max_speed=30.0)
            time.sleep(1.0)
            self._brain.camera.enable(resolution)
            calib.disarm()

        readings = calib.readings
        if readings:
            avg_offset = sum(readings) / len(readings)
            calib.freeze(avg_offset)
            logger.info(
                "Robot center detected: lateral_offset=%+.2f mm (avg of %d frames)",
                avg_offset,
                len(readings),
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
