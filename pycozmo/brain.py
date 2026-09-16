"""Brain — top-level orchestrator for the Cozmo robot.

Creates typed state containers and wires them to domain classes.
Domain classes receive the specific state container they need, not
a reference to Brain itself.

Usage::

    brain = Brain()
    brain.run(cli)

"""

import time

import pycozmo
from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.cubes.manager import CubeManager
from pycozmo.logger import logger_cubes
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.navigator import Navigator
from pycozmo.protocol import protocol_encoder
from pycozmo.vision.display import VisionDisplay
from pycozmo.vision.frame_state import FrameState
from pycozmo.web.viewer import WebBridge
from pycozmo.web.viewer_server import WebViewerServer


class Brain:
    """Cozmo robot brain — orchestrates vision, navigation, cubes, and web viewer.

    Creates typed state containers and passes them to domain classes.
    Brain itself is a thin lifecycle runner, not a shared-memory bus.
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
        self.web_viewer.start()

        # Vision processor (backing field for property)
        self._vision_processor = None

        # Remote control (COZMO-25)
        self._remote_control = None

        # Domain sub-components — each receives the state it needs
        self.vision = VisionDisplay(self.frame_state, self.cube_tracking)
        self.navigator = Navigator(self.dock_state)
        self.cubes = CubeManager(self.cube_tracking)
        self.web = WebBridge(self.frame_state, self.cube_tracking, self.dock_state)

        # Wire cross-domain references
        self.vision.web_viewer = self.web_viewer
        self.vision.web = self.web
        self.navigator.cli = self.cli
        self.navigator.web_viewer = self.web_viewer
        self.web.web_viewer = self.web_viewer
        self.web.vision_processor = self.vision_processor

    @property
    def cli(self):
        return self._cli

    @cli.setter
    def cli(self, value):
        self._cli = value
        self.navigator.cli = value
        self.web.cli = value

    @property
    def vision_processor(self):
        return self._vision_processor

    @vision_processor.setter
    def vision_processor(self, value):
        self._vision_processor = value
        self.vision.vision_processor = value
        self.web.vision_processor = value
        # Wire floor pipeline to VisionDisplay if available
        if hasattr(value, "floor_pipeline"):
            self.vision.floor_pipeline = value.floor_pipeline

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
            exposure_ms=100,
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
            exposure_ms=100,
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
    _RESOLUTION_MAP = {
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
        cli.set_head_light(settings.get("head_light", True))

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
        self.web_viewer.close()
