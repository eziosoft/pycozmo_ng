"""Brain — top-level orchestrator for the Cozmo robot.

Holds shared state, the run loop, and lifecycle management. Domain classes
(VisionDisplay, Navigator, CubeManager, WebBridge) receive a reference to
a Brain instance and access shared state via ``self.brain``.

Usage::

    brain = Brain()
    brain.run(cli)

"""

import os
import threading
import time

import cv2

import pycozmo
from pycozmo.cubes.cube_imu import CubeMoveDetector
from pycozmo.cubes.cycler import CubeCycler
from pycozmo.cubes.manager import CubeManager
from pycozmo.logger import logger_cubes
from pycozmo.navigation.navigator import Navigator
from pycozmo.vision.display import VisionDisplay
from pycozmo.web.viewer import WebBridge
from pycozmo.web.viewer_server import WebViewerServer


class Brain:
    """Cozmo robot brain — orchestrates vision, navigation, cubes, and web viewer.

    Domain classes receive a reference to this instance via ``self.brain``
    and read/write shared state through it.
    """

    def __init__(self):
        # Robot connection
        self.cli = None

        # Web viewer
        self.web_viewer = WebViewerServer()
        self.web_viewer.start()

        # Vision processor
        self.vision_processor = None

        # Display state
        self.running = True
        self.show_fps = True
        self.detection_enabled = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.headless = not os.environ.get("DISPLAY")
        if self.headless:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        self.window_name = "PyCozmo Vision Test - Normal (Left) | Annotated (Right)"

        # Camera lifecycle (COZMO-22)
        self._camera_active = False
        self._camera_enabled_at: float | None = None
        self._camera_resolution = None

        # Detections lock (COZMO-3)
        self._detections_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections: list = []
        self.latest_debug_images: dict = {}
        self.latest_floor_image = None
        self.latest_occupancy_grid = None

        # Cube tracking
        self.cube_factory_id_map: dict[int, int] = {}
        self._cube_object_id_map: dict[int, int] = {}
        self._cube_last_seen_frame: dict[int, int] = {}
        self._cube_cycler: CubeCycler | None = None
        self._cube_move_detectors: dict[int, CubeMoveDetector] = {}
        self._cube_suppress: dict = {
            "ids": set(),
            "fresh_frames": {},
            "fresh_frames_needed": 5,
        }

        # Dock / charging state (COZMO-28)
        self._dock_lock = threading.Lock()
        self._on_charger = False
        self._charging = False
        self._dock_pose = None
        self._origin_reset_origin_id = None

        # Remote control (COZMO-25)
        self._remote_control = None

        # Staleness detection (COZMO-22)
        self._last_frame_time = time.time()

        # Domain sub-components
        self.vision = VisionDisplay(self)
        self.navigator = Navigator(self)
        self.cubes = CubeManager(self)
        self.web = WebBridge(self)

    # ------------------------------------------------------------------
    # Camera lifecycle (COZMO-22)
    # ------------------------------------------------------------------

    def install_camera_lifecycle(self, cli, resolution):
        """Bind the Cozmo camera on/off to WebSocket viewer activity."""
        self._camera_resolution = resolution

        def _on_clients_change(active_count: int) -> None:
            if active_count > 0 and not self._camera_active:
                print(f"{active_count} viewer(s) connected - enabling Cozmo camera")
                cli.enable_camera(
                    enable=True,
                    color=False,
                    resolution=self._camera_resolution,
                )
                cli.set_camera_params(
                    gain=0.0,
                    exposure_ms=100,
                    auto_exposure_enabled=True,
                )
                self._camera_active = True
                self._camera_enabled_at = time.time()
            elif active_count == 0 and self._camera_active:
                print("No viewers - disabling Cozmo camera")
                cli.enable_camera(enable=False)
                self._camera_active = False
                self._camera_enabled_at = None

        self.web_viewer.add_active_clients_listener(_on_clients_change)

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
        if self.cli is not None:
            self._dock_pose = self.cli.pose
        print("Browser keyboard control enabled: WASD drive, R/F head, T/G lift")
        print("Click-to-navigate enabled: click ground to drive there")
        print("Click cube to approach and align")
        print("Z to dock: drive back to start position")

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
        """Main display loop."""
        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        _CONNECTED = pycozmo.conn.Connection.CONNECTED

        while self.running:
            if cli.conn.state != _CONNECTED:
                logger_cubes.warning(
                    "Connection state is %s - exiting main loop",
                    cli.conn.state,
                )
                self.running = False
                break

            if (
                self._camera_active
                and self._camera_enabled_at is not None
                and time.time() - self._camera_enabled_at > 5.0
            ):
                stale_for = time.time() - self._last_frame_time
                if stale_for > 5.0:
                    logger_cubes.warning(
                        "No camera frames for %.1fs - assuming Wi-Fi drop, "
                        "exiting main loop",
                        stale_for,
                    )
                    self.running = False
                    break

            self.vision.display_frames()

            if self.headless:
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                pkt = pycozmo.protocol_encoder.StopAllMotors()
                cli.conn.send(pkt)
                self.running = False
                break
            elif key == ord("s") or key == ord("S"):
                self.vision.save_frames()
            elif key == ord("c") or key == ord("C"):
                self.detection_enabled = not self.detection_enabled
                if self.vision_processor is not None:
                    if self.detection_enabled:
                        self.vision_processor.resume()
                    else:
                        self.vision_processor.pause()
                print(
                    f"Detection {'ENABLED' if self.detection_enabled else 'DISABLED'}"
                )
            elif key == ord("f") or key == ord("F"):
                self.show_fps = not self.show_fps
                print(f"FPS overlay {'ON' if self.show_fps else 'OFF'}")
            elif key == ord("i") or key == ord("I"):
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=50, rwheel_speed_mmps=50
                )
                cli.conn.send(pkt)
            elif key == ord("k") or key == ord("K"):
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=-50, rwheel_speed_mmps=-50
                )
                cli.conn.send(pkt)
            elif key == ord("j") or key == ord("J"):
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=-50, rwheel_speed_mmps=50
                )
                cli.conn.send(pkt)
            elif key == ord("l") or key == ord("L"):
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=50, rwheel_speed_mmps=-50
                )
                cli.conn.send(pkt)
            elif key == ord(" "):
                pkt = pycozmo.protocol_encoder.StopAllMotors()
                cli.conn.send(pkt)
            elif key == ord("p") or key == ord("P"):
                self.web_viewer.clear_path()
                print("Path trail cleared")

        if not self.headless:
            cv2.destroyAllWindows()
        self.stop_remote_control()
        self.web_viewer.close()
