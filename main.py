#!/usr/bin/env python
"""

Main test script for PyCozmo vision system.

Displays both normal and annotated camera frames side-by-side using OpenCV.
Streams cube poses to browser via WebSocket (Three.js viewer).

Controls:
- Press 'q' to quit
- Press 's' to save current frames
- Press 'c' to toggle cube detection
- Press 'f' to toggle FPS display

The web viewer shows the raw Cozmo camera feed — no
brightness/gamma post-processing. Camera settings (exposure,
gain) are sent to Cozmo directly; auto-exposure is on by default.

Detection throttle (only every Nth frame goes through cube detection):
  python main.py --process-every-n-frames 2   # 15 fps detection, 30 fps preview
  python main.py --process-every-n-frames 3   # 10 fps detection

"""

import argparse
import logging
import os
import subprocess
import threading
import time

from typing import Optional

import cv2
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pycozmo
from pycozmo.logger import logger_cubes
from pycozmo import vision
from pycozmo.camera_calibration import CameraCalibrationRetriever
from pycozmo.cube_imu import CubeMoveDetector
from pycozmo.web_viewer_server import WebViewerServer


def connect_to_cozmo_wifi():
    """Auto-connect to Cozmo WiFi AP if not already connected."""
    ssid = os.environ.get("COZMO_WIFI_SSID")
    password = os.environ.get("COZMO_WIFI_PASSWORD")
    if not ssid:
        return True  # No credentials configured, skip

    # Check if already connected to this SSID via nmcli
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,DEVICE,STATE", "con", "show"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == ssid and parts[2] == "activated":
                print(f"✅ Already connected to {ssid}")
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Connect via nmcli
    print(f"📡 Connecting to {ssid}...")
    try:
        result = subprocess.run(
            ["sudo", "nmcli", "dev", "wifi", "connect", ssid, "password", password],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            print(f"✅ Connected to {ssid}")
            time.sleep(2)  # Wait for DHCP
            return True
        else:
            print(f"⚠️  WiFi connect failed: {result.stderr.strip()}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"⚠️  WiFi connect error: {e}")
        return False


class CubeCycler:
    """Rapidly cycles through cubes, connecting to one at a time.

    The Cozmo robot firmware only supports 1 BLE cube connection at a time.
    This class cycles through all known cubes, connecting to each for a brief
    period to collect accel data, then disconnecting and moving to the next.

    Timing per cube:
      - Connect: ~10-50ms
      - Accel stream enable: ~30ms
      - Data collection: configurable (default 100ms)
      - Disconnect: ~10ms
    Total: ~150-200ms per cube, ~450-600ms for 3 cubes.
    """

    def __init__(self, cycle_ms: int = 150):
        self._cycle_ms = cycle_ms
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._current_cube_id: Optional[int] = None
        self._current_object_id: Optional[int] = None
        self._cube_map: dict[int, int] = {}  # cube_id -> factory_id
        self._object_map: dict[int, int] = {}  # cube_id -> object_id
        self._lock = threading.Lock()
        self._stats = {"cycles": 0, "connects": 0, "disconnects": 0, "errors": 0}

    def start(self, cli, cube_map: dict[int, int]):
        """Start cycling. cube_map = {cube_id: factory_id}."""
        with self._lock:
            self._cube_map = dict(cube_map)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(cli,), name="CubeCycler", daemon=True
        )
        self._thread.start()

    def stop(self):
        """Stop cycling."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_current(self) -> tuple[Optional[int], Optional[int]]:
        """Return (cube_id, object_id) of the currently connected cube."""
        with self._lock:
            return self._current_cube_id, self._current_object_id

    @property
    def stats(self):
        return dict(self._stats)

    def _loop(self, cli):
        """Main cycling loop."""
        import pycozmo.protocol_encoder as pe

        while not self._stop.is_set():
            with self._lock:
                cubes = list(self._cube_map.items())
            if not cubes:
                self._stop.wait(0.5)
                continue

            for cube_id, factory_id in cubes:
                if self._stop.is_set():
                    break
                self._cycle_one(cli, cube_id, factory_id, pe)

    def _cycle_one(self, cli, cube_id, factory_id, pe):
        """Connect to one cube, collect data briefly, disconnect."""
        oid = None
        try:
            # Connect
            cli.conn.send(pe.ObjectConnect(factory_id=factory_id, connect=True))
            confirmations = cli.wait_for_factory_ids(
                [factory_id], connect=True, timeout=1.0
            )
            if factory_id not in confirmations:
                self._stats["errors"] += 1
                return
            oid = confirmations[factory_id]
            with self._lock:
                self._current_cube_id = cube_id
                self._current_object_id = oid
                self._object_map[cube_id] = oid
            self._stats["connects"] += 1

            # Enable accel stream + set CubeId
            cli.conn.send(pe.CubeId(object_id=oid))
            # cli.conn.send(pe.StreamObjectAccel(object_id=oid, enable=True))  # Disabled (COZMO-28)

            # Collect data for cycle_ms
            self._stop.wait(self._cycle_ms / 1000.0)

        except Exception:
            self._stats["errors"] += 1
        finally:
            # Disconnect
            try:
                cli.conn.send(pe.ObjectConnect(factory_id=factory_id, connect=False))
                self._stop.wait(0.05)  # brief pause for disconnect to propagate
            except Exception:
                pass
            with self._lock:
                if self._current_cube_id == cube_id:
                    self._current_cube_id = None
                    self._current_object_id = None
            self._stats["disconnects"] += 1
            self._stats["cycles"] += 1


class VisionDisplay:
    """Handles display of normal and annotated frames + web viewer streaming."""

    def __init__(self):
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections = []
        self.latest_debug_images = {}
        self.running = True
        self.show_fps = True
        self.detection_enabled = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.vision_processor = None
        self.cli = None  # set after pycozmo.connect() succeeds (used for robot pose)
        self._dock_pose = (
            None  # robot pose at start — assumed to be the dock (COZMO-28)
        )
        # Charging state (COZMO-28): set by EvtRobotOnChargerChange /
        # EvtRobotChargingChange handlers running on the event-dispatch
        # thread; read by the dock thread. Guarded by _dock_lock.
        self._dock_lock = threading.Lock()
        self._on_charger = False
        self._charging = False
        # Origin id of the last pose-origin reset, so charging-triggered
        # resets are idempotent. (COZMO-28)
        self._origin_reset_origin_id = None

        # Camera lifecycle (COZMO-22): the Cozmo camera is off by default
        # so we don't flood its lossy Wi-Fi AP. A listener on
        # ``self.web_viewer`` turns the camera on when the first browser
        # tab connects and off again when the last one closes.
        self._camera_active = False

        # `latest_detections` is mutated by the event-dispatcher thread
        # (on_annotated_camera_image) and read by the main loop
        # (update_cube_lights_on_detection, display_frames). Guard all
        # access with this lock. (COZMO-3)
        self._detections_lock = threading.Lock()

        # Detect headless (no display available)
        self.headless = not os.environ.get("DISPLAY")
        if self.headless:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        # Web viewer
        self.web_viewer = WebViewerServer()
        self.web_viewer.start()

        # Window names
        self.window_name = "PyCozmo Vision Test - Normal (Left) | Annotated (Right)"

        # Cube tracking variables
        self.cube_factory_id_map = {}
        self._cube_object_id_map = {}  # cube_id → object_id (assigned by robot)
        # Per-cube "last seen" frame index, for diagnostics only. The cube-
        # light worker no longer uses it to drive disconnects — Cozmo drops
        # out-of-range cubes on its own, and old proactive disconnect
        # logic caused spurious drops once multi-cube connects were enabled.
        # (COZMO-20)
        self._cube_last_seen_frame = {}
        # Cube-movement detectors (cube_id 1..3 -> CubeMoveDetector). Fed by
        # Rapid cube cycler — cycles through cubes, connecting to one at a
        # time. Replaces the old keepalive worker which tried to maintain
        # simultaneous connections (impossible due to firmware limit).
        # (COZMO-23)
        self._cube_cycler: Optional[CubeCycler] = None
        self._cube_move_detectors: dict[int, CubeMoveDetector] = {}
        # Cube-measurement suppression while a cube is being relocated (COZMO-34).
        # Mapping: ids=set of cube_id being suppressed; fresh_frames=consecutive
        # re-detection frame count per cube; fresh_frames_needed=threshold.
        self._cube_suppress: dict = {
            "ids": set(),
            "fresh_frames": {},
            "fresh_frames_needed": 5,
        }
        # Wall-clock timestamp of the most recently received camera frame.
        # Used by the main run loop to detect a silent Wi-Fi drop (see run).
        self._last_frame_time = time.time()

    def on_object_connection_state(self, cli, pkt):
        """Log cube connection state transitions for diagnostics.

        The CubeCycler manages connections; this handler just logs state
        transitions and tracks object_id mappings for other subsystems
        (vision, move detectors).
        (COZMO-20, COZMO-23)
        """
        cube_id = self._cube_id_for_factory(pkt.factory_id)
        if cube_id is None:
            logger_cubes.warning(
                "ObjectConnectionState for unknown factory_id 0x%08x "
                "(connected=%s, object_id=%s)",
                pkt.factory_id,
                pkt.connected,
                pkt.object_id,
            )
            return
        if pkt.connected:
            self._cube_object_id_map[cube_id] = pkt.object_id
            # Start the accel grace period + discard the old resting baseline:
            # freshly-connected cubes emit unreliable accel for a moment.
            self._cube_move_detectors.setdefault(
                cube_id, CubeMoveDetector()
            ).on_connection(time.time())
            logger_cubes.debug(
                "Cube %d connected: object_id=%s",
                cube_id,
                pkt.object_id,
            )
        else:
            # Clear the mapping — the cycler will reconnect if needed.
            self._cube_object_id_map.pop(cube_id, None)
            logger_cubes.debug(
                "Cube %d disconnected (object_id=%s)",
                cube_id,
                pkt.object_id,
            )

    def _cube_id_for_factory(self, factory_id: int) -> Optional[int]:
        """Reverse lookup: factory_id -> detected cube_id (1..3)."""
        for cube_id, fid in self.cube_factory_id_map.items():
            if fid == factory_id:
                return cube_id
        return None

    def _cube_id_for_object_id(self, cli, object_id: int) -> Optional[int]:
        """Reverse lookup: temporary object_id -> detected cube_id (1..3).

        ``cli.connected_objects`` is keyed by the temporary object_id Cozmo
        assigns per connection; each entry holds the permanent ``factory_id``.
        Several cubes can be connected at once, so this is the only reliable
        way to map an ObjectMoved/ObjectAccel packet back to a cube.
        """
        info = cli.connected_objects.get(object_id)
        if info is None:
            return None
        return self._cube_id_for_factory(info.get("factory_id"))

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.latest_raw_frame = image
        self.frame_count += 1
        # Stamp the time of the last received frame so the main loop
        # can detect a silent Wi-Fi drop (no Disconnect packet arrives
        # in that case, and cli.conn.state stays CONNECTED forever).
        self._last_frame_time = time.time()

        # Push frame for MJPEG streaming (encoding happens off this thread)
        if image is not None:
            self.web_viewer.push_frame(image)

        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.latest_annotated_frame = annotated_image
        # Detections arrive on the event-dispatcher thread; the main loop reads
        # them. Take the lock so readers see a consistent snapshot. (COZMO-3)
        with self._detections_lock:
            self.latest_detections = detections
        # Cube-light cycling now runs on its own worker thread; the
        # dispatcher just stamps the latest frame seen per cube for the
        # HUD. (COZMO-20)
        self._stamp_detections_seen()
        self.update_web_viewer()

    def _stamp_detections_seen(self):
        """Record the current frame index for every detected cube.

        Diagnostic only — the cube-light worker reads from
        ``latest_detections`` directly under the lock, so this is just a
        convenience for the HUD/debug prints. (COZMO-20)
        """
        with self._detections_lock:
            detected_cube_ids = {det.cube_id for det in self.latest_detections}
        for cube_id in detected_cube_ids:
            self._cube_last_seen_frame[cube_id] = self.frame_count

    def start_cube_cycler(self, cli):
        """Start the rapid cube cycler.

        Replaces the old keepalive worker. The cycler cycles through all
        known cubes, connecting to one at a time to collect accel data.
        (COZMO-23)
        """
        if not self.cube_factory_id_map:
            return
        self._cube_cycler = CubeCycler(cycle_ms=150)
        self._cube_cycler.start(cli, self.cube_factory_id_map)
        print(f"🔄 Cube cycler started — cycling {len(self.cube_factory_id_map)} cubes")

    def stop_cube_cycler(self):
        """Stop the rapid cube cycler."""
        if self._cube_cycler:
            self._cube_cycler.stop()
            stats = self._cube_cycler.stats
            print(
                f"🔄 Cube cycler stopped — {stats['cycles']} cycles, {stats['errors']} errors"
            )
            self._cube_cycler = None

    # Legacy aliases kept for compatibility
    def start_cube_light_worker(self, cli):
        self.start_cube_cycler(cli)

    def stop_cube_light_worker(self):
        self.stop_cube_cycler()

    def on_debug_images(self, cli, debug_images, detections):
        """Handler for debug images from cube detection pipeline."""
        self.latest_debug_images = debug_images

    def update_web_viewer(self):
        """Feed the latest cube measurements into the world map.

        The camera-frame cube measurements are transformed into world frame
        and fused by ``WorldMap`` (owned by ``self.web_viewer``), then
        broadcast to any connected browser. The robot's world pose is pushed
        separately by the RobotState handler (``_on_robot_state_for_trail``),
        which is independent of the camera stream. (World-map refactor)

        Cubes that have been invalidated (moved, ``forget_cube``) are
        suppressed here until they are re-detected stably — otherwise the
        per-frame vision tracker keeps re-feeding the cube's *old* camera pose
        the instant it is forgotten, and the visible position never actually
        resets. (COZMO-34)
        """
        if not self.vision_processor:
            return

        head_angle = self.cli.head_angle.radians if self.cli is not None else 0.0

        suppressed = self._cube_suppress.get("ids", set())
        fresh_frames = self._cube_suppress.get("fresh_frames", {})
        new_fresh_frames = {}

        for m in self.vision_processor.get_cube_measurements():
            cube_id = m["cube_id"]
            if cube_id in suppressed:
                # Count consecutive frames the cube has been (re-)detected.
                # Only once it has been tracked for FRESH_FRAMES_NEEDED
                # consecutive frames do we trust it has settled somewhere new.
                n = fresh_frames.get(cube_id, 0) + 1
                new_fresh_frames[cube_id] = n
                if n < self._cube_suppress["fresh_frames_needed"]:
                    continue
                # Settlement confirmed — stop suppressing.
                suppressed.discard(cube_id)
                logger_cubes.info(
                    "Cube %d re-settled — re-enabling measurements", cube_id + 1
                )

            self.web_viewer.update_cube_measurement(
                cube_id,
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
            )

        # A cube that was suppressed but is NO LONGER detected at all resets
        # its freshness counter (it may have been out of frame while moving —
        # the counter only counts continuous re-detection).
        self._cube_suppress["fresh_frames"] = new_fresh_frames

        self.web_viewer.broadcast()

    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format."""
        if pil_image is None:
            return None
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def add_text_overlay(self, frame, text_lines, position=(10, 30)):
        """Add text overlay to frame."""
        x, y = position
        for i, line in enumerate(text_lines):
            y_pos = y + i * 25
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x - 5, y_pos - text_height - 5),
                (x + text_width + 5, y_pos + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    def display_frames(self):
        """Display both raw and annotated frames side by side."""
        if self.headless:
            return

        raw_cv2 = self.pil_to_cv2(self.latest_raw_frame)
        annotated_cv2 = self.pil_to_cv2(self.latest_annotated_frame)

        if raw_cv2 is None:
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                waiting_frame,
                "Waiting for camera frames...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.window_name, waiting_frame)
            return

        if annotated_cv2 is None:
            annotated_cv2 = raw_cv2.copy()

        height, width = raw_cv2.shape[:2]
        if annotated_cv2.shape[:2] != (height, width):
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

        raw_labeled = raw_cv2.copy()
        annotated_labeled = annotated_cv2.copy()

        cv2.rectangle(raw_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            raw_labeled,
            "Normal Camera Feed",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(annotated_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            annotated_labeled,
            "Annotated (Detections)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if self.show_fps:
            info_lines = [
                f"FPS: {self.fps:.1f}",
                f"Cubes: {self._safe_detection_count()}",
                f"Detection: {'ON' if self.detection_enabled else 'OFF'}",
            ]
            self.add_text_overlay(raw_labeled, info_lines, (10, height - 90))

            with self._detections_lock:
                detections_snapshot = list(self.latest_detections)
            if detections_snapshot:
                det_lines = [f"Detected Cubes: {len(detections_snapshot)}"]
                for i, det in enumerate(detections_snapshot[:3]):
                    det_lines.append(f"  ID:{det.cube_id} Conf:{det.confidence:.2f}")
                self.add_text_overlay(annotated_labeled, det_lines, (10, height - 90))

        combined = np.hstack([raw_labeled, annotated_labeled])
        cv2.imshow(self.window_name, combined)

        for name, img in self.latest_debug_images.items():
            cv2.imshow(f"Debug: {name}", img)

    def save_frames(self):
        """Save current frames to disk."""
        if self.latest_raw_frame is None:
            print("⚠️  No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        raw_filename = f"frame_raw_{timestamp}.png"
        self.latest_raw_frame.save(raw_filename)
        print(f"💾 Saved raw frame: {raw_filename}")

        if self.latest_annotated_frame is not None:
            annotated_filename = f"frame_annotated_{timestamp}.png"
            self.latest_annotated_frame.save(annotated_filename)
            print(f"💾 Saved annotated frame: {annotated_filename}")

        with self._detections_lock:
            n_det = len(self.latest_detections)
        print(f"   Detections: {n_det}")

    def _safe_detection_count(self) -> int:
        """Read the detection count under the lock (COZMO-3)."""
        with self._detections_lock:
            return len(self.latest_detections)

    def install_camera_lifecycle(self, cli, resolution):
        """Bind the Cozmo camera on/off to WebSocket viewer activity.

        Adds a listener on ``self.web_viewer`` that turns the camera on
        when the first browser tab connects and off when the last one
        closes. This is the structural fix for COZMO-22: the 30 fps
        QVGA stream floods Cozmo's lossy AP and drops the Wi-Fi within
        ~60s, but it's only needed while someone is actually watching.

        Idempotent — calling twice or during a session with no
        transitions is a no-op. The camera starts in the OFF state;
        it will not enable until at least one client connects.

        Args:
            cli: pycozmo client.
            resolution: ``pycozmo.protocol_encoder.ImageResolution`` value
                (e.g. ``QVGA``).
        """
        self._camera_resolution = resolution

        # Timestamp of the last camera on/off transition. Used by the
        # main loop to give Cozmo time to start streaming after we ask
        # for frames — empirically it can take 5-15 s between
        # EnableCamera(enable=True) and the first frame arriving on
        # firmware 2458, so a short staleness window would falsely
        # flag a healthy connection as a Wi-Fi drop. (COZMO-22)
        self._camera_enabled_at: Optional[float] = None

        def _on_clients_change(active_count: int) -> None:
            if active_count > 0 and not self._camera_active:
                print(f"🎥 {active_count} viewer(s) connected — enabling Cozmo camera")
                cli.enable_camera(
                    enable=True,
                    color=False,
                    resolution=self._camera_resolution,
                )
                # Let Cozmo's auto-exposure pick the right gain and
                # shutter speed for the scene, clamped to sane bounds so
                # bright scenes don't wash out. Sending
                # auto_exposure_enabled=True also overrides any prior
                # manual exposure (e.g. from a previous app run). (COZMO-24)
                cli.set_camera_params(
                    gain=0.0,
                    exposure_ms=100,
                    auto_exposure_enabled=True,
                )
                self._camera_active = True
                self._camera_enabled_at = time.time()
                # Reset the staleness clock — it takes a moment for the
                # first frame to arrive after EnableCamera, and the
                # main loop's "no frames for >2 s ⇒ Wi-Fi drop" check
                # would otherwise fire and tear the session down
                # immediately. (COZMO-22)
            elif active_count == 0 and self._camera_active:
                print("🎥 No viewers — disabling Cozmo camera (keeps the Wi-Fi quiet)")
                cli.enable_camera(enable=False)
                self._camera_active = False
                self._camera_enabled_at = None
            # active_count > 0 and _camera_active=True: nothing to do
            # active_count == 0 and _camera_active=False: nothing to do

        self.web_viewer.add_active_clients_listener(_on_clients_change)

    def install_remote_control(self, cli) -> None:
        """Wire browser keyboard input to Cozmo's motors. (COZMO-25)

        Creates a :class:`pycozmo.remote_control.RemoteControl`, starts its
        polling thread, and registers it as the WebViewer's control
        listener. The browser sends a ``{"type": "control", "keys": [...]}``
        message on every keydown/keyup; the RemoteControl instance reads
        the latest set at ~20 Hz and applies ``drive_wheels`` /
        ``move_head`` / ``move_lift`` commands to ``cli``. Stops motors
        cleanly when the last browser tab disconnects or when the
        controller thread is stopped.
        """
        from pycozmo.remote_control import RemoteControl

        self._remote_control = RemoteControl(cli)
        self._remote_control.start()
        self.web_viewer.set_control_listener(self._remote_control.update_keys)
        self.web_viewer.set_navigate_listener(self._on_navigate_to)
        self.web_viewer.set_approach_listener(self._on_approach_cube)
        self.web_viewer.set_dock_listener(self._on_dock)
        # Remember the pose when the app starts — this is assumed to be the dock.
        if self.cli is not None:
            self._dock_pose = self.cli.pose
        print("🎮 Browser keyboard control enabled: WASD drive, R/F head, T/G lift")
        print("📍 Click-to-navigate enabled: click ground to drive there")
        print("📦 Click cube to approach and align")
        print("🛟 Z to dock: drive back to start position")

    def stop_remote_control(self) -> None:
        if hasattr(self, "_remote_control") and self._remote_control is not None:
            self._remote_control.stop()
            self._remote_control = None

    def _on_navigate_to(self, world_forward: float, world_left: float) -> None:
        """Handle click-to-navigate from browser.

        world_forward = X (Anki frame), world_left = Y (Anki frame).
        Long distances use Cozmo's built-in path planner via go_to_pose().
        Short distances (<= _SHORT_DIST_MM) use a closed-loop differential
        drive that can reverse, so targets just behind the robot don't
        force it to sweep around forward-only. (COZMO-28 / COZMO-30)
        Runs in a thread so camera preview keeps working.
        """
        if world_forward is None:
            # Navigation cancelled
            if self.cli:
                self.cli.stop_all_motors()
            print("📍 Navigation cancelled")
            return

        if not self.cli:
            print("📍 Navigation: no robot connected")
            return

        import math
        import threading
        from pycozmo.util import Pose, Angle

        SHORT_DIST_MM = 100.0  # at most this far: attempt reverse-capable drive
        NAV_SPEED_MMPS = 60.0  # closed-loop approach speed
        NAV_STOP_MM = 8.0  # arrive within this distance
        NAV_TURN_FRAC = 1.2  # differential turn magnitude vs heading error
        NAV_TIMEOUT_S = 15.0

        # Current robot pose
        robot_pose = self.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        # Calculate heading to target (Anki frame: X=fwd, Y=left)
        dx = world_forward - robot_x
        dy = world_left - robot_y
        heading = math.atan2(dy, dx)

        # Distance
        dist = math.sqrt(dx * dx + dy * dy)

        def _short_navigate(tx: float, ty: float) -> None:
            """Closed-loop drive to a close target, reversing if needed.

            Computes the heading error to the target; if the target is
            more than 90° behind the robot, it drives in reverse (which
            flips the differential-steering sign so the robot curves
            toward the target while backing up). Stops within NAV_STOP_MM.
            """
            import time as _time

            t0 = _time.monotonic()
            while True:
                if _time.monotonic() - t0 > NAV_TIMEOUT_S:
                    print("📍 Short nav timeout")
                    break

                pose = self.cli.pose
                cx, cy = pose.position.x, pose.position.y
                ex = tx - cx
                ey = ty - cy
                d = math.hypot(ex, ey)
                if d < NAV_STOP_MM:
                    break

                # Heading error to target, normalized to [-pi, pi].
                cur_h = pose.rotation.angle_z.radians
                err = heading - cur_h
                while err > math.pi:
                    err -= 2.0 * math.pi
                while err < -math.pi:
                    err += 2.0 * math.pi

                reverse = abs(err) > math.pi / 2.0
                if reverse:
                    # Drive in reverse. The effective steering error is the
                    # angle of the REAR direction toward the target
                    # (err_rear = err - pi, normalized), and positive
                    # err_rear wants positive omega (r > l), same as forward.
                    if err > 0.0:
                        err -= math.pi
                    else:
                        err += math.pi
                    turn = err * NAV_TURN_FRAC * NAV_SPEED_MMPS
                    l_speed = -NAV_SPEED_MMPS - turn
                    r_speed = -NAV_SPEED_MMPS + turn
                else:
                    turn = err * NAV_TURN_FRAC * NAV_SPEED_MMPS
                    l_speed = NAV_SPEED_MMPS - turn
                    r_speed = NAV_SPEED_MMPS + turn

                # Slow down as we get close so we don't overshoot.
                slow = max(0.2, min(1.0, d / 40.0))
                self.cli.drive_wheels(
                    lwheel_speed=l_speed * slow,
                    rwheel_speed=r_speed * slow,
                )
                _time.sleep(0.05)

            self.cli.stop_all_motors()
            print(f"📍 Arrived at ({tx:.0f}, {ty:.0f})")

        print(
            f"📍 Navigating to ({world_forward:.0f}, {world_left:.0f}) — "
            f"dist={dist:.0f}mm, heading={math.degrees(heading):.0f}°"
        )

        def _nav_thread():
            try:
                if dist <= SHORT_DIST_MM:
                    _short_navigate(world_forward, world_left)
                else:
                    # Use Cozmo's built-in path planner for longer hops.
                    target_pose = Pose(
                        world_forward,
                        world_left,
                        0,
                        angle_z=Angle(radians=heading),
                    )
                    self.cli.go_to_pose(target_pose, relative_to_robot=False)
                    print("📍 Arrived at target")
            except Exception as e:
                print(f"📍 Navigation error: {e}")

        t = threading.Thread(target=_nav_thread, daemon=True)
        t.start()

    def _on_approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Handle cube click from browser — approach cube along its face normal.

        Determines which face is closest to the robot, then approaches
        perpendicular to that face in two steps: 120mm then 80mm.
        (COZMO-28)
        """
        if not self.cli:
            print("📦 Approach: no robot connected")
            return

        import math
        import threading
        import time as _time

        import numpy as np
        from pycozmo.util import Angle, Pose

        STEP1_DIST = 120.0  # mm from cube center (first approach)
        STEP2_DIST = 120.0  # mm from cube center (final alignment)
        BACK_AWAY_DIST = 150.0  # mm from cube center (back away if too close)
        TOO_CLOSE = 100.0  # mm - if closer than this, back away first

        # Current robot pose
        robot_pose = self.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        # Check if too close — need to back away before turning
        dist_to_cube = math.sqrt((cube_x - robot_x) ** 2 + (cube_y - robot_y) ** 2)
        need_back_away = dist_to_cube < TOO_CLOSE

        # Get cube rotation from world map to find face normals
        world = self.web_viewer.world_map.get_world_state()
        cubes = world.get("cubes", {})
        if cube_idx not in cubes:
            print(f"📦 Cube {cube_idx + 1} not in world map")
            return

        cube_rot = np.array(cubes[cube_idx]["rotation"])  # 3x3 rotation matrix

        # Cube's local axes in world frame (Anki frame: X=fwd, Y=left, Z=up)
        local_x = cube_rot @ np.array([1.0, 0.0, 0.0])  # face normal 1
        local_y = cube_rot @ np.array([0.0, 1.0, 0.0])  # face normal 2

        # Four side face normals (ignore top/bottom Z faces)
        face_normals = [
            local_x,  # +X face
            -local_x,  # -X face
            local_y,  # +Y face
            -local_y,  # -Y face
        ]

        # Direction from cube to robot
        to_robot = np.array([robot_x - cube_x, robot_y - cube_y, 0.0])
        to_robot_len = np.linalg.norm(to_robot)
        if to_robot_len < 1:
            print("📦 Already at cube")
            return
        to_robot_unit = to_robot / to_robot_len

        # Find face normal most aligned with direction to robot
        # (the face the robot is closest to / approaching from)
        best_face = None
        best_dot = -2.0
        for normal in face_normals:
            # Project to 2D (ignore Z)
            n2d = np.array([normal[0], normal[1]])
            n2d_len = np.linalg.norm(n2d)
            if n2d_len < 0.01:
                continue
            n2d = n2d / n2d_len
            dot = np.dot(n2d, to_robot_unit[:2])
            if dot > best_dot:
                best_dot = dot
                best_face = n2d

        if best_face is None:
            print("📦 Could not determine approach face")
            return

        # Approach direction: along face normal (from cube toward robot)
        ux, uy = best_face[0], best_face[1]

        # Ensure approach point is BETWEEN robot and cube, not behind cube
        # Check if the face normal points toward the robot or away
        approach_x = cube_x + ux * STEP1_DIST
        approach_y = cube_y + uy * STEP1_DIST
        dist_robot_to_approach = math.sqrt(
            (approach_x - robot_x) ** 2 + (approach_y - robot_y) ** 2
        )
        dist_robot_to_cube = math.sqrt(
            (cube_x - robot_x) ** 2 + (cube_y - robot_y) ** 2
        )
        if dist_robot_to_approach > dist_robot_to_cube:
            # Face normal points away from robot — flip it
            ux, uy = -ux, -uy

        # Heading: face the cube (opposite of approach direction)
        heading = math.atan2(-uy, -ux)

        # Step 1: 120mm away along face normal
        step1_x = cube_x + ux * STEP1_DIST
        step1_y = cube_y + uy * STEP1_DIST

        print(
            f"📦 Approaching cube {cube_idx + 1} at ({cube_x:.0f}, {cube_y:.0f}) "
            f"face normal=({ux:.2f}, {uy:.2f})"
        )

        def _nav_thread():
            try:
                # Tilt the head down so the camera sees the cube closer up
                # while navigating. (head angle in radians, negative = down)
                HEAD_TILT_DEG = -10.0
                self.cli.set_head_angle(math.radians(HEAD_TILT_DEG))
                print(f"📦 Head tilt {HEAD_TILT_DEG:.0f}°")
                _time.sleep(0.3)

                # If too close, back away first so turning doesn't bump cube
                if need_back_away:
                    back_heading = math.atan2(robot_y - cube_y, robot_x - cube_x)
                    back_x = cube_x + (robot_x - cube_x) / dist_to_cube * BACK_AWAY_DIST
                    back_y = cube_y + (robot_y - cube_y) / dist_to_cube * BACK_AWAY_DIST
                    pose_back = Pose(
                        back_x, back_y, 0, angle_z=Angle(radians=back_heading)
                    )
                    print(f"📦 Backing away to ({back_x:.0f}, {back_y:.0f})")
                    self.cli.go_to_pose(pose_back, relative_to_robot=False)
                    _time.sleep(0.3)

                # Step 1: Drive to 120mm
                pose1 = Pose(step1_x, step1_y, 0, angle_z=Angle(radians=heading))
                print(f"📦 Step 1: 120mm ({step1_x:.0f}, {step1_y:.0f})")
                self.cli.go_to_pose(pose1, relative_to_robot=False)

                _time.sleep(0.3)

                # Re-fetch cube position with fresh detections (more accurate close up)
                world2 = self.web_viewer.world_map.get_world_state()
                cubes2 = world2.get("cubes", {})
                if cube_idx in cubes2:
                    cube_x2 = float(cubes2[cube_idx]["position"][0])
                    cube_y2 = float(cubes2[cube_idx]["position"][1])
                    cube_rot2 = np.array(cubes2[cube_idx]["rotation"])

                    # Recalculate face normal with updated rotation
                    local_x2 = cube_rot2 @ np.array([1.0, 0.0, 0.0])
                    local_y2 = cube_rot2 @ np.array([0.0, 1.0, 0.0])
                    normals2 = [local_x2, -local_x2, local_y2, -local_y2]

                    rp = self.cli.pose
                    to_robot2 = np.array(
                        [rp.position.x - cube_x2, rp.position.y - cube_y2, 0.0]
                    )
                    to_robot2_len = np.linalg.norm(to_robot2)
                    if to_robot2_len > 1:
                        to_robot2_unit = to_robot2 / to_robot2_len
                        best2 = None
                        best_dot2 = -2.0
                        for n in normals2:
                            n2d = np.array([n[0], n[1]])
                            if np.linalg.norm(n2d) < 0.01:
                                continue
                            n2d = n2d / np.linalg.norm(n2d)
                            d = np.dot(n2d, to_robot2_unit[:2])
                            if d > best_dot2:
                                best_dot2 = d
                                best2 = n2d
                        if best2 is not None:
                            ux2, uy2 = best2[0], best2[1]
                            # Ensure approach point is between robot and cube
                            app2_x = cube_x2 + ux2 * STEP2_DIST
                            app2_y = cube_y2 + uy2 * STEP2_DIST
                            d_app = math.sqrt(
                                (app2_x - rp.position.x) ** 2
                                + (app2_y - rp.position.y) ** 2
                            )
                            d_cube = math.sqrt(
                                (cube_x2 - rp.position.x) ** 2
                                + (cube_y2 - rp.position.y) ** 2
                            )
                            if d_app > d_cube:
                                ux2, uy2 = -ux2, -uy2
                            step2_x = cube_x2 + ux2 * STEP2_DIST
                            step2_y = cube_y2 + uy2 * STEP2_DIST
                            heading2 = math.atan2(-uy2, -ux2)
                            print(
                                f"📦 Updated: cube at ({cube_x2:.0f}, {cube_y2:.0f}) face=({ux2:.2f}, {uy2:.2f})"
                            )

                # Step 2: Drive to 80mm with updated data
                pose2 = Pose(step2_x, step2_y, 0, angle_z=Angle(radians=heading2))
                print(f"📦 Step 2: 80mm ({step2_x:.0f}, {step2_y:.0f})")
                self.cli.go_to_pose(pose2, relative_to_robot=False)

                print(f"📦 Aligned with cube {cube_idx + 1}")
            except Exception as e:
                print(f"📦 Approach error: {e}")

        t = threading.Thread(target=_nav_thread, daemon=True)
        t.start()

    def _reset_pose_origin(self) -> None:
        """Re-anchor the localization origin at the current pose.

        Sends ``SetOrigin`` so the robot's current pose becomes the new
        world frame (0,0,0) and records it as the dock pose. The tires
        slip when they contact the dock, so this must re-baseline
        whenever charging is detected — during docking AND when the
        robot is driven onto the pad manually. Idempotent per origin id.
        (COZMO-28)
        """
        cli = self.cli
        if cli is None:
            return
        with self._dock_lock:
            if self._origin_reset_origin_id == cli.pose.origin_id:
                return
        import time as _time

        reset_pkt = pycozmo.protocol_encoder.SetOrigin(
            pose_frame_id=cli.pose_frame_id + 1,
            pose_origin_id=cli.pose.origin_id + 1,
        )
        cli.conn.send(reset_pkt)
        _time.sleep(0.5)
        with self._dock_lock:
            self._origin_reset_origin_id = cli.pose.origin_id
            self._dock_pose = cli.pose
        print("🛟 Pose origin reset — current pose is now origin")

    def _on_charger_change(self, cli, state: bool) -> None:
        """Record whether the robot is physically on the charger. (COZMO-28)"""
        del cli
        with self._dock_lock:
            self._on_charger = state
        if state:
            print("🔋 On charger detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _on_charging_change(self, cli, state: bool) -> None:
        """Record whether the robot is actively charging. (COZMO-28)"""
        del cli
        with self._dock_lock:
            self._charging = state
        if state:
            print("🔋 Charging detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _on_dock(self) -> None:
        """Handle 'z' keypress — drive robot back into the dock.

        Assumes robot started at the dock. Drives 150mm forward (away
        from the dock), then reverses back to the dock position and
        expects charging there. If charging isn't detected at contact,
        backs away and retries; gives up after ``DOCK_MAX_RETRIES``.
        On success the pose origin is reset so the slip that happens
        when the tires contact the dock doesn't bias future poses.
        (COZMO-28)
        """
        if not self.cli:
            print("🛟 Dock: no robot connected")
            return
        if self._dock_pose is None:
            print("🛟 Dock: no start pose recorded")
            return

        import math
        import threading
        import time as _time

        from pycozmo.util import Angle, Pose

        DOCK_DRIVE_DIST = 150.0  # mm forward (away from dock)
        DOCK_REVERSE_SPEED = 80.0  # mm/s reverse speed
        DOCK_AIM_BEHIND = 40.0  # mm behind dock center — drives the rear
        # contacts fully onto the charger, not just to the robot center.
        DOCK_CONTACT_DIST = 15.0  # mm — "at" the dock (expect charge here)
        DOCK_MAX_RETRIES = 3  # times to back away and re-attempt
        DOCK_RETRY_SPEED = 60.0  # mm/s when backing away between attempts
        DOCK_RETRY_DRIVE = 40.0  # mm to back away between attempts
        DOCK_STALL_ITERS = 20  # stop reversing after this many no-progress iters

        dock = self._dock_pose
        dock_x = dock.position.x
        dock_y = dock.position.y
        dock_heading = dock.rotation.angle_z.radians

        # Point 150mm in front of the dock (along the dock heading).
        # The robot keeps heading dock_heading here (facing away from the
        # dock) so the dock ends up directly behind it.
        turn_x = dock_x + DOCK_DRIVE_DIST * math.cos(dock_heading)
        turn_y = dock_y + DOCK_DRIVE_DIST * math.sin(dock_heading)

        # Aim slightly BEHIND the dock so the robot's rear (contacts)
        # seat on the charger — reversing to the dock center leaves the
        # robot a robot-length short of good contact. (COZMO-28)
        aim_x = dock_x - DOCK_AIM_BEHIND * math.cos(dock_heading)
        aim_y = dock_y - DOCK_AIM_BEHIND * math.sin(dock_heading)

        def _is_charging() -> bool:
            with self._dock_lock:
                return self._on_charger or self._charging

        def _dist_to_dock() -> float:
            rp = self.cli.pose
            dx = aim_x - rp.position.x
            dy = aim_y - rp.position.y
            return math.sqrt(dx * dx + dy * dy)

        def _reverse_until_contact() -> bool:
            """Reverse until contact/stall. Returns True if charge detected."""
            prev_dist = None
            stall = 0
            while True:
                if _is_charging():
                    return True

                dist = _dist_to_dock()

                # At the dock position — expect charging now.
                if dist < DOCK_CONTACT_DIST:
                    return False

                # Stall: no longer making progress (grinding on the dock).
                if prev_dist is not None and dist >= prev_dist - 1.0:
                    stall += 1
                    if stall >= DOCK_STALL_ITERS:
                        return False
                else:
                    stall = 0
                prev_dist = dist

                self.cli.drive_wheels(
                    lwheel_speed=-DOCK_REVERSE_SPEED,
                    rwheel_speed=-DOCK_REVERSE_SPEED,
                )
                _time.sleep(0.05)

        def _dock_thread():
            try:
                # Step 1: Drive to 150mm in front, keeping the heading
                turn_pose = Pose(turn_x, turn_y, 0, angle_z=Angle(radians=dock_heading))
                print("🛟 Driving to 150mm in front of dock")
                self.cli.go_to_pose(turn_pose, relative_to_robot=False)
                _time.sleep(0.3)

                # Step 2: Reverse in; if no charge at contact, back away and
                # retry. Give up after DOCK_MAX_RETRIES attempts.
                docked = False
                for attempt in range(1, DOCK_MAX_RETRIES + 1):
                    print(f"🛟 Docking attempt {attempt}/{DOCK_MAX_RETRIES}")
                    if _reverse_until_contact():
                        docked = True
                        break

                    if _is_charging():
                        docked = True
                        break

                    # At the dock but no charge: back away and try again.
                    if attempt < DOCK_MAX_RETRIES:
                        print("🛟 No charge at contact — backing away to retry")
                        self.cli.drive_wheels(
                            lwheel_speed=DOCK_RETRY_SPEED,
                            rwheel_speed=DOCK_RETRY_SPEED,
                            duration=DOCK_RETRY_DRIVE / DOCK_RETRY_SPEED,
                        )
                        _time.sleep(0.3)
                    else:
                        print(
                            f"🛟 Docking failed after {DOCK_MAX_RETRIES} attempts "
                            f"(dist={_dist_to_dock():.0f}mm, no charge)"
                        )

                self.cli.stop_all_motors()

                # Always reset the origin when charging is detected — the
                # tires slip when they contact the dock, so the pose is no
                # longer trustworthy. Reset makes the dock the new (0,0,0)
                # and updates _dock_pose for the next docking run.
                # (Handlers also trigger this for manual driving onto the
                # pad; _reset_pose_origin is idempotent per origin id.)
                if docked or _is_charging():
                    self._reset_pose_origin()
                    print("🛟 Docked and pose reset — dock is now origin")
                elif not docked:
                    print("🛟 Docking did not complete — no pose reset")
            except Exception as e:
                print(f"🛟 Dock error: {e}")

        t = threading.Thread(target=_dock_thread, daemon=True)
        t.start()

    def setup_cube_mapping(self, cli):
        """Discover available cubes and create mapping from detected cube_id to factory_id."""
        print("\n" + "=" * 70)
        print("🔗 DISCOVERING CUBES")
        print("=" * 70)

        timeout = 5.0
        start_time = time.time()
        print(f"   Waiting up to {timeout}s for cubes to be discovered...")
        while time.time() - start_time < timeout:
            available_objects = dict(cli.available_objects)
            if available_objects:
                print(f"   ✅ Found {len(available_objects)} objects!")
                break
            time.sleep(0.1)
        else:
            print("   ⏱️  Timeout waiting for objects")

        available_objects = dict(cli.available_objects)
        if not available_objects:
            print("⚠️  No cubes found")
            return

        cube_types = [
            pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1,
            pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE2,
            pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE3,
        ]

        available_cubes = []
        for factory_id, obj in available_objects.items():
            if obj.object_type in cube_types:
                available_cubes.append((factory_id, obj.object_type))

        if not available_cubes:
            print("⚠️  No light cubes found")
            return

        print("\n📋 Available cubes:")
        for factory_id, obj_type in available_cubes:
            if obj_type == pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1:
                detected_cube_id = 1
            elif obj_type == pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE2:
                detected_cube_id = 2
            elif obj_type == pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE3:
                detected_cube_id = 3
            else:
                continue

            self.cube_factory_id_map[detected_cube_id] = factory_id
            print(
                f"   Detected cube_id {detected_cube_id} -> Factory ID 0x{factory_id:08x} ({obj_type.name})"
            )

        print(f"\n✅ Discovered {len(self.cube_factory_id_map)} cube(s)")
        print(f"📋 Cube mapping: {self.cube_factory_id_map}")
        logger_cubes.info("Cube factory_id map: %s", self.cube_factory_id_map)

        # Start the rapid cube cycler — cycles through cubes one at a time
        # to collect accel data from all cubes despite the 1-connection limit.
        self.start_cube_cycler(cli)

        print("=" * 70 + "\n")

    def run(self, cli):
        """Main display loop."""
        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # Cache the integer value of Connection.CONNECTED once.
        _CONNECTED = pycozmo.conn.Connection.CONNECTED

        while self.running:
            # Detect Cozmo disconnect (Wi-Fi loss, robot reboot, etc.)
            # and break out cleanly so the worker thread stops too.
            if cli.conn.state != _CONNECTED:
                logger_cubes.warning(
                    "Connection state is %s — exiting main loop",
                    cli.conn.state,
                )
                self.running = False
                break

            # No camera frames for >5 s means the Wi-Fi AP dropped us.
            # cli.conn.state stays CONNECTED in that case because no
            # Disconnect packet arrives; the staleness check is the only
            # reliable signal. Skipped when the camera is off (lazy mode,
            # COZMO-22) because no frames is the normal state then.
            # Also given a 5-second grace period after enable, because
            # Cozmo firmware 2458 can take a long time to actually start
            # streaming after EnableCamera. (COZMO-22)
            if (
                self._camera_active
                and self._camera_enabled_at is not None
                and time.time() - self._camera_enabled_at > 5.0
            ):
                stale_for = time.time() - self._last_frame_time
                if stale_for > 5.0:
                    logger_cubes.warning(
                        "No camera frames for %.1fs — assuming Wi-Fi drop, "
                        "exiting main loop",
                        stale_for,
                    )
                    self.running = False
                    break

            self.display_frames()

            if self.headless:
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                # Quit cleanly (COZMO-1)
                pkt = pycozmo.protocol_encoder.StopAllMotors()
                cli.conn.send(pkt)
                self.running = False
                break
            elif key == ord("s") or key == ord("S"):
                self.save_frames()
            elif key == ord("c") or key == ord("C"):
                self.detection_enabled = not self.detection_enabled
                # Pause/resume the actual pipeline so the HUD matches reality.
                # Previously only the cube-light update was gated, leaving
                # detection, bbox annotation and web-viewer broadcasts running
                # while the user thought detection was off. (COZMO-14)
                if self.vision_processor is not None:
                    if self.detection_enabled:
                        self.vision_processor.resume()
                    else:
                        self.vision_processor.pause()
                print(
                    f"🔍 Detection {'ENABLED' if self.detection_enabled else 'DISABLED'}"
                )
            elif key == ord("f") or key == ord("F"):
                self.show_fps = not self.show_fps
                print(f"📊 FPS overlay {'ON' if self.show_fps else 'OFF'}")
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
                print("🗑️  Path trail cleared")

        if not self.headless:
            cv2.destroyAllWindows()
        self.stop_remote_control()
        self.web_viewer.close()


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="PyCozmo Vision System Test")
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Disable image undistortion (disabled by default - undistortion may cause warping)",
    )
    parser.add_argument(
        "--process-every-n-frames",
        type=int,
        default=1,
        help=(
            "Run cube detection every Nth camera frame. 1 = process every "
            "frame (default, ~30 fps on Pi 5). 2 = every other frame "
            "(~15 fps, lighter CPU). Higher values further throttle the "
            "pipeline; useful if the detection/preview starts to lag. "
            "Raw camera preview is unaffected."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help=(
            "Log level for the pycozmo.* loggers. INFO enables the cube-light "
            "worker diagnostics (cube connect/disconnect, per-tick action), "
            "DEBUG adds per-packet detail."
        ),
    )
    parser.add_argument(
        "--cubes-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help=(
            "Override the log level specifically for the pycozmo.cubes "
            "logger (cube-light worker diagnostics). Useful when "
            "--log-level is WARNING but you still want cube trace."
        ),
    )
    parser.add_argument(
        "--no-cube-lights",
        action="store_true",
        help=(
            "Disable the cube keepalive worker (no reconnection / "
            "CubeLights refresh). Cubes will disconnect after the robot's "
            "idle timeout and not be reconnected."
        ),
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help=(
            "Skip enabling the Cozmo camera (no image streaming, no "
            "detection, no display). Use this to isolate whether the "
            "30-fps grayscale camera stream is overloading Cozmo's Wi-Fi "
            "AP. With this flag you should see no Wi-Fi drop for as long "
            "as the app runs."
        ),
    )
    parser.add_argument(
        "--camera-resolution",
        choices=("QQQVGA", "QQVGA", "QVGA", "VGA"),
        default="QVGA",
        help=(
            "Camera resolution (default: QVGA = 320x240). Smaller "
            "resolutions are ~4x less Wi-Fi bandwidth per frame, which "
            "reduces the chance of Cozmo's AP dropping the connection. "
            "Try QQVGA (160x120) if the Wi-Fi keeps dropping. The cube "
            "detector runs at any of these sizes — markers are still "
            "detected at 160x120."
        ),
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=0.0,
        help=(
            "**Experimental, currently broken.** If >0, attempts to use "
            "ImageSendMode.SingleShot with a Timer-based re-arm to cap the "
            "inbound rate. In practice this DROPS the Wi-Fi FASTER than "
            "the default Stream mode (~8.5s vs ~60s) on firmware 2458, "
            "so leave at 0 (default) until COZMO-22 is fixed. (COZMO-20)"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_cubes.setLevel(getattr(logging, args.cubes_log_level))

    display = VisionDisplay()

    try:
        connect_to_cozmo_wifi()

        print("🤖 Connecting to Cozmo...")
        with pycozmo.connect() as cli:
            print("✅ Connected!\n")

            print("📷 Positioning robot...")
            angle = (
                pycozmo.robot.MAX_HEAD_ANGLE.radians
                - pycozmo.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0

            cli.set_head_angle(angle)
            time.sleep(1.0)

            print("📐 Retrieving camera calibration from Cozmo...")
            calibration_retriever = CameraCalibrationRetriever(cli)
            camera_calibration = calibration_retriever.get_calibration(timeout=10.0)

            print("✅ Camera calibration retrieved:")
            print(f"   fx={camera_calibration.fx:.2f}, fy={camera_calibration.fy:.2f}")
            print(f"   cx={camera_calibration.cx:.2f}, cy={camera_calibration.cy:.2f}")
            print(
                f"   k1={camera_calibration.k1:.6f}, k2={camera_calibration.k2:.6f}, k3={camera_calibration.k3:.6f}"
            )
            print(f"   p1={camera_calibration.p1:.6f}, p2={camera_calibration.p2:.6f}")
            print(
                f"   Image size: {camera_calibration.image_width}x{camera_calibration.image_height}"
            )
            print("   ℹ️  Using zero distortion coefficients for Cozmo reliability")

            display.cli = cli
            cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage, display.on_raw_camera_image
            )
            cli.add_handler(
                pycozmo.event.EvtRobotOnChargerChange, display._on_charger_change
            )
            cli.add_handler(
                pycozmo.event.EvtRobotChargingChange, display._on_charging_change
            )

            print("🎨 Initializing vision processor with cubeDetector.py...")
            undistort_enabled = not args.no_undistort
            print(
                f"   Undistortion: {'ENABLED' if undistort_enabled else 'DISABLED (default - undistortion may cause warping)'}"
            )
            vision_processor = vision.AnnotatedVisionProcessor(
                client=cli,
                camera_calibration=camera_calibration,
                process_every_n_frames=max(1, args.process_every_n_frames),
                undistort=undistort_enabled,
            )

            print("✅ Cube detector initialized with camera calibration")

            display.vision_processor = vision_processor

            vision_processor.show_confidence = True
            vision_processor.show_cube_id = True
            vision_processor.box_thickness = 2

            cli.add_handler(
                pycozmo.event.EvtAnnotatedCameraImage, display.on_annotated_camera_image
            )

            cli.add_handler(
                pycozmo.event.EvtCubeDetectorDebugImages, display.on_debug_images
            )

            cli.add_handler(
                pycozmo.protocol_encoder.ObjectConnectionState,
                display.on_object_connection_state,
            )

            # ------------------------------------------------------------------
            # Cube relocation — DISABLED (COZMO-28)
            # User requested no cube position reset on move.
            # ------------------------------------------------------------------
            def _invalidate_cube_pose(cube_id, source):
                pass  # Disabled — don't reset cube position on move

            def _on_object_accel(c, pkt):
                pass  # Disabled — no cube relocation

            def _on_object_moved(c, pkt):
                pass  # Disabled — no cube relocation

            # cli.add_handler(pycozmo.protocol_encoder.ObjectAccel, _on_object_accel)
            # cli.add_handler(pycozmo.protocol_encoder.ObjectMoved, _on_object_moved)
            if not args.no_cube_lights:
                display.start_cube_light_worker(cli)
            else:
                print(
                    "ℹ️  Cube keepalive worker disabled (--no-cube-lights)."
                    "\n   Cubes will disconnect after idle timeout."
                )

            if not args.no_camera:
                resolution = getattr(
                    pycozmo.protocol_encoder.ImageResolution, args.camera_resolution
                )
                if args.camera_fps > 0:
                    print(
                        f"   ⚠️  --camera-fps {args.camera_fps} is experimental "
                        "and currently makes the Wi-Fi drop FASTER. Use --no-camera "
                        "instead. (COZMO-20)"
                    )

                # Lazy camera: the camera starts OFF and only turns on
                # when a browser tab opens the web viewer. This is the
                # COZMO-22 fix — 30 fps QVGA floods Cozmo's lossy AP
                # and drops the Wi-Fi within ~60s, but the camera is
                # only needed while someone is actually watching.
                print("🎥 Camera is OFF by default.")
                print(
                    "   It will turn on automatically when a browser opens the "
                    "web viewer, and off again when the last tab closes."
                )
                print(f"   Resolution when enabled: {args.camera_resolution}")
                display.install_camera_lifecycle(cli, resolution)

                time.sleep(2.0)

                display.setup_cube_mapping(cli)

                # Browser-side WASD/RF/TG remote control. (COZMO-25)
                display.install_remote_control(cli)
            else:
                print("ℹ️  Camera disabled (--no-camera).")
                print("   No image streaming, no detection, no display.")
                print("   Keeping the connection alive for diagnostics only.")

            print("📍 Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin set to current position")

            # Pose trail + robot pose — push every RobotState into the world
            # map so the path is tracked even with --no-camera.  The map's
            # robot pose is what transforms cube measurements into world
            # frame, so it is refreshed here (~30 Hz, independent of the
            # camera).  The map is reset on origin change (pick-up /
            # put-down) and the 'p' key. (COZMO-27 / world-map refactor)
            def _on_robot_state_for_trail(c):
                try:
                    pose = c.pose
                    pos = pose.position.x_y_z
                    yaw = pose.rotation.angle_z.radians
                    display.web_viewer.update_robot_pose(
                        pos, yaw, origin_id=pose.origin_id, record_path=True
                    )
                except Exception:  # noqa: BLE001 — pose may be transiently invalid
                    pass

                # Telemetry for the browser HUD. Charging flags come from
                # the charge-change handlers (display._on_charger /
                # display._charging); the rest is read straight off the
                # latest RobotState packet. (COZMO-28)
                try:
                    with display._dock_lock:
                        on_charger = display._on_charger
                        charging = display._charging
                    display.web_viewer.update_telemetry(
                        charging=charging,
                        on_charger=on_charger,
                        battery_voltage=round(c.battery_voltage, 3),
                        left_wheel_speed=round(c.left_wheel_speed.mmps, 1),
                        right_wheel_speed=round(c.right_wheel_speed.mmps, 1),
                        head_angle=round(c.head_angle.degrees, 1),
                        lift_height=round(c.lift_position.height.mm, 1),
                    )
                except Exception:  # noqa: BLE001 — fields may be unavailable
                    pass

                display.web_viewer.broadcast()

            cli.add_handler(
                pycozmo.event.EvtRobotStateUpdated, _on_robot_state_for_trail
            )

            # Clear the trail immediately on pick-up so we don't record
            # stale drifting poses while Cozmo is in the air.  The put-down
            # path is already handled by origin_id change (SetOrigin fires
            # in client._on_robot_picked_up). (COZMO-27)
            def _on_picked_up(c, state):
                if state:
                    display.web_viewer.clear_path()

            cli.add_handler(pycozmo.event.EvtRobotPickedUpChange, _on_picked_up)

            if args.no_camera:
                print(
                    "ℹ️  --no-camera: skipping display loop. "
                    "Connection will stay idle. Press Ctrl-C to quit."
                )
                try:
                    # Idle loop — just keep the connection alive so we can
                    # see whether Cozmo drops Wi-Fi on its own, without any
                    # of our app's traffic (camera, detection, cube lights,
                    # display).
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    pass
            else:
                print("\n✨ Starting display...\n")
                print("🌐 Open http://<this-machine-ip>:8766 in a browser\n")

                display.run(cli)
                display.stop_cube_light_worker()

                print("\n✅ Display closed")

    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()
