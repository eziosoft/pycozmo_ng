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
        # Per-cube "last seen" frame index, for diagnostics only. The cube-
        # light worker no longer uses it to drive disconnects — Cozmo drops
        # out-of-range cubes on its own, and our old proactive disconnect
        # caused spurious drops when paired with multi-cube connects. (COZMO-20)
        self._cube_last_seen_frame = {}
        # Wall-clock timestamp of the most recently received camera frame.
        # Used by the main run loop to detect a silent Wi-Fi drop (see run).
        self._last_frame_time = time.time()

    def on_object_connection_state(self, cli, pkt):
        """Log cube connection state transitions for diagnostics.

        Cozmo's firmware only allows ONE cube connected at a time and
        always assigns ``object_id=1`` to that single slot. ``cli.connected_objects``
        never holds more than one entry. We don't need to cache object_ids
        — the cube-light worker hardcodes ``object_id=1`` and just relies
        on each ``CubeId + CubeLights`` arriving while the right cube is
        in the slot. (COZMO-20)
        """
        cube_id = self._cube_id_for_factory(pkt.factory_id)
        if cube_id is None:
            logger_cubes.warning(
                "ObjectConnectionState for unknown factory_id 0x%08x "
                "(connected=%s, object_id=%s)",
                pkt.factory_id, pkt.connected, pkt.object_id,
            )
            return
        if pkt.connected:
            logger_cubes.info(
                "Cube %d connected: object_id=%s", cube_id, pkt.object_id,
            )
        else:
            logger_cubes.info(
                "Cube %d disconnected (object_id=%s)",
                cube_id, pkt.object_id,
            )

    def _cube_id_for_factory(self, factory_id: int) -> Optional[int]:
        """Reverse lookup: factory_id -> detected cube_id (1..3)."""
        for cube_id, fid in self.cube_factory_id_map.items():
            if fid == factory_id:
                return cube_id
        return None

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

    def start_cube_light_worker(self, cli):
        """Spawn a background thread that cycles cube lights every second.

        Runs on its own thread so the small sleeps inside the cycle
        don't block the pycozmo event dispatcher (which would drop camera
        frames — see COZMO-2). The worker exits when ``self.running`` goes
        False. (COZMO-20)
        """
        self._cube_light_stop = threading.Event()

        def _loop():
            # Run one cycle, then pause for CYCLE_PERIOD_S before the next.
            # Empirically the 0.8 s cycle triggered a Wi-Fi drop on the
            # Pi after a few seconds — Cozmo's connection stack seems to
            # give up if we hammer ObjectConnect too often. The 2 s cycle
            # sustained 100 % packet delivery in earlier runs, so use that
            # and trust the LEDs to persist across the gap. (COZMO-20)
            CYCLE_PERIOD_S = 2.0
            while not self._cube_light_stop.is_set():
                try:
                    self._cube_light_tick(cli)
                except Exception as e:
                    print(f"⚠️  Cube light tick failed: {e}")
                if self._cube_light_stop.wait(timeout=CYCLE_PERIOD_S):
                    return

        thread = threading.Thread(
            target=_loop, name="CubeLightWorker", daemon=True
        )
        thread.start()
        return thread

    def stop_cube_light_worker(self):
        """Signal the cube-light worker to exit."""
        if hasattr(self, "_cube_light_stop"):
            self._cube_light_stop.set()

    def on_debug_images(self, cli, debug_images, detections):
        """Handler for debug images from cube detection pipeline."""
        self.latest_debug_images = debug_images

    def update_web_viewer(self):
        """Update web viewer with current cube states."""
        if not self.vision_processor:
            return

        fused_cubes = self.vision_processor.get_fused_cube_states()

        if not hasattr(self, "_prev_cube_ids"):
            self._prev_cube_ids = set()

        current_cube_ids = set(fused_cubes.keys())

        # Remove only cubes that truly disappeared. Diffing instead of
        # clear_cubes() keeps the browser's cube state stable (no pop or
        # teleport on transient detection flicker).
        for cube_id in self._prev_cube_ids - current_cube_ids:
            self.web_viewer.remove_cube(f"cube_{cube_id + 1}")

        for cube_id, cube_state in fused_cubes.items():
            cube_name = f"cube_{cube_id + 1}"
            self.web_viewer.update_cube(
                cube_name,
                cube_state["position"],
                cube_state["rotation_matrix"],
                cube_size=44,
            )

        self._prev_cube_ids = current_cube_ids
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
                print(
                    f"🎥 {active_count} viewer(s) connected — "
                    "enabling Cozmo camera"
                )
                cli.enable_camera(
                    enable=True, color=False, resolution=self._camera_resolution,
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
                self._last_frame_time = time.time()
            elif active_count == 0 and self._camera_active:
                print(
                    "🎥 No viewers — disabling Cozmo camera "
                    "(keeps the Wi-Fi quiet)"
                )
                cli.enable_camera(enable=False)
                self._camera_active = False
                self._camera_enabled_at = None
            # active_count > 0 and _camera_active=True: nothing to do
            # active_count == 0 and _camera_active=False: nothing to do

        self.web_viewer.add_active_clients_listener(_on_clients_change)

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
        print(
            "ℹ️  Cozmo connects one cube at a time; the light worker cycles"
            "\n   them so all detected cubes light up."
        )
        print("=" * 70 + "\n")

    def _cube_light_tick(self, cli):
        """Run one cube-light update cycle.

        Cozmo only allows one cube connected at a time and always assigns
        ``object_id=1`` to the single connected cube. The protocol docs
        call object_id "temporary", but in practice it is a fixed slot
        index — there is only one slot, hence always 1.

        For each detected cube we fire ``ObjectConnect`` + ``CubeId`` +
        ``CubeLights`` back-to-back with no waiting. The robot processes
        the packets in order: it implicitly disconnects whatever was in
        the slot, connects our cube, then receives the CubeId and
        CubeLights as soon as the slot has been re-allocated. The lights
        then need to persist across the next disconnect — empirically
        they do, for a few seconds at least, so cycling through N cubes
        every cycle keeps them all illuminated. (COZMO-20)
        """
        # Bail out if the robot has dropped the Wi-Fi connection.
        if cli.conn.state != pycozmo.conn.Connection.CONNECTED:
            logger_cubes.debug(
                "Skipping light tick: conn state=%s", cli.conn.state,
            )
            return

        if not self.detection_enabled:
            return
        if not self.cube_factory_id_map:
            return

        # Snapshot detections under the lock (COZMO-3).
        with self._detections_lock:
            detected_cube_ids = sorted(
                {det.cube_id for det in self.latest_detections}
            )

        if not hasattr(self, "_last_detected_ids"):
            self._last_detected_ids = set()

        if set(detected_cube_ids) != self._last_detected_ids:
            print(f"🔍 Detected cube IDs: {detected_cube_ids}")
            self._last_detected_ids = set(detected_cube_ids)

        if not detected_cube_ids:
            return

        logger_cubes.debug(
            "Light tick: detected=%s", detected_cube_ids,
        )

        # Cozmo's single connection slot always gets object_id=1.
        OBJECT_ID = 1
        # Pause between cubes so the robot has time to register each
        # connect before the next one arrives. Empirically <50 ms is
        # not enough; 200 ms works reliably without making the cycle
        # feel sluggish.
        BETWEEN_CUBE_S = 0.2

        for cube_id in detected_cube_ids:
            if cube_id not in self.cube_factory_id_map:
                continue
            factory_id = self.cube_factory_id_map[cube_id]

            try:
                # All three packets back-to-back. The robot will accept
                # CubeId+CubeLights addressed to the slot even if the
                # connect ack hasn't reached us yet — the robot processes
                # commands in the order we send them.
                cli.conn.send(
                    pycozmo.protocol_encoder.ObjectConnect(
                        factory_id=factory_id, connect=True
                    )
                )
                cli.conn.send(
                    pycozmo.protocol_encoder.CubeId(object_id=OBJECT_ID)
                )
                cli.conn.send(
                    pycozmo.protocol_encoder.CubeLights(
                        states=(
                            pycozmo.lights.blue_light,
                            pycozmo.lights.blue_light,
                            pycozmo.lights.blue_light,
                            pycozmo.lights.blue_light,
                        )
                    )
                )
                logger_cubes.debug(
                    "Cube %d: cycle sent (factory_id=0x%08x)",
                    cube_id, factory_id,
                )
            except Exception as e:
                logger_cubes.warning(
                    "Cube %d: cycle failed: %s", cube_id, e,
                )

            # Brief settle before moving on to the next cube, so the
            # robot can process this cube's connect before we ask it to
            # disconnect the same cube and connect a different one.
            time.sleep(BETWEEN_CUBE_S)

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
            if self._camera_active and self._camera_enabled_at is not None \
                    and time.time() - self._camera_enabled_at > 5.0:
                stale_for = time.time() - self._last_frame_time
                if stale_for > 5.0:
                    logger_cubes.warning(
                        "No camera frames for %.1fs — assuming Wi-Fi drop, "
                        "exiting main loop", stale_for,
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

        if not self.headless:
            cv2.destroyAllWindows()
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
            "Disable the cube-light worker entirely (no ObjectConnect / "
            "CubeLights cycling). Use this for debugging Wi-Fi stability — "
            "if Cozmo stays connected with --no-cube-lights, the worker "
            "is the cause of the disconnects."
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

            cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage, display.on_raw_camera_image
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
            if not args.no_cube_lights:
                display.start_cube_light_worker(cli)
            else:
                print(
                    "ℹ️  Cube-light worker disabled (--no-cube-lights)."
                    "\n   No cubes will be lit; detection still runs."
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
            else:
                print("ℹ️  Camera disabled (--no-camera).")
                print("   No image streaming, no detection, no display.")
                print("   Keeping the connection alive for diagnostics only.")

            print("📍 Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin set to current position")

            # NOTE: Robot state is already maintained on the client
            # (cli.head_angle, cli.battery_voltage, cli.pose, ...). No
            # application-level handler is needed, so we don't subscribe to
            # EvtRobotStateUpdated. (COZMO-5)

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
                print(
                    "🌐 Open http://<this-machine-ip>:8766 in a browser\n"
                )

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
