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

Brightness/gamma tuning for the web viewer MJPEG stream:
  python main.py --gamma 0.5          # compensate for Cozmo's over-bright feed
  python main.py --brightness 0.7     # dim the image a bit
  python main.py --gamma 0.4 --brightness 0.8  # combine both

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

    def __init__(self, brightness: float = 1.0, gamma: float = 1.0):
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
        self.web_viewer = WebViewerServer(brightness=brightness, gamma=gamma)
        self.web_viewer.start()

        # Window names
        self.window_name = "PyCozmo Vision Test - Normal (Left) | Annotated (Right)"

        # Cube tracking variables
        self.cube_factory_id_map = {}
        # Map: detected cube_id (1..3) -> object_id reported by the robot in
        # ObjectConnectionState. Populated from on_object_connection_state()
        # on the event-dispatcher thread, read by the cube-light worker
        # thread. Guarded by ``_object_id_lock``. (COZMO-20)
        self._factory_id_to_object_id = {}
        self._object_id_lock = threading.Lock()
        # Per-cube "last seen" frame index, for diagnostics only. The cube-
        # light worker no longer uses it to drive disconnects — Cozmo drops
        # out-of-range cubes on its own, and our old proactive disconnect
        # caused spurious drops when paired with multi-cube connects. (COZMO-20)
        self._cube_last_seen_frame = {}

    def on_object_connection_state(self, cli, pkt):
        """Track per-cube object_ids so CubeLights can target them later.

        Cozmo's firmware only allows ONE cube connected at a time, so
        ``cli.connected_objects`` never holds more than one entry. To light
        multiple cubes we must cycle ``ObjectConnect``/``CubeId``/``CubeLights``
        for each cube in turn. Remembering the ``object_id`` per cube lets us
        send ``CubeId`` confidently on the next connect cycle even if the
        robot has already auto-disconnected the cube.

        Important: ``object_id`` is stable per physical cube (factory_id) for
        the lifetime of the Cozmo session. We therefore **keep the cached
        object_id** even when the robot sends ``connected=False`` — the
        disconnect is just informational (we explicitly cycled to another
        cube) and the same object_id will be re-issued on reconnect. (COZMO-20)
        """
        cube_id = self._cube_id_for_factory(pkt.factory_id)
        if cube_id is None:
            # Unknown factory_id — robot reports a cube we haven't mapped.
            # This can happen if a new cube is discovered mid-session.
            logger_cubes.warning(
                "ObjectConnectionState for unknown factory_id 0x%08x "
                "(connected=%s, object_id=%s)",
                pkt.factory_id, pkt.connected, pkt.object_id,
            )
            return
        with self._object_id_lock:
            if pkt.connected:
                prev = self._factory_id_to_object_id.get(cube_id)
                self._factory_id_to_object_id[cube_id] = pkt.object_id
                if prev != pkt.object_id:
                    logger_cubes.info(
                        "Cube %d connected: object_id=%s (was %s)",
                        cube_id, pkt.object_id, prev,
                    )
            else:
                # Don't drop the cached object_id — see docstring above.
                logger_cubes.info(
                    "Cube %d disconnected (object_id=%s kept in cache)",
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
            # Cycle fast enough that, even if Cozmo turns a cube's LEDs off
            # the moment it disconnects, every cube gets re-lit before a
            # human eye notices the gap. With 3 cubes detected and
            # ~200 ms ObjectConnect settle per cube, one full cycle is
            # ~0.6 s — fast enough to look continuous.
            tick_period = 0.6
            tick_count = 0
            while not self._cube_light_stop.is_set():
                try:
                    self._cube_light_tick(cli)
                except Exception as e:
                    print(f"⚠️  Cube light tick failed: {e}")
                tick_count += 1
                # Every 5 ticks (≈3 s) log a one-line status summary so
                # the operator can see which cubes have a cached object_id
                # vs which are still waiting for the first ack.
                if tick_count % 5 == 0:
                    with self._object_id_lock:
                        known = dict(self._factory_id_to_object_id)
                    logger_cubes.info(
                        "cube map: known_object_ids=%s", known,
                    )
                # Interruptible sleep: bail out immediately on shutdown.
                if self._cube_light_stop.wait(timeout=tick_period):
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

        Cozmo only allows one cube connected at a time, so to light N cubes
        we cycle ``ObjectConnect`` -> ``CubeId`` -> ``CubeLights`` for each
        one. This runs on a background thread so the small sleeps between
        cycles don't block the pycozmo event dispatcher (COZMO-2). The
        detection pipeline (``on_annotated_camera_image``) only enqueues a
        request; the worker does the actual work. (COZMO-20)
        """
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

        # Snapshot object_ids under the lock so the worker doesn't race
        # the ObjectConnectionState handler.
        with self._object_id_lock:
            object_id_map = dict(self._factory_id_to_object_id)

        logger_cubes.debug(
            "Light tick: detected=%s known_object_ids=%s",
            detected_cube_ids, object_id_map,
        )

        # For each detected cube: connect, wait for ack, light it.
        # Between cubes we wait long enough for the connect ack to land
        # so CubeId always targets the right cube.
        connect_timeout_s = 0.1
        poll_interval_s = 0.02

        for cube_id in detected_cube_ids:
            if cube_id not in self.cube_factory_id_map:
                continue
            factory_id = self.cube_factory_id_map[cube_id]
            object_id = object_id_map.get(cube_id)

            try:
                cli.conn.send(
                    pycozmo.protocol_encoder.ObjectConnect(
                        factory_id=factory_id, connect=True
                    )
                )
            except Exception as e:
                logger_cubes.warning(
                    "Cube %d: ObjectConnect send failed: %s", cube_id, e,
                )
                continue

            # If we already have a cached object_id from a previous tick,
            # trust it. Otherwise poll for the ack by watching the
            # ObjectConnectionState handler (which writes to our map under
            # _object_id_lock) until it records this factory_id, or the
            # timeout elapses.
            if object_id is None:
                deadline = time.time() + connect_timeout_s
                while time.time() < deadline:
                    time.sleep(poll_interval_s)
                    with self._object_id_lock:
                        object_id = self._factory_id_to_object_id.get(cube_id)
                    if object_id is not None:
                        break

            if object_id is None:
                logger_cubes.warning(
                    "Cube %d: no object_id after %.2fs — skipping lights",
                    cube_id, connect_timeout_s,
                )
                continue

            try:
                cli.conn.send(
                    pycozmo.protocol_encoder.CubeId(object_id=object_id)
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
                    "Cube %d: sent CubeId(object_id=%s) + CubeLights",
                    cube_id, object_id,
                )
            except Exception as e:
                logger_cubes.warning(
                    "Cube %d: CubeId/CubeLights failed: %s", cube_id, e,
                )

    def run(self, cli):
        """Main display loop."""
        if not self.headless:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
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
        "--brightness",
        type=float,
        default=1.0,
        help="MJPEG camera brightness multiplier (1.0 = identity, <1 = darker, >1 = brighter)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="MJPEG camera gamma exponent (1.0 = identity, <1 = brighter, >1 = darker; try 0.4–0.7 for Cozmo)",
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
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_cubes.setLevel(getattr(logging, args.cubes_log_level))

    display = VisionDisplay(brightness=args.brightness, gamma=args.gamma)

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
            display.start_cube_light_worker(cli)

            print("🎥 Enabling camera...")
            cli.enable_camera(enable=True, color=False)
            cli.set_camera_params(exposure_ms=100)

            time.sleep(2.0)

            display.setup_cube_mapping(cli)

            print("📍 Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin set to current position")

            # NOTE: Robot state is already maintained on the client
            # (cli.head_angle, cli.battery_voltage, cli.pose, ...). No
            # application-level handler is needed, so we don't subscribe to
            # EvtRobotStateUpdated. (COZMO-5)

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
