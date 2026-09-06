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

"""

import argparse
import os
import subprocess
import time

import cv2
import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import pycozmo
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
            capture_output=True, text=True, timeout=5, check=False,
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
            capture_output=True, text=True, timeout=30, check=False,
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
        self.currently_connected_cube_id = None

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.latest_raw_frame = image
        self.frame_count += 1

        # Push JPEG to web viewer for MJPEG streaming
        if image is not None:
            import io

            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=70)
            self.web_viewer.push_frame(buf.getvalue())

        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.latest_annotated_frame = annotated_image
        self.latest_detections = detections
        self.update_cube_lights_on_detection(cli)
        self.update_web_viewer()

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

        if current_cube_ids != self._prev_cube_ids:
            self.web_viewer.clear_cubes()
            self._prev_cube_ids = current_cube_ids

        for cube_id, cube_state in fused_cubes.items():
            cube_name = f"cube_{cube_id + 1}"
            self.web_viewer.update_cube(
                cube_name,
                cube_state["position"],
                cube_state["rotation_matrix"],
                cube_size=44,
            )

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
                f"Cubes: {len(self.latest_detections)}",
                f"Detection: {'ON' if self.detection_enabled else 'OFF'}",
            ]
            self.add_text_overlay(raw_labeled, info_lines, (10, height - 90))

            if self.latest_detections:
                det_lines = [f"Detected Cubes: {len(self.latest_detections)}"]
                for i, det in enumerate(self.latest_detections[:3]):
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

        print(f"   Detections: {len(self.latest_detections)}")

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
        print("⚠️  Note: Cozmo can only connect to ONE cube at a time")
        print("   Cubes will be connected on-demand when detected")
        print("=" * 70 + "\n")

    def update_cube_lights_on_detection(self, cli):
        """Update cube lights based on current detections."""
        if not self.detection_enabled:
            return

        if not self.cube_factory_id_map:
            return

        current_time = time.time()
        if not hasattr(self, "_last_light_update"):
            self._last_light_update = 0

        if current_time - self._last_light_update < 1.0:
            return

        self._last_light_update = current_time

        detected_cube_ids = {det.cube_id for det in self.latest_detections}

        if not hasattr(self, "_last_detected_ids"):
            self._last_detected_ids = set()

        if detected_cube_ids != self._last_detected_ids:
            print(f"🔍 Detected cube IDs: {sorted(detected_cube_ids)}")
            self._last_detected_ids = detected_cube_ids

        for cube_id in detected_cube_ids:
            if cube_id in self.cube_factory_id_map:
                factory_id = self.cube_factory_id_map[cube_id]

                if self.currently_connected_cube_id != cube_id:
                    try:
                        pkt = pycozmo.protocol_encoder.ObjectConnect(
                            factory_id=factory_id, connect=True
                        )
                        cli.conn.send(pkt)
                        time.sleep(0.1)
                        self.currently_connected_cube_id = cube_id
                    except Exception as e:
                        print(f"⚠️  Error connecting to cube {cube_id}: {e}")
                        continue

                connected_objects = dict(cli.connected_objects)
                if connected_objects:
                    object_id = next(iter(connected_objects.keys()))

                    try:
                        pkt = pycozmo.protocol_encoder.CubeId(object_id=object_id)
                        cli.conn.send(pkt)

                        pkt = pycozmo.protocol_encoder.CubeLights(
                            states=(
                                pycozmo.lights.blue_light,
                                pycozmo.lights.blue_light,
                                pycozmo.lights.blue_light,
                                pycozmo.lights.blue_light,
                            )
                        )
                        cli.conn.send(pkt)

                    except Exception as e:
                        print(f"⚠️  Error controlling cube {cube_id} lights: {e}")

    def on_robot_state(self, cli):
        """Handler for robot state updates."""
        if not hasattr(self, "latest_robot_state"):
            self.latest_robot_state = None

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

            if key == ord("i") or key == ord("I"):
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
    args = parser.parse_args()

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
                process_every_n_frames=1,
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

            print("🎥 Enabling camera...")
            cli.enable_camera(enable=True, color=False)
            cli.set_camera_params(exposure_ms=100)

            time.sleep(2.0)

            display.setup_cube_mapping(cli)

            print("📍 Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin set to current position")

            cli.add_handler(pycozmo.event.EvtRobotStateUpdated, display.on_robot_state)

            print("\n✨ Starting display...\n")
            print("🌐 Open http://<this-machine-ip>:8766 in a browser\n")

            display.run(cli)

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
