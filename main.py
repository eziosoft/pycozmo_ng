#!/usr/bin/env python
"""

Main test script for PyCozmo vision system.

Displays both normal and annotated camera frames side-by-side using OpenCV.
Includes 3D viewer for cube tracking.

Controls:
- Press 'q' to quit
- Press 's' to save current frames
- Press 'c' to toggle cube detection
- Press 'f' to toggle FPS display
- Press '3' to toggle 3D viewer

"""

import argparse
import os
import subprocess
import sys
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

# Import 3D viewer
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), "pycozmo", "cubeDetector"))
    from world_3d_viewer import World3DViewer

    VIEWER_3D_AVAILABLE = True
except ImportError:
    VIEWER_3D_AVAILABLE = False
    print("⚠️  3D Viewer not available")

# Import web viewer server
try:
    from pycozmo.web_viewer_server import WebViewerServer

    WEB_VIEWER_AVAILABLE = True
except ImportError:
    WEB_VIEWER_AVAILABLE = False


def connect_to_cozmo_wifi():
    """Auto-connect to Cozmo WiFi AP if not already connected."""
    ssid = os.environ.get("COZMO_WIFI_SSID")
    password = os.environ.get("COZMO_WIFI_PASSWORD")
    if not ssid:
        return True  # No credentials configured, skip

    # Check current WiFi
    try:
        result = subprocess.run(
            ["iwgetid", "-r"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        current_ssid = result.stdout.strip()
        if current_ssid == ssid:
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
    """Handles display of normal and annotated frames."""

    def __init__(self, use_3d_viewer=True, use_web_viewer=False):
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

        # 3D viewer setup
        self.use_3d_viewer = (
            use_3d_viewer and VIEWER_3D_AVAILABLE and not use_web_viewer
        )
        self.viewer_3d = None
        self.show_3d = True

        # Web viewer setup
        self.use_web_viewer = use_web_viewer and WEB_VIEWER_AVAILABLE
        self.web_viewer = None

        if self.use_3d_viewer:
            try:
                self.viewer_3d = World3DViewer(window_size=(800, 600))
                print("✅ 3D Viewer initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize 3D viewer: {e}")
                self.use_3d_viewer = False

        if self.use_web_viewer:
            try:
                self.web_viewer = WebViewerServer()
                self.web_viewer.start()
                print("✅ Web Viewer server started")
            except Exception as e:
                print(f"⚠️  Failed to start web viewer: {e}")
                self.use_web_viewer = False

        # Window names
        self.window_name = "PyCozmo Vision Test - Normal (Left) | Annotated (Right)"

        # Cube tracking variables
        self.cube_factory_id_map = {}  # Map detected cube_id (1,2,3) to factory_id
        self.currently_connected_cube_id = (
            None  # Track which cube is currently connected
        )

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.latest_raw_frame = image
        self.frame_count += 1

        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.latest_annotated_frame = annotated_image
        self.latest_detections = detections

        # Update cube lights based on detections
        self.update_cube_lights_on_detection(cli)

        # Update 3D viewer if enabled
        if self.use_3d_viewer and self.show_3d and self.viewer_3d:
            self.update_3d_viewer()

    def on_debug_images(self, cli, debug_images, detections):
        """Handler for debug images from cube detection pipeline."""
        self.latest_debug_images = debug_images

    def update_3d_viewer(self):
        """Update 3D viewer (local or web) with current cube states."""
        if not self.vision_processor:
            return

        # Get fused cube states from vision processor
        fused_cubes = self.vision_processor.get_fused_cube_states()

        # Track which cubes are present
        if not hasattr(self, "_prev_cube_ids"):
            self._prev_cube_ids = set()

        current_cube_ids = set(fused_cubes.keys())

        # Determine active viewer
        active_viewer = None
        if self.use_web_viewer and self.web_viewer:
            active_viewer = self.web_viewer
        elif self.use_3d_viewer and self.show_3d and self.viewer_3d:
            active_viewer = self.viewer_3d

        if active_viewer is None:
            return

        # Only clear if the set of cubes changed (expensive operation)
        if current_cube_ids != self._prev_cube_ids:
            try:
                active_viewer.clear_cubes()
                self._prev_cube_ids = current_cube_ids
            except Exception as e:
                print(f"⚠️  Error clearing 3D viewer: {e}")
                return

        # Update cube poses (cheap operation)
        try:
            for cube_id, cube_state in fused_cubes.items():
                cube_name = f"cube_{cube_id + 1}"
                active_viewer.update_cube(
                    cube_name,
                    cube_state["position"],
                    cube_state["rotation_matrix"],
                    cube_size=44,
                )
            # Broadcast to browser clients if web viewer
            if self.use_web_viewer:
                active_viewer.broadcast()
        except Exception as e:
            print(f"⚠️  Error updating 3D viewer: {e}")

    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format."""
        if pil_image is None:
            return None
        # Convert PIL RGB to OpenCV BGR
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def add_text_overlay(self, frame, text_lines, position=(10, 30)):
        """Add text overlay to frame."""
        x, y = position
        for i, line in enumerate(text_lines):
            y_pos = y + i * 25
            # Draw text background
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
            # Draw text
            cv2.putText(
                frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    def display_frames(self):
        """Display both raw and annotated frames side by side."""
        # Convert PIL images to OpenCV format
        raw_cv2 = self.pil_to_cv2(self.latest_raw_frame)
        annotated_cv2 = self.pil_to_cv2(self.latest_annotated_frame)

        # If no frames yet, show waiting message
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

        # Use raw frame for both if no annotated frame yet
        if annotated_cv2 is None:
            annotated_cv2 = raw_cv2.copy()

        # Ensure both frames are the same size
        height, width = raw_cv2.shape[:2]
        if annotated_cv2.shape[:2] != (height, width):
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

        # Add labels to frames
        raw_labeled = raw_cv2.copy()
        annotated_labeled = annotated_cv2.copy()

        # Add "Normal" label to raw frame
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

        # Add "Annotated" label to annotated frame
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

        # Add FPS and detection info
        if self.show_fps:
            info_lines = [
                f"FPS: {self.fps:.1f}",
                f"Cubes: {len(self.latest_detections)}",
                f"Detection: {'ON' if self.detection_enabled else 'OFF'}",
            ]
            self.add_text_overlay(raw_labeled, info_lines, (10, height - 90))

            # Add detection details on annotated frame
            if self.latest_detections:
                det_lines = [f"Detected Cubes: {len(self.latest_detections)}"]
                for i, det in enumerate(self.latest_detections[:3]):  # Show max 3
                    det_lines.append(f"  ID:{det.cube_id} Conf:{det.confidence:.2f}")
                self.add_text_overlay(annotated_labeled, det_lines, (10, height - 90))

        # Combine frames side by side
        combined = np.hstack([raw_labeled, annotated_labeled])

        # Display
        cv2.imshow(self.window_name, combined)

        # Display debug images
        for name, img in self.latest_debug_images.items():
            cv2.imshow(f"Debug: {name}", img)

    def save_frames(self):
        """Save current frames to disk."""
        if self.latest_raw_frame is None:
            print("⚠️  No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save raw frame
        raw_filename = f"frame_raw_{timestamp}.png"
        self.latest_raw_frame.save(raw_filename)
        print(f"💾 Saved raw frame: {raw_filename}")

        # Save annotated frame
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

        # Wait for objects to be discovered
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

        # Find all light cubes
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
        # Map detected cube_id to factory_id based on ObjectType
        # The cube detector assigns:
        # cube_id 1 (markers 0-5)   -> Block_LIGHTCUBE1
        # cube_id 2 (markers 6-11)  -> Block_LIGHTCUBE2
        # cube_id 3 (markers 12-17) -> Block_LIGHTCUBE3
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

        # If no mapping exists, we can't control lights
        if not self.cube_factory_id_map:
            return

        # Throttle updates to once per second to avoid slowing down video
        current_time = time.time()
        if not hasattr(self, "_last_light_update"):
            self._last_light_update = 0

        if current_time - self._last_light_update < 1.0:
            return  # Skip update if less than 1 second has passed

        self._last_light_update = current_time

        # Get currently detected cube IDs
        detected_cube_ids = {det.cube_id for det in self.latest_detections}

        # Only print when detections change
        if not hasattr(self, "_last_detected_ids"):
            self._last_detected_ids = set()

        if detected_cube_ids != self._last_detected_ids:
            print(f"🔍 Detected cube IDs: {sorted(detected_cube_ids)}")
            self._last_detected_ids = detected_cube_ids

        # Light up detected cubes (connect and set blue)
        for cube_id in detected_cube_ids:
            if cube_id in self.cube_factory_id_map:
                factory_id = self.cube_factory_id_map[cube_id]

                # Connect to this cube if not already connected
                if self.currently_connected_cube_id != cube_id:
                    try:
                        pkt = pycozmo.protocol_encoder.ObjectConnect(
                            factory_id=factory_id, connect=True
                        )
                        cli.conn.send(pkt)
                        time.sleep(0.1)  # Brief wait for connection
                        self.currently_connected_cube_id = cube_id
                    except Exception as e:
                        print(f"⚠️  Error connecting to cube {cube_id}: {e}")
                        continue

                # Get the object_id from connected_objects
                connected_objects = dict(cli.connected_objects)
                if connected_objects:
                    object_id = next(
                        iter(connected_objects.keys())
                    )  # Should only be one

                    try:
                        # Select cube
                        pkt = pycozmo.protocol_encoder.CubeId(object_id=object_id)
                        cli.conn.send(pkt)

                        # Set blue lights
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

    def on_robot_delocalized(self, cli):
        """Handler for robot delocalization events."""
        print("🚫 Robot delocalized! Resetting origin...")
        try:
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin reset to current position")
        except Exception as e:
            print(f"⚠️  Error resetting origin: {e}")

    def on_robot_state(self, cli):
        """Handler for robot state updates."""
        # Robot state is available in cli.pose, cli.pose_pitch, etc.
        # State updates arrive every 30ms (~33Hz)
        # For now, just store for potential use (could update 3D viewer robot position)
        if not hasattr(self, "latest_robot_state"):
            self.latest_robot_state = None

        # Access state from cli object if needed:
        # - cli.pose (position and orientation)
        # - cli.head_angle, cli.lift_position
        # - cli.left_wheel_speed, cli.right_wheel_speed
        # - cli.battery_voltage, cli.accel, cli.gyro

    def run(self, cli):
        """Main display loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while self.running:
            self.display_frames()

            # Render 3D viewer
            if (
                self.use_3d_viewer
                and self.show_3d
                and self.viewer_3d
                and not self.viewer_3d.render()
            ):
                print("\n🛑 3D viewer closed")
                self.running = False

            # Handle keyboard input for robot control
            key = cv2.waitKey(1) & 0xFF

            if key == ord("i") or key == ord("I"):
                # Forward
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=50, rwheel_speed_mmps=50
                )
                cli.conn.send(pkt)
            elif key == ord("k") or key == ord("K"):
                # Backward
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=-50, rwheel_speed_mmps=-50
                )
                cli.conn.send(pkt)
            elif key == ord("j") or key == ord("J"):
                # Left turn
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=-50, rwheel_speed_mmps=50
                )
                cli.conn.send(pkt)
            elif key == ord("l") or key == ord("L"):
                # Right turn
                pkt = pycozmo.protocol_encoder.DriveWheels(
                    lwheel_speed_mmps=50, rwheel_speed_mmps=-50
                )
                cli.conn.send(pkt)
            elif key == ord(" "):
                # Stop
                pkt = pycozmo.protocol_encoder.StopAllMotors()
                cli.conn.send(pkt)

        cv2.destroyAllWindows()
        if self.viewer_3d:
            self.viewer_3d.close()
        if self.web_viewer:
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
        "--web-viewer",
        action="store_true",
        help="Use browser-based Three.js viewer instead of local 3D viewer",
    )
    args = parser.parse_args()

    # Create display handler with 3D or web viewer
    display = VisionDisplay(
        use_3d_viewer=VIEWER_3D_AVAILABLE, use_web_viewer=args.web_viewer
    )

    try:
        # Auto-connect to Cozmo WiFi if credentials in .env
        connect_to_cozmo_wifi()

        # Connect to Cozmo
        print("🤖 Connecting to Cozmo...")
        with pycozmo.connect() as cli:
            print("✅ Connected!\n")

            # Position robot for better viewing
            print("📷 Positioning robot...")
            angle = (
                pycozmo.robot.MAX_HEAD_ANGLE.radians
                - pycozmo.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0

            cli.set_head_angle(angle)
            time.sleep(1.0)

            # Retrieve camera calibration from Cozmo
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

            # Register handler for raw frames
            cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage, display.on_raw_camera_image
            )

            # Create vision processor with cubeDetector.py
            print("🎨 Initializing vision processor with cubeDetector.py...")
            undistort_enabled = not args.no_undistort
            print(
                f"   Undistortion: {'ENABLED' if undistort_enabled else 'DISABLED (default - undistortion may cause warping)'}"
            )
            vision_processor = vision.AnnotatedVisionProcessor(
                client=cli,
                camera_calibration=camera_calibration,
                process_every_n_frames=1,  # Process every frame (fast computer)
                undistort=undistort_enabled,
            )

            print("✅ Cube detector initialized with camera calibration")

            # Store reference for 3D viewer updates
            display.vision_processor = vision_processor

            # Configure annotation display
            vision_processor.show_confidence = True
            vision_processor.show_cube_id = True
            vision_processor.box_thickness = 2

            # Register handler for annotated frames
            cli.add_handler(
                pycozmo.event.EvtAnnotatedCameraImage, display.on_annotated_camera_image
            )

            # Register handler for debug images
            cli.add_handler(
                pycozmo.event.EvtCubeDetectorDebugImages, display.on_debug_images
            )

            # Enable camera
            print("🎥 Enabling camera...")
            cli.enable_camera(enable=True, color=False)
            # Set camera params for auto exposure
            cli.set_camera_params(exposure_ms=100)

            # Wait for camera to stabilize
            time.sleep(2.0)

            # Discover cubes and setup mapping
            display.setup_cube_mapping(cli)

            # Set origin for localization
            print("📍 Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("✅ Origin set to current position")

            # Register handler for robot state updates (position/orientation every 30ms)
            # Note: EvtDelocalized not available in this pycozmo version
            cli.add_handler(pycozmo.event.EvtRobotStateUpdated, display.on_robot_state)

            print("✨ Starting display...\n")
            if display.use_web_viewer:
                print(
                    "🌐 Web Viewer active — open http://<this-machine-ip>:8766 in a browser\n"
                )
            elif display.use_3d_viewer:
                print("🎮 3D Viewer Controls:")
                print("   W/S/A/D - Move camera")
                print("   Q/E - Move camera up/down")
                print("   Arrow keys - Rotate camera\n")

            # Run display loop
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
