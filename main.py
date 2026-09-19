#!/usr/bin/env python
"""Main test script for PyCozmo vision system.

Connects to Cozmo, initializes vision processor, and runs the headless loop.
All reusable logic lives in pycozmo.brain; this file is only argparse + wiring.
The interface is the web viewer (http://<this-machine-ip>:8766).

Detection throttle:
  python main.py --process-every-n-frames 2   # 15 fps detection, 30 fps preview
  python main.py --process-every-n-frames 3   # 10 fps detection
"""

import argparse
import logging
import time

import pycozmo
from pycozmo.brain import Brain
from pycozmo.logger import logger_cubes
from pycozmo.vision.camera_calibration import CameraCalibrationRetriever
from pycozmo.wifi import connect_to_cozmo_wifi


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
            "as the brain runs."
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
            "detector runs at any of these sizes - markers are still "
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
    parser.add_argument(
        "--no-floor-projection",
        action="store_true",
        help=(
            "Disable the floor-projected (bird's-eye) camera view. "
            "Enabled by default — projects the camera image onto the "
            "floor plane for occupancy mapping."
        ),
    )
    parser.add_argument(
        "--no-face-detection",
        action="store_true",
        help=(
            "Disable Haar cascade face detection. "
            "Enabled by default — detects faces and renders them as "
            "orange globes in the 3D world viewer."
        ),
    )
    parser.add_argument(
        "--floor-size",
        type=float,
        default=600.0,
        help="Floor grid size in mm (default: 600, centred 200 mm ahead).",
    )
    parser.add_argument(
        "--floor-resolution",
        type=float,
        default=5.0,
        help="Floor grid resolution in mm/pixel (default: 5).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_cubes.setLevel(getattr(logging, args.cubes_log_level))

    brain = Brain()

    try:
        connect_to_cozmo_wifi()

        print("Connecting to Cozmo...")
        with pycozmo.connect() as cli:
            print("Connected!\n")

            print("Positioning robot...")
            angle = (
                pycozmo.robot.MAX_HEAD_ANGLE.radians
                - pycozmo.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0

            cli.set_head_angle(angle)
            time.sleep(1.0)

            print("Retrieving camera calibration from Cozmo...")
            calibration_retriever = CameraCalibrationRetriever(cli)
            camera_calibration = calibration_retriever.get_calibration(timeout=10.0)

            print("Camera calibration retrieved:")
            print(f"   fx={camera_calibration.fx:.2f}, fy={camera_calibration.fy:.2f}")
            print(f"   cx={camera_calibration.cx:.2f}, cy={camera_calibration.cy:.2f}")
            print(
                f"   k1={camera_calibration.k1:.6f}, k2={camera_calibration.k2:.6f}, k3={camera_calibration.k3:.6f}"
            )
            print(f"   p1={camera_calibration.p1:.6f}, p2={camera_calibration.p2:.6f}")
            print(
                f"   Image size: {camera_calibration.image_width}x{camera_calibration.image_height}"
            )
            print("   Using zero distortion coefficients for Cozmo reliability")

            brain.cli = cli
            cli.add_handler(
                pycozmo.event.EvtNewRawCameraImage, brain.vision.on_raw_camera_image
            )
            cli.add_handler(
                pycozmo.event.EvtRobotOnChargerChange,
                brain.navigator._on_charger_change,
            )
            cli.add_handler(
                pycozmo.event.EvtRobotChargingChange,
                brain.navigator._on_charging_change,
            )

            print("Initializing vision pipeline with cubeDetector.py...")
            undistort_enabled = not args.no_undistort
            print(
                f"   Undistortion: {'ENABLED' if undistort_enabled else 'DISABLED (default - undistortion may cause warping)'}"
            )
            floor_enabled = not args.no_floor_projection
            face_enabled = not args.no_face_detection
            brain.setup_vision(
                cli,
                camera_calibration=camera_calibration,
                process_every_n_frames=max(1, args.process_every_n_frames),
                undistort=undistort_enabled,
                enable_floor_projection=floor_enabled,
                floor_center_mm=(200.0, 0.0),
                floor_size_mm=(args.floor_size, args.floor_size),
                mm_per_pixel=args.floor_resolution,
                enable_face_detection=face_enabled,
            )

            print("Cube detector initialized with camera calibration")
            if face_enabled:
                print("Face detection ENABLED (orange globes in 3D viewer)")

            # Configure annotation engine
            brain.annotation_engine.show_confidence = True
            brain.annotation_engine.show_cube_id = True
            brain.annotation_engine.box_thickness = 2

            cli.add_handler(
                pycozmo.event.EvtAnnotatedCameraImage,
                brain.vision.on_annotated_camera_image,
            )

            if floor_enabled:
                cli.add_handler(
                    pycozmo.event.EvtFloorProjectedImage,
                    brain.vision.on_floor_projected_image,
                )
                print("Floor projection ENABLED")

            cli.add_handler(
                pycozmo.protocol_encoder.ObjectConnectionState,
                brain.cubes.on_object_connection_state,
            )

            if not args.no_cube_lights:
                brain.cubes.start_cube_light_worker(cli)
            else:
                print(
                    "Cube keepalive worker disabled (--no-cube-lights)."
                    "\n   Cubes will disconnect after idle timeout."
                )

            if not args.no_camera:
                resolution = getattr(
                    pycozmo.protocol_encoder.ImageResolution, args.camera_resolution
                )
                if args.camera_fps > 0:
                    print(
                        f"   --camera-fps {args.camera_fps} is experimental "
                        "and currently makes the Wi-Fi drop FASTER. Use --no-camera "
                        "instead. (COZMO-20)"
                    )

                print("Camera is ON by default.")
                print(
                    "   It stays on for the whole session (not tied to the web viewer)."
                )
                print(f"   Resolution: {args.camera_resolution}")
                brain.enable_camera_persistent(cli, resolution)

                # Robot center calibration: enable camera, tilt head, detect
                print("Calibrating robot center (head to max-down)...")
                # Temporarily enable camera for calibration with manual
                # exposure — auto-exposure makes the image too dark when
                # head is at max-down (robot body fills the frame).
                cli.enable_camera(
                    enable=True,
                    color=False,
                    resolution=resolution,
                )
                cli.set_camera_params(
                    gain=8.0,
                    exposure_ms=300,
                    auto_exposure_enabled=False,
                )
                time.sleep(2.0)  # Let camera warm up

                cli.set_head_angle(
                    pycozmo.robot.MIN_HEAD_ANGLE.radians,
                    accel=30.0,
                    max_speed=30.0,
                )
                time.sleep(3.0)  # Let head settle and camera see circles

                # Collect multiple detections and average them
                brain.detection_pipeline._robot_center_measured = False
                offsets = []
                for _ in range(300):  # up to 30s
                    time.sleep(0.1)
                    rc = brain.detection_pipeline.robot_center_result
                    if rc is not None and rc.detected:
                        offsets.append(rc.lateral_offset_mm)
                        if len(offsets) >= 10:
                            break

                # Return head to normal position and restore auto-exposure
                cli.set_head_angle(0.0, accel=30.0, max_speed=30.0)
                time.sleep(1.0)
                brain.post_calibration(cli, resolution)

                if offsets:
                    avg_offset = sum(offsets) / len(offsets)
                    brain.detection_pipeline.camera_lateral_offset_mm = avg_offset
                    brain.detection_pipeline._robot_center_measured = True
                    print(
                        f"   Robot center detected: "
                        f"lateral_offset={avg_offset:+.2f}mm "
                        f"(avg of {len(offsets)} frames)"
                    )
                else:
                    print("   WARNING: Robot center NOT detected (circles not found)")

                time.sleep(1.0)

                brain.cubes.setup_cube_mapping(cli)

                brain.install_remote_control(cli)
            else:
                print("Camera disabled (--no-camera).")
                print("   No image streaming, no detection, no display.")
                print("   Keeping the connection alive for diagnostics only.")

            print("Setting localization origin...")
            pkt = pycozmo.protocol_encoder.SetOrigin()
            cli.conn.send(pkt)
            print("Origin set to current position")

            cli.add_handler(
                pycozmo.event.EvtRobotStateUpdated, brain.web._on_robot_state_for_trail
            )

            cli.add_handler(pycozmo.event.EvtRobotPickedUpChange, brain._on_picked_up)

            if args.no_camera:
                print(
                    "--no-camera: skipping event loop. "
                    "Connection will stay idle. Press Ctrl-C to quit."
                )
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    pass
            else:
                print("\nStarting headless run...\n")
                print("Open http://<this-machine-ip>:8766 in a browser\n")

                brain.run(cli)
                brain.cubes.stop_cube_light_worker()

                print("\nRun closed")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except (OSError, ValueError, RuntimeError) as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nGoodbye!\n")


if __name__ == "__main__":
    main()
