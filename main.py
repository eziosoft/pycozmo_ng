#!/usr/bin/env python
"""Main test script for PyCozmo vision system.

Connects to Cozmo, initializes vision processor, and runs the headless loop.
All reusable logic lives in pycozmo; this file is only argparse + wiring.
The interface is the web viewer (http://<this-machine-ip>:8765).
"""

import argparse
import logging
import time

import pycozmo
from pycozmo import Cozmo
from pycozmo.logger import logger_cubes
from pycozmo.wifi import connect_to_cozmo_wifi


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(description="PyCozmo Vision System Test")
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
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=12.0,
        metavar="N",
        help=(
            "Camera frame-rate cap (SingleShot mode, default: 12). Cozmo's "
            "native Stream mode (~30 fps) saturates the lossy Cozmo AP and "
            "drops the link within a minute; capping the rate keeps the "
            "vision feed smooth without starving the control channel."
        ),
    )
    parser.add_argument(
        "--face-detect-every",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Run face detection every Nth processed frame (default: 3). "
            "Higher values reduce CPU load at the cost of slower face "
            "tracking. Set to 1 to detect every frame."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_cubes.setLevel(getattr(logging, args.cubes_log_level))

    cozmo = Cozmo()

    try:
        connect_to_cozmo_wifi()

        print("Connecting to Cozmo...")
        with cozmo.connect():
            print("Connected!\n")

            print("Starting web viewer...")
            cozmo.web_viewer.start()
            time.sleep(1.0)

            print("Positioning robot...")
            angle = (
                pycozmo.robot.MAX_HEAD_ANGLE.radians
                - pycozmo.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0
            cozmo.cli.set_head_angle(angle)
            time.sleep(1.0)

            print("Initializing vision pipeline...")

            cozmo.setup_vision(
                floor_center_mm=(200.0, 0.0),
                floor_size_mm=(args.floor_size, args.floor_size),
                mm_per_pixel=args.floor_resolution,
                face_detect_every_n_frames=args.face_detect_every,
            )

            cozmo.set_vision_annotation_options(
                show_confidence=True, show_cube_id=True, box_thickness=2
            )

            if not args.no_cube_lights:
                cozmo.start_cube_lights()
            else:
                print(
                    "Cube keepalive worker disabled (--no-cube-lights)."
                    "\n   Cubes will disconnect after idle timeout."
                )

            cozmo.enable_camera(camera_fps=args.camera_fps)
            cozmo.cli.set_camera_params(exposure_ms=30)

            # Robot center calibration
            cozmo.calibrate_robot_center(timeout_s=30.0)

            time.sleep(1.0)

            cozmo.discover_cubes()

            cozmo.enable_remote_control()

            cozmo.say("Hello, I am Cozmo")

            cozmo.set_origin()

            print("Open http://<this-machine-ip>:8765 in a browser\n")

            cozmo.run()

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
