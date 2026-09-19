#!/usr/bin/env python
"""Main test script for PyCozmo vision system.

Connects to Cozmo, initializes vision processor, and runs the headless loop.
All reusable logic lives in pycozmo; this file is only argparse + wiring.
The interface is the web viewer (http://<this-machine-ip>:8766).
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

            print("Positioning robot...")
            angle = (
                pycozmo.robot.MAX_HEAD_ANGLE.radians
                - pycozmo.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0
            cozmo.set_head_angle(angle.degrees)
            time.sleep(1.0)

            print("Initializing vision pipeline...")

            cozmo.setup_vision(
                floor_center_mm=(200.0, 0.0),
                floor_size_mm=(args.floor_size, args.floor_size),
                mm_per_pixel=args.floor_resolution,
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

            cozmo.enable_camera()

            # Robot center calibration
            cozmo.calibrate_robot_center(timeout_s=30.0)

            time.sleep(1.0)

            cozmo.discover_cubes()

            cozmo.enable_remote_control()

            cozmo.set_origin()

            cozmo.web_viewer.start()

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
