#!/usr/bin/env python
"""Main test script for PyCozmo-NG vision system.

Connects to Cozmo, initializes vision processor, and runs the headless loop.
All reusable logic lives in pycozmo_ng; this file is only argparse + wiring.
The interface is the web viewer (http://<machine-ip>:8765) and, with --mcp,
an MCP endpoint for LLM clients (http://127.0.0.1:8767/mcp).

Progress is reported through the ``pycozmo_ng.demo`` logger (never ``print``),
as numbered startup steps so it is obvious what the script is doing — and
where it is stuck — while the robot comes up.
"""

import argparse
import logging
import time

from dotenv import load_dotenv

import pycozmo_ng
from pycozmo_ng import Cozmo
from pycozmo_ng.logger import logger_cubes, setup_basic_logging
from pycozmo_ng.transport.conn import ROBOT_ADDR
from pycozmo_ng.wifi import connect_to_cozmo_wifi, get_local_ip

logger = logging.getLogger("pycozmo_ng.demo")

# Number of startup phases reported by _step().
TOTAL_STEPS = 13

VIEWER_PORT = 8765


def _step(n: int, msg: str, *args: object) -> None:
    """Log a numbered startup step, e.g. ``[3/12] Starting web viewer...``.

    Extra arguments are lazily formatted into ``msg`` (printf-style), like
    regular ``logger`` calls.
    """
    logger.info("[%d/%d] " + msg, n, TOTAL_STEPS, *args)


def main():
    """Main entry point."""
    # Wi-Fi credentials (COZMO_WIFI_SSID / COZMO_WIFI_PASSWORD) come from
    # .env — the same file start.sh sources. Never hardcode them.
    load_dotenv()

    parser = argparse.ArgumentParser(description="PyCozmo-NG Vision System Test")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="WARNING",
        help=(
            "Root log level (library pycozmo_ng.* channels have their own "
            "defaults). INFO enables the cube-light worker diagnostics "
            "(cube connect/disconnect, per-tick action), DEBUG adds "
            "per-packet detail."
        ),
    )
    parser.add_argument(
        "--cubes-log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help=(
            "Override the log level specifically for the pycozmo_ng.cubes "
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
    parser.add_argument(
        "--mcp",
        nargs="?",
        const=8767,
        type=int,
        default=None,
        metavar="PORT",
        help=(
            "Serve an MCP endpoint for LLM clients (streamable HTTP, "
            "default port 8767, localhost only). Requires the optional "
            "extra: pip install 'pycozmo-ng[mcp]'."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_cubes.setLevel(getattr(logging, args.cubes_log_level))
    # Install the pycozmo_ng channel handlers now: Cozmo.connect() calls this
    # again with the same defaults (idempotent), but without the early call
    # everything logged before the connection — e.g. the step-1 Wi-Fi check —
    # would be dropped (pycozmo_ng.* channels have no handler yet).
    setup_basic_logging()
    # Demo progress is this script's UI — always show it at INFO, regardless
    # of --log-level (which tunes the root logger for third-party output).
    logger.setLevel(logging.INFO)

    t_start = time.monotonic()
    cozmo = Cozmo()

    try:
        _step(1, "Checking Cozmo Wi-Fi AP connection...")
        connect_to_cozmo_wifi()

        _step(2, "Connecting to robot at %s:%d...", ROBOT_ADDR[0], ROBOT_ADDR[1])
        with cozmo.connect():
            logger.info("Robot connected - protocol handshake complete")

            _step(
                3,
                "Starting web viewer (HTTP + WebSocket, port %d)...",
                VIEWER_PORT,
            )
            cozmo.web_viewer.start()
            time.sleep(1.0)

            _step(4, "Positioning head camera to startup tilt...")
            angle = (
                pycozmo_ng.robot.MAX_HEAD_ANGLE.radians
                - pycozmo_ng.robot.MIN_HEAD_ANGLE.radians
            ) / 80.0
            cozmo.cli.set_head_angle(angle)
            time.sleep(1.0)

            _step(
                5,
                "Initializing vision pipeline (floor %.0fx%.0f mm @ %.1f mm/px, "
                "face detection every %d frame%s, overlay shows confidence + cube ID)...",
                args.floor_size,
                args.floor_size,
                args.floor_resolution,
                args.face_detect_every,
                "" if args.face_detect_every == 1 else "s",
            )
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
                _step(
                    6,
                    "Starting cube keepalive worker "
                    "(BLE cycling + CubeLights refresh)...",
                )
                cozmo.start_cube_lights()
            else:
                _step(6, "Skipping cube keepalive worker (--no-cube-lights)")
                logger.warning(
                    "Cubes will disconnect after the robot's idle timeout "
                    "and will not be reconnected."
                )

            _step(7, "Enabling camera (grayscale, 30 ms exposure)...")
            cozmo.enable_camera(exposure_ms=30)

            _step(8, "Measuring camera lateral offset via robot-center markers...")
            cozmo.calibrate_robot_center(timeout_s=30.0)
            time.sleep(1.0)

            _step(9, "Discovering cubes and building cube-ID mapping...")
            cozmo.discover_cubes()

            _step(10, "Enabling browser keyboard remote control...")
            cozmo.enable_remote_control()

            _step(11, 'Cozmo greets: "Hello, I am Cozmo"')
            cozmo.say("Hello, I am Cozmo")

            _step(12, "Setting localization origin at the current robot pose...")
            cozmo.set_origin()

            if args.mcp is not None:
                _step(13, "Starting MCP endpoint (streamable HTTP, port %d)...", args.mcp)
                cozmo.mcp.start(port=args.mcp)
                logger.info("MCP endpoint for LLM clients: http://127.0.0.1:%d/mcp", args.mcp)
            else:
                _step(13, "Skipping MCP endpoint (--mcp not given)")

            logger.info("READY - startup took %.1f s", time.monotonic() - t_start)
            logger.info(
                "Open the web viewer in a browser: http://%s:%d",
                get_local_ip(),
                VIEWER_PORT,
            )
            logger.info("Press Ctrl+C to stop.")

            cozmo.run()
            cozmo.mcp.stop()

            logger.info("Run loop ended - disconnected from robot")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except (OSError, ValueError, RuntimeError):
        logger.exception("Fatal error")
    finally:
        logger.info("Goodbye")


if __name__ == "__main__":
    main()
