"""

Logging.

"""

import logging
import os
import sys

__all__ = [
    "logger",
    "logger_animation",
    "logger_behavior",
    "logger_cubes",
    "logger_protocol",
    "logger_reaction",
    "logger_robot",
    "setup_basic_logging",
]


# General logger - general PyCozmo log messages.
logger = logging.getLogger("pycozmo.general")
# Protocol logger - log messages related to the Cozmo protocol.
logger_protocol = logging.getLogger("pycozmo.protocol")
# Robot logger - log messages coming from the robot microcontrollers.
logger_robot = logging.getLogger("pycozmo.robot")
# Reaction logger.
logger_reaction = logging.getLogger("pycozmo.reaction")
# Behavior logger.
logger_behavior = logging.getLogger("pycozmo.behavior")
# Animation logger.
logger_animation = logging.getLogger("pycozmo.animation")
# Cube logger - cube-light worker diagnostics (connections, object_id cache).
logger_cubes = logging.getLogger("pycozmo.cubes")


def setup_basic_logging(
    log_level: str | None = None,
    protocol_log_level: str | None = None,
    robot_log_level: str | None = None,
    target=sys.stderr,
) -> None:
    """Configure basic logging for all pycozmo loggers.

    Replaces any previously installed pycozmo handler on each logger
    (marked with ``_pycozmo_basic``) so repeated calls do not stack
    duplicate handlers.

    Args:
        log_level: Log level for the general logger. Defaults to
            ``PYCOZMO_LOG_LEVEL`` env var or ``INFO``.
        protocol_log_level: Log level for the protocol logger. Defaults
            to ``PYCOZMO_PROTOCOL_LOG_LEVEL`` env var or ``INFO``.
        robot_log_level: Log level for robot/behavior/reaction/animation
            loggers. Defaults to ``PYCOZMO_ROBOT_LOG_LEVEL`` env var or
            ``WARNING`` (kept at WARNING due to
            "AnimationController.IsReadyToPlay.BufferStarved" messages).
    """
    if log_level is None:
        log_level = os.environ.get("PYCOZMO_LOG_LEVEL", logging.INFO)
    if protocol_log_level is None:
        protocol_log_level = os.environ.get("PYCOZMO_PROTOCOL_LOG_LEVEL", logging.INFO)
    if robot_log_level is None:
        # Keeping the default to WARNING due to "AnimationController.IsReadyToPlay.BufferStarved" messages.
        robot_log_level = os.environ.get("PYCOZMO_ROBOT_LOG_LEVEL", logging.WARNING)
    handler = logging.StreamHandler(stream=target)
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(name)-20s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    # Mark this handler so repeated setup calls replace ours instead of
    # stacking a second copy (which printed every record twice).
    handler._pycozmo_basic = True  # type: ignore[attr-defined]
    for lg in (
        logger,
        logger_protocol,
        logger_robot,
        logger_reaction,
        logger_behavior,
        logger_animation,
    ):
        lg.handlers = [
            h for h in lg.handlers if not getattr(h, "_pycozmo_basic", False)
        ]
        lg.addHandler(handler)
        # We install our own handler — don't also emit through the root
        # logger's handlers (e.g. an app's logging.basicConfig).
        lg.propagate = False
    logger.setLevel(log_level)
    logger_protocol.setLevel(protocol_log_level)
    logger_robot.setLevel(robot_log_level)
    logger_reaction.setLevel(robot_log_level)
    logger_behavior.setLevel(robot_log_level)
    logger_animation.setLevel(robot_log_level)


# TODO: See cozmo_resources/config/engine/console_filter_config.json
