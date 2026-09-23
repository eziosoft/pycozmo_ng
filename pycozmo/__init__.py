"""
PyCozmo - a pure-Python Cozmo robot communication library.

Main entry point:
    from pycozmo import Cozmo

    cozmo = Cozmo()
    with cozmo.connect():
        cozmo.vision.start()
        print(cozmo.sensors.battery_voltage)
        cozmo.navigator.drive_to(x=100, y=0)
"""

import sys

# ruff: noqa: I001 - logger must be imported first to avoid circular imports
from .logger import *

# Re-export moved modules so `from . import X` still works in root-level files.
# Order matters: some modules are imported eagerly by others during package init.
from .util import coords, filter, object, robot, robot_debug  # noqa: F401
from .util.robot import *

from . import (  # noqa: F401
    brain,
    exception,
    lights,
    protocol,
    sound,
    vision,
)
from .animation import (  # noqa: F401
    anim_controller,
)
from .client import Client
from .display import expressions, procedural_face  # noqa: F401
from .event import *
from .protocol import (  # noqa: F401
    protocol_base,
    protocol_declaration,
    protocol_encoder,
    protocol_generator,
    protocol_utils,
)
from .transport import conn, frame, window  # noqa: F401
from .vision import camera, image_encoder  # noqa: F401
from .cozmo import Cozmo

__version__ = "0.8.0"

__all__ = [
    "ROBOT_ADDR",
    "Client",
    "Cozmo",
    "Frame",
    "logger",
    "logger_protocol",
    "logger_robot",
    "setup_basic_logging",
]

del sys
