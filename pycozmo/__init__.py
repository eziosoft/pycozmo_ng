"""

PyCozmo - a pure-Python Cozmo robot communication library.

"""

import sys

from .logger import *

from . import protocol  # noqa: I001,F401
from .protocol import (  # noqa: I001,F401
    protocol_base,
    protocol_declaration,
    protocol_encoder,
    protocol_generator,
    protocol_utils,
)

# Re-export moved modules so `from . import X` still works in root-level files.
# Order matters: some modules are imported eagerly by others during package init.
from .util import coords, filter, object, robot, robot_debug  # noqa: F401
from .transport import conn, frame, window  # noqa: F401
from .animation import anim_controller, anim_encoder  # noqa: F401
from .animation import activity, anim, behavior  # noqa: F401

from .run import *

from . import (  # noqa: F401
    sound,
    brain,
    exception,
    image_encoder,
    lights,
    vision,
)
from .client import Client  # noqa: F401
from .display import emotions, expressions, procedural_face  # noqa: F401
from .event import *  # noqa: F401
from .util.robot import *  # noqa: F401
from .vision import camera  # noqa: F401

__version__ = "0.8.0"

__all__ = [
    "ROBOT_ADDR",
    "Client",
    "Frame",
    "connect",
    "logger",
    "logger_protocol",
    "logger_robot",
    "setup_basic_logging",
]

if sys.version_info < (3, 6, 0):
    sys.exit("ERROR: PyCozmo requires Python 3.6.0 or newer.")
del sys
