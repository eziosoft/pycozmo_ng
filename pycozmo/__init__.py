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

from .run import *

from . import (  # noqa: F401
    activity,
    anim,
    anim_encoder,
    audiokinetic,
    behavior,
    brain,
    exception,
    filter,
    image_encoder,
    lights,
    object,
    util,
    vision,
    window,
)
from .client import Client  # noqa: F401
from .conn import ROBOT_ADDR  # noqa: F401
from .display import emotions, expressions, procedural_face  # noqa: F401
from .event import *  # noqa: F401
from .frame import Frame  # noqa: F401
from .robot import *  # noqa: F401
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
