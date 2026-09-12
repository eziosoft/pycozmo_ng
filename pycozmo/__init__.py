"""

PyCozmo - a pure-Python Cozmo robot communication library.

"""

import sys

from .logger import *

from . import protocol
from .protocol import protocol_base
from .protocol import protocol_declaration
from .protocol import protocol_generator
from .protocol import protocol_encoder
from .protocol import protocol_utils

from .run import *

from .frame import Frame
from .conn import ROBOT_ADDR
from .client import Client
from .robot import *
from .event import *

from . import exception
from . import util
from . import window
from . import lights
from .vision import camera
from . import object
from . import filter
from . import anim
from . import anim_encoder
from . import image_encoder
from .display import procedural_face
from . import activity
from . import behavior
from .display import emotions
from . import brain
from . import audiokinetic
from .display import expressions
from . import vision


__version__ = "0.8.0"

__all__ = [
    "logger",
    "logger_protocol",
    "logger_robot",

    "Frame",
    "ROBOT_ADDR",
    "Client",

    "setup_basic_logging",
    "connect",
]

if sys.version_info < (3, 6, 0):
    sys.exit("ERROR: PyCozmo requires Python 3.6.0 or newer.")
del sys
