"""

AudioKinetic WWise exceptions.

"""

import pycozmo

__all__ = [
    "AudioKineticBaseError",
    "AudioKineticFormatError",
    "AudioKineticIOError",
]


class AudioKineticBaseError(pycozmo.exception.PyCozmoException):
    """AudioKinetic WWise base error."""


class AudioKineticFormatError(AudioKineticBaseError):
    """Invalid file format error."""


class AudioKineticIOError(AudioKineticBaseError):
    """File I/O error."""
