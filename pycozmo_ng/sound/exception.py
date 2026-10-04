"""

AudioKinetic WWise and TTS exceptions.

"""

import pycozmo_ng

__all__ = [
    "AudioKineticBaseError",
    "AudioKineticFormatError",
    "AudioKineticIOError",
    "TTSGenerationError",
    "TTSNotInstalledError",
]


class AudioKineticBaseError(pycozmo_ng.exception.PyCozmoNGException):
    """AudioKinetic WWise base error."""


class AudioKineticFormatError(AudioKineticBaseError):
    """Invalid file format error."""


class AudioKineticIOError(AudioKineticBaseError):
    """File I/O error."""


class TTSNotInstalledError(pycozmo_ng.exception.PyCozmoNGException):
    """Raised when pyttsx3 is not installed."""


class TTSGenerationError(pycozmo_ng.exception.PyCozmoNGException):
    """Raised when TTS synthesis fails."""
