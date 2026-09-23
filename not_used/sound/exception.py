"""

AudioKinetic WWise and TTS exceptions.

"""

import pycozmo

__all__ = [
    "AudioKineticBaseError",
    "AudioKineticFormatError",
    "AudioKineticIOError",
    "TTSGenerationError",
    "TTSNotInstalledError",
]


class AudioKineticBaseError(pycozmo.exception.PyCozmoException):
    """AudioKinetic WWise base error."""


class AudioKineticFormatError(AudioKineticBaseError):
    """Invalid file format error."""


class AudioKineticIOError(AudioKineticBaseError):
    """File I/O error."""


class TTSNotInstalledError(pycozmo.exception.PyCozmoException):
    """Raised when pyttsx3 is not installed."""


class TTSGenerationError(pycozmo.exception.PyCozmoException):
    """Raised when TTS synthesis fails."""
