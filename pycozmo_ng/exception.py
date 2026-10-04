"""

Exception declarations.

"""

__all__ = [
    "ConnectionTimeout",
    "NoSpace",
    "PyCozmoNGConnectionError",
    "PyCozmoNGException",
    "Timeout",
]


class PyCozmoNGException(Exception):
    """Base class for all PyCozmo-NG exceptions."""


class PyCozmoNGConnectionError(PyCozmoNGException):
    """Base class for all PyCozmo-NG connection exceptions."""


class ConnectionTimeout(PyCozmoNGConnectionError):
    """Connection timeout."""


class Timeout(PyCozmoNGException):
    """Operation timed out."""


class NoSpace(PyCozmoNGException):
    """Out of space."""


class InvalidOperation(PyCozmoNGException):
    """Invalid operation."""


class ResourcesNotFound(PyCozmoNGException):
    """Cozmo resources not found."""
