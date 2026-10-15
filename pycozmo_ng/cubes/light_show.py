"""CubeLightShow — runs a :class:`LightPattern` across all connected cubes.

The pattern decides *what* the LEDs show; this class decides *when* to send it.
Frames go only to cubes the manager currently holds connected, so a cube that
drops out is skipped until it comes back rather than erroring.
"""

import threading

from pycozmo_ng.cubes.light_pattern import LightPattern
from pycozmo_ng.logger import logger_cubes

__all__ = ["CubeLightShow"]


class CubeLightShow:
    """Drives a light pattern over the connected cubes in a background thread.

    Args:
        cubes: The :class:`~pycozmo_ng.cubes.manager.CubeManager` that owns the
            connections and knows which cubes are up.
        pattern: Pattern to play. Defaults to a rainbow.
        interval_s: Seconds between frames.
    """

    #: Default seconds between frames.
    DEFAULT_INTERVAL_S = 0.25

    def __init__(
        self,
        cubes,
        pattern: LightPattern | None = None,
        interval_s: float = DEFAULT_INTERVAL_S,
    ) -> None:
        self._cubes = cubes
        self._pattern = pattern if pattern is not None else LightPattern()
        self._interval_s = float(interval_s)
        self._frame_index = 0
        self._cli = None
        self._stop_event: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        """Whether the show thread is running."""
        return self._thread is not None

    @property
    def pattern(self) -> LightPattern:
        """The pattern being played."""
        return self._pattern

    def set_pattern(self, pattern: LightPattern) -> None:
        """Swap the pattern. Takes effect on the next frame."""
        self._pattern = pattern

    def start(self, cli) -> None:
        """Start playing. No-op if already running."""
        if self._thread is not None:
            return
        self._cli = cli
        self._frame_index = 0
        self._cubes.set_light_refresh(False)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="CubeLightShow", daemon=True)
        self._thread.start()
        logger_cubes.info(
            "Cube light show started (pattern=%s, %.2f s/frame)",
            self._pattern.name,
            self._interval_s,
        )

    def stop(self) -> None:
        """Stop playing. No-op if not running."""
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._stop_event = None
        self._cubes.set_light_refresh(True)
        logger_cubes.info("Cube light show stopped")

    def tick(self) -> None:
        """Send the current frame to the connected cubes and advance.

        Exposed separately from the thread so the sequence can be stepped
        without a robot in tests.
        """
        states = self._pattern.frame(self._frame_index)
        self._cubes.set_cube_states(self._cli, states)
        self._frame_index += 1

    def _loop(self) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001
                logger_cubes.debug("Light show frame failed", exc_info=True)
            if self._stop_event is not None and self._stop_event.wait(self._interval_s):
                return
