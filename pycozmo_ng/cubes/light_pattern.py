"""Cube LED patterns — pure frame generation, no robot required.

A :class:`LightPattern` maps a frame index to the LED state of every cube: four
:class:`~pycozmo_ng.protocol.protocol_encoder.LightState` values per cube, one per
top corner. Nothing here touches the robot, so patterns are unit-testable; the
runner that sends frames lives in :mod:`pycozmo_ng.cubes.light_show`.
"""

from collections.abc import Callable

from pycozmo_ng.lights import Color
from pycozmo_ng.protocol.protocol_encoder import LightState

__all__ = [
    "CORNERS",
    "PATTERN_NAMES",
    "LightPattern",
]

#: Corners on the top face of a light cube.
CORNERS = 4

#: Built-in pattern names.
PATTERN_NAMES = ("rainbow", "chase", "pulse", "solid")


def _hue_rgb(hue_deg: float) -> tuple[int, int, int]:
    """HSV(hue, 1.0, 1.0) with hue in degrees -> RGB tuple of 0-255."""
    h = (hue_deg % 360.0) / 60.0
    x = round(255 * (1.0 - abs(h % 2.0 - 1.0)))
    if h < 1.0:
        return (255, x, 0)
    if h < 2.0:
        return (x, 255, 0)
    if h < 3.0:
        return (0, 255, x)
    if h < 4.0:
        return (0, x, 255)
    if h < 5.0:
        return (x, 0, 255)
    return (255, 0, x)


class LightPattern:
    """Deterministic cube LED frames for a named pattern.

    Args:
        name: One of :data:`PATTERN_NAMES`.
        cube_ids: Cube IDs to include, e.g. ``(1, 2, 3)``.
        hue_step_deg: Hue advance per frame for the cycling patterns.
        fade_frames: Firmware transition length in frames; 0 switches instantly.
    """

    def __init__(
        self,
        name: str = "rainbow",
        cube_ids: tuple[int, ...] = (1, 2, 3),
        hue_step_deg: float = 12.0,
        fade_frames: int = 0,
    ) -> None:
        if name not in PATTERN_NAMES:
            raise ValueError(f"unknown pattern {name!r}, expected one of {PATTERN_NAMES}")
        self.name = name
        self.cube_ids = tuple(cube_ids)
        self.hue_step_deg = float(hue_step_deg)
        self.fade_frames = int(fade_frames)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def frame(self, index: int) -> dict[int, tuple[LightState, ...]]:
        """LED states for every cube at frame ``index``.

        Returns a mapping ``{cube_id: (corner0, corner1, corner2, corner3)}``.
        Deterministic: the same index always yields the same states.
        """
        if self.name == "rainbow":
            return self._per_cube(index, self._rainbow_color)
        if self.name == "chase":
            return self._chase(index)
        if self.name == "pulse":
            return self._pulse(index)
        return self._per_cube(index, self._solid_color)

    def cube_count(self) -> int:
        """Number of cubes this pattern addresses."""
        return len(self.cube_ids)

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def _spread_deg(self) -> float:
        """Hue separation between neighbouring cubes."""
        return 360.0 / max(1, self.cube_count())

    def _cue_brightness(self, index: int) -> float:
        """Triangle wave in 0.15-1.0, period 2 * cube count frames."""
        period = max(1, 2 * self.cube_count())
        phase = (index % period) / period
        return 0.15 + 0.85 * (1.0 - abs(2.0 * phase - 1.0))

    def _rainbow_color(self, index: int, cube_pos: int) -> int:
        hue = index * self.hue_step_deg + cube_pos * self._spread_deg()
        return Color(rgb=_hue_rgb(hue)).to_int16()

    def _solid_color(self, index: int, cube_pos: int) -> int:
        hue = cube_pos * self._spread_deg()
        return Color(rgb=_hue_rgb(hue)).to_int16()

    def _pulse_color(self, index: int, cube_pos: int) -> int:
        red, green, blue = _hue_rgb(cube_pos * self._spread_deg())
        scale = self._cue_brightness(index + cube_pos)
        dimmed = (
            round(red * scale),
            round(green * scale),
            round(blue * scale),
        )
        return Color(rgb=dimmed).to_int16()

    # ------------------------------------------------------------------
    # Pattern bodies
    # ------------------------------------------------------------------

    def _per_cube(self, index: int, color_of: Callable[[int, int], int]) -> dict[int, tuple[LightState, ...]]:
        """Every corner of each cube the same colour."""
        return {cube_id: (self._state(color_of(index, pos)),) * CORNERS for pos, cube_id in enumerate(self.cube_ids)}

    def _chase(self, index: int) -> dict[int, tuple[LightState, ...]]:
        """One lit corner per cube, travelling around the top face."""
        states: dict[int, tuple[LightState, ...]] = {}
        for pos, cube_id in enumerate(self.cube_ids):
            lit = (index + pos) % CORNERS
            colour = self._rainbow_color(index, pos)
            states[cube_id] = tuple(self._state(colour if corner == lit else 0) for corner in range(CORNERS))
        return states

    def _pulse(self, index: int) -> dict[int, tuple[LightState, ...]]:
        """Cubes breathe out of phase, each holding its own hue."""
        return {
            cube_id: (self._state(self._pulse_color(index, pos)),) * CORNERS
            for pos, cube_id in enumerate(self.cube_ids)
        }

    def _state(self, int_color: int) -> LightState:
        return LightState(
            on_color=int_color,
            off_color=int_color,
            transition_on_frames=self.fade_frames,
            transition_off_frames=self.fade_frames,
        )
