"""SafetyLimits — guards on LLM-initiated motion (COZMO-68).

Physical caps applied by every motion tool in the MCP toolset before a
command reaches the robot. The defaults were agreed in the design
session: extent 2000 mm from the world origin, speed 150 mm/s (Cozmo's
top speed is ~250 mm/s), single motion command 60 s, approach/dock
visits 120 s, and motion confirmation off unless a deployment turns it
on.

The limits are plain data — construct ``SafetyLimits(extent_mm=...)``
to override per deployment (a smaller play area, a slower robot).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyLimits:
    """Physical caps enforced on MCP-initiated motion, in world units."""

    #: Reject navigate targets farther than this from the world origin (mm).
    extent_mm: float = 2000.0
    #: Hard cap on requested cruise speed (mm/s).
    speed_mmps: float = 150.0
    #: Timeout passed to a single closed-loop drive command (seconds).
    motion_timeout_s: float = 60.0
    #: Give-up window for a whole approach/dock visit (seconds).
    visit_timeout_s: float = 120.0
    #: When True, every motion tool must be called with ``confirm=True``.
    require_confirmation: bool = False

    def target_error(self, x_mm: float, y_mm: float) -> str | None:
        """Why *x_mm, y_mm* may not be driven to, or ``None`` when in range."""
        distance = math.hypot(x_mm, y_mm)
        if distance > self.extent_mm:
            return (
                f"Target ({x_mm:.0f}, {y_mm:.0f}) mm is {distance:.0f} mm from the origin, "
                f"beyond the {self.extent_mm:.0f} mm extent limit - pick a closer point"
            )
        return None

    def clamp_speed(self, requested_mmps: float) -> float:
        """*requested_mmps* capped at :attr:`speed_mmps`."""
        return max(0.0, min(requested_mmps, self.speed_mmps))

    def confirmation_error(self, confirm: bool) -> str | None:
        """Why a motion command needs re-sending, or ``None`` when allowed."""
        if self.require_confirmation and not confirm:
            return "Motion requires confirmation - call again with confirm=true"
        return None
