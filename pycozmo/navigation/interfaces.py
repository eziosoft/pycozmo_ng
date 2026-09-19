"""Navigation interfaces — abstractions for swappable navigation algorithms."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CubeApproachInterface(Protocol):
    """Protocol for cube approach implementations."""

    cli: object | None

    def approach(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Start approaching a cube at given world coordinates."""
        ...

    @staticmethod
    def pick_face_normal(
        cube_pos: np.ndarray,
        cube_rot: np.ndarray,
        robot_pos: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float, float] | None:
        """Pick the cube face to approach."""
        ...


class NavigatorInterface(Protocol):
    """Protocol for navigation implementations.

    Any navigation algorithm can be plugged in by implementing this interface.
    """

    # Cross-domain refs (set by Brain after construction)
    cli: object | None
    web_viewer: object | None

    def drive_to_point(
        self,
        tx: float,
        ty: float,
        *,
        stop_mm: float = 20.0,
        timeout_s: float = 15.0,
        speed_mmps: float = 60.0,
        final_heading: float | None = None,
        align_tol_rad: float = 0.06,
    ) -> bool:
        """Drive to a world-frame point (mm)."""
        ...

    def approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Approach and align with a cube."""
        ...

    def dock(self) -> None:
        """Drive back to dock/origin."""
        ...

    def stop(self) -> None:
        """Stop all movement."""
        ...

    # Event handlers (set by Brain)
    def _on_charger_change(self, cli, state: bool) -> None:
        """Handle charger state change."""
        ...

    def _on_charging_change(self, cli, state: bool) -> None:
        """Handle charging state change."""
        ...

    def _on_navigate_to(self, world_forward: float, world_left: float) -> None:
        """Handle click-to-navigate from browser."""
        ...

    def _on_approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Handle cube click from browser."""
        ...

    def _on_dock(self) -> None:
        """Handle dock request from browser."""
        ...


class DriveToPointFn(Protocol):
    """Function signature for drive-to-point callback."""

    def __call__(
        self,
        tx: float,
        ty: float,
        *,
        stop_mm: float = 20.0,
        timeout_s: float = 15.0,
        speed_mmps: float = 60.0,
        final_heading: float | None = None,
        align_tol_rad: float = 0.06,
    ) -> bool:
        ...


class MoveForwardFn(Protocol):
    """Function signature for move-forward callback."""

    def __call__(self, dist_mm: float, timeout_s: float = 6.0) -> bool:
        ...