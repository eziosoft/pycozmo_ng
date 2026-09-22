"""WorldMap interfaces — abstractions for swappable world mapping."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from pycozmo.util import Pose


class WorldMapInterface(Protocol):
    """Protocol for world map implementations.

    Provides read-only access to fused world-frame state.
    """

    def get_robot_pose(self) -> Pose | None:
        """Get current robot pose in world frame."""
        ...

    def get_cube_pose(self, cube_id: int) -> dict | None:
        """Get a specific cube's world pose.

        Returns dict with position (x,y,z mm), rotation_matrix, confidence.
        """
        ...

    def get_all_cube_poses(self) -> dict[int, dict]:
        """Get all known cube poses in world frame."""
        ...

    def get_path(self) -> list[tuple[float, float, float]]:
        """Get the robot's path trail as list of (x, y, z) positions."""
        ...

    # Measurement updates (called from vision pipeline)
    def update_cube_measurement(
        self,
        cube_id: int,
        position_cam: np.ndarray,
        rotation_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        num_faces: int | None = None,
        camera_lateral_offset_mm: float = 0.0,
        rotation_source: str | None = None,
    ) -> None:
        """Fuse a camera-frame cube measurement into world map."""
        ...

    def forget_cube(self, cube_id: int) -> None:
        """Drop a cube from the world map (e.g. on IMU move)."""
        ...

    def update_face_measurement(
        self,
        face_id: int,
        position_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        camera_lateral_offset_mm: float = 0.0,
        label: str = "face",
    ) -> None:
        """Fuse a camera-frame face measurement into world map."""
        ...

    def forget_face(self, face_id: int) -> None:
        """Remove a face from the world map."""
        ...

    def add_path_point(self, position: tuple[float, float, float]) -> None:
        """Append a world-frame position to the path trail."""
        ...

    def reset_path(self) -> None:
        """Clear the path trail."""
        ...

    def reset_map(self) -> None:
        """Reset the whole world map (entities + path)."""
        ...
