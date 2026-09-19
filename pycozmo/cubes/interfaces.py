"""Cube interfaces — abstractions for swappable cube management."""

from __future__ import annotations

from typing import Protocol

import pycozmo


class CubeCyclerInterface(Protocol):
    """Protocol for cube BLE cycler implementations."""

    stats: dict

    def start(self, cli: pycozmo.Client, cube_factory_id_map: dict[int, int]) -> None:
        """Start cycling cubes."""
        ...

    def stop(self) -> None:
        """Stop cycling."""
        ...


class CubeMoveDetectorInterface(Protocol):
    """Protocol for cube movement detection."""

    def on_connection(self, timestamp: float) -> None:
        """Called when cube connects."""
        ...

    def on_accel(self, timestamp: float, accel: tuple[float, float, float]) -> bool:
        """Process accelerometer data. Returns True if movement detected."""
        ...

    def on_disconnect(self) -> None:
        """Called when cube disconnects."""
        ...


class CubeInterface(Protocol):
    """Protocol for individual cube representation."""

    cube_id: int
    factory_id: int
    object_id: int | None

    def set_lights(self, color) -> None:
        """Set cube LED color."""
        ...


class CubeManagerInterface(Protocol):
    """Protocol for cube management implementations."""

    ct: object  # CubeTrackingState

    def on_object_connection_state(self, cli: pycozmo.Client, pkt) -> None:
        """Handle cube connection state changes."""
        ...

    def setup_cube_mapping(self, cli: pycozmo.Client) -> None:
        """Discover available cubes and create mapping."""
        ...

    def start_cube_cycler(self, cli: pycozmo.Client) -> None:
        """Start the BLE cube cycler."""
        ...

    def stop_cube_cycler(self) -> None:
        """Stop the BLE cube cycler."""
        ...

    def get_cube(self, cube_id: int) -> CubeInterface | None:
        """Get a cube by ID (1-3)."""
        ...

    @property
    def connected_cubes(self) -> dict[int, CubeInterface]:
        """Get connected cubes."""
        ...

    # Legacy aliases
    def start_cube_light_worker(self, cli: pycozmo.Client) -> None:
        """Start cube keepalive worker (legacy)."""
        ...

    def stop_cube_light_worker(self) -> None:
        """Stop cube keepalive worker (legacy)."""
        ...