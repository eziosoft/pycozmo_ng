"""Cube tracking state — typed container for cube mapping and BLE cycling."""

from pycozmo_ng.cubes.cycler import CubeCycler


class CubeTrackingState:
    """Mutable container for cube factory/object mapping and the cycler.

    Owned by CubeManager.
    """

    def __init__(self) -> None:
        self.cube_factory_id_map: dict[int, int] = {}
        self.object_id_map: dict[int, int] = {}
        self.cycler: CubeCycler | None = None
