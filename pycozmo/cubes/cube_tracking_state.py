"""Cube tracking state — typed container for cube mapping and BLE cycling."""

from pycozmo.cubes.cube_imu import CubeMoveDetector
from pycozmo.cubes.cycler import CubeCycler


class CubeTrackingState:
    """Mutable container for cube factory/object mapping, cycler, and movement.

    Owned by CubeManager; read by VisionDisplay._stamp_detections_seen().
    """

    def __init__(self) -> None:
        self.cube_factory_id_map: dict[int, int] = {}
        self.object_id_map: dict[int, int] = {}
        self.last_seen_frame: dict[int, int] = {}
        self.cycler: CubeCycler | None = None
        self.move_detectors: dict[int, CubeMoveDetector] = {}
        self.suppress: dict = {
            "ids": set(),
            "fresh_frames": {},
            "fresh_frames_needed": 5,
        }
