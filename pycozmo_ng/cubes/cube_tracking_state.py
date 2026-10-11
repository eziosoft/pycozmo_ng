"""Cube tracking state — typed container for cube factory/object mapping.

Tracks the mapping between public cube IDs (1-3), permanent factory IDs
assigned at the factory, and temporary object IDs assigned by the robot
firmware per BLE connection. No cycler state lives here — connection
management (persistent multi-connection keepalive) is handled by
:class:`~pycozmo_ng.cubes.manager.CubeManager`.
"""


class CubeTrackingState:
    """Mutable container for the cube ID ↔ factory_id ↔ object_id mapping.

    Owned by CubeManager.
    """

    def __init__(self) -> None:
        self.cube_factory_id_map: dict[int, int] = {}
        self.object_id_map: dict[int, int] = {}
