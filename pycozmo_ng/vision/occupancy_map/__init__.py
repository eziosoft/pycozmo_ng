"""Occupancy mapping subsystem.

Re-exports the core occupancy-grid API so existing imports keep working:

    from pycozmo_ng.vision.occupancy_map import OccupancyMap, FREE, OCCUPIED, UNKNOWN
"""

from pycozmo_ng.vision.occupancy_map.occupancy_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyMap,
)

__all__ = ["FREE", "OCCUPIED", "UNKNOWN", "OccupancyMap"]
