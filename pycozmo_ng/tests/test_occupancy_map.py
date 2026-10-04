"""Occupancy map flip latency.

Regression for "the occupancy map reacts too fast": an established
OCCUPIED cell must not turn FREE on a single contradictory frame — it has
to survive several consistent free observations first (damped clearing),
while a *new* obstacle still marks cells in one frame (undamped reaction).

Geometry mirrors the real projection: FloorProjector default
600x600 mm @ 5 mm/px -> 120x120 px, floor_center (200, 0).
"""

import numpy as np

from pycozmo_ng.vision.occupancy_map.occupancy_map import (
    FREE,
    OCCUPIED,
    OccupancyMap,
)

W = H = 120
_XS = np.linspace(-100.0, 500.0, W)
_YS = np.linspace(-300.0, 300.0, H)
BODY_X = np.tile(_XS, H)
BODY_Y = np.repeat(_YS, W)

# Bright floor: every pixel free, no edges.
FREE_FRAME = np.full((H, W, 3), 200, np.uint8)

# Dim floor (seen, but below free_threshold) with a bright object: Canny
# fires on the outline, and the cells just outside it see occupied pixels
# and no free pixels, so they are the cells that should become OCCUPIED.
OBSTACLE_FRAME = np.full((H, W, 3), 50, np.uint8)
OBSTACLE_FRAME[60:90, 60:90] = 220


def _step(occ: OccupancyMap, frame: np.ndarray) -> None:
    occ.update(frame, BODY_X, BODY_Y, 0.0, 0.0, 0.0)


class TestFreeClearingIsDamped:
    def test_occupied_cell_survives_free_frames(self):
        occ = OccupancyMap()

        _step(occ, FREE_FRAME)
        _step(occ, OBSTACLE_FRAME)
        # Guard: the harness really does establish occupied cells in one frame.
        assert (occ.grid == OCCUPIED).any()

        # The symptom: a few free frames must NOT wipe an established cell.
        for _ in range(5):
            _step(occ, FREE_FRAME)
        assert (occ.grid == OCCUPIED).any(), "occupied cell freed too fast"

    def test_occupied_cell_eventually_clears(self):
        # Damped, not frozen: enough consistent free observations clear it.
        occ = OccupancyMap()
        _step(occ, FREE_FRAME)
        _step(occ, OBSTACLE_FRAME)
        assert (occ.grid == OCCUPIED).any()

        for _ in range(60):
            _step(occ, FREE_FRAME)
        assert not (occ.grid == OCCUPIED).any()

    def test_free_cell_flips_to_occupied_in_one_frame(self):
        # The other direction stays fast: reaction to a new obstacle is
        # not damped, only clearing is.
        occ = OccupancyMap()
        _step(occ, FREE_FRAME)
        _step(occ, FREE_FRAME)
        _step(occ, FREE_FRAME)
        assert (occ.grid == FREE).any()

        _step(occ, OBSTACLE_FRAME)
        assert (occ.grid == OCCUPIED).any()
