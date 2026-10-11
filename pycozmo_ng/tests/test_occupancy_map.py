"""Occupancy map — adaptive floor segmentation, contact line, log-odds.

Regression + behavior tests for the detection rework (COZMO-71):

- A new obstacle establishes OCCUPIED cells in one frame, and the occupied
  band sits at the obstacle's *base* (contact line), not smeared past it —
  the projection flattens z=0, so the silhouette lands too far away and only
  the near edge is at the true ground position.
- An established obstacle survives several contradictory free frames
  (damped clearing — transient blur/shadow frames must not flip the grid
  under A*), but eventually clears.
- A *shadow* over the floor (same hue, lower lightness) is not an obstacle —
  classification leans on chroma, not brightness.
- Any floor works: a dark floor classifies FREE via the adaptive model where
  the old fixed brightness threshold said "unseen"/obstacle.
- The dim interpolation ring at the projection silhouette votes nothing.

Geometry mirrors the real projection: FloorProjector default
600x600 mm @ 5 mm/px -> 120x120 px, floor_center (200, 0); row 0 = farthest
forward, col 0 = left. Robot at world origin, yaw 0 (world == body).
"""

import numpy as np

from pycozmo_ng.vision.occupancy_map.occupancy_map import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    OccupancyMap,
)

W = H = 120
_ROW_X = 500.0 - (np.arange(H) + 0.5) * 5.0  # body X: 497.5 .. -97.5 (row 0 = far)
_COL_Y = 300.0 - (np.arange(W) + 0.5) * 5.0  # body Y: 297.5 .. -297.5 (col 0 = left)
BODY_X = np.repeat(_ROW_X, W)
BODY_Y = np.tile(_COL_Y, H)

FLOOR_BGR = np.array([150, 160, 170], np.uint8)  # neutral bright floor
OBSTACLE_BGR = np.array([60, 60, 230], np.uint8)  # red object
DARK_FLOOR_BGR = np.array([55, 55, 55], np.uint8)  # dark floor (old threshold said unseen)

# Obstacle blob: rows 60..89 (body x ~197..52 mm), cols 60..89 (y ~-2..-147 mm).
# Its floor contact (near edge) is row 89 -> body x ~52 mm.
BLOB = (slice(60, 90), slice(60, 90))


def _frame(color: np.ndarray = FLOOR_BGR) -> np.ndarray:
    return np.full((H, W, 3), color, np.uint8)


def _obstacle_frame() -> np.ndarray:
    f = _frame()
    f[BLOB] = OBSTACLE_BGR
    return f


def _shadow_frame() -> np.ndarray:
    # Same hue, ~40% darker: what a real shadow does to the floor.
    f = _frame()
    f[BLOB] = (FLOOR_BGR * 0.6).astype(np.uint8)
    return f


def _step(occ: OccupancyMap, frame: np.ndarray) -> None:
    occ.update(frame, BODY_X, BODY_Y, 0.0, 0.0, 0.0)


def _occupied_world_xy(occ: OccupancyMap) -> np.ndarray:
    """World (x, y) mm of every occupied cell (origin at 0, 0, 40x40 cells)."""
    idx = np.argwhere(occ.grid == OCCUPIED)  # (grid_row, grid_col) = (world_y, world_x)
    half = (occ.grid_size_mm / occ.cell_size_mm) / 2.0
    x = (idx[:, 1] - half) * occ.cell_size_mm
    y = (idx[:, 0] - half) * occ.cell_size_mm
    return np.column_stack([x, y])


class TestBasicDetection:
    def test_fresh_grid_is_unknown(self):
        assert (OccupancyMap().grid == UNKNOWN).all()

    def test_floor_marks_free_in_one_frame(self):
        occ = OccupancyMap()
        _step(occ, _frame())
        assert (occ.grid == FREE).any()

    def test_dark_floor_is_free_via_adaptive_model(self):
        # The old fixed brightness threshold (60) could not free a 55-grey
        # floor; the adaptive model has no fixed threshold at all.
        occ = OccupancyMap()
        _step(occ, _frame(DARK_FLOOR_BGR))
        assert (occ.grid == FREE).any(), "dark floor not classified free"

    def test_new_obstacle_marks_occupied_in_one_frame(self):
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _obstacle_frame())
        assert (occ.grid == OCCUPIED).any()


class TestContactLine:
    def test_occupied_band_sits_at_the_obstacle_base(self):
        # Blob spans body x ~52..197 mm. Only its base may be occupied: the
        # contact line is at x ~52 mm (+/- inflation); the smear from
        # projecting the whole silhouette would reach x ~150..200 mm.
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _obstacle_frame())

        xy = _occupied_world_xy(occ)
        assert len(xy) > 0
        assert xy[:, 0].min() >= 0.0, f"occupied behind the base: {xy[:, 0].min()}"
        assert xy[:, 0].max() <= 100.0, f"occupied smeared past the base: {xy[:, 0].max()}"
        # And the band spans the blob's width (lateral), not a single pixel.
        assert xy[:, 1].min() <= -100.0
        assert xy[:, 1].max() >= -50.0


class TestClearingIsDamped:
    def test_occupied_cell_survives_three_free_frames(self):
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _obstacle_frame())
        # Guard: the harness really does establish occupied cells in one frame.
        assert (occ.grid == OCCUPIED).any()

        for _ in range(3):
            _step(occ, _frame())
        assert (occ.grid == OCCUPIED).any(), "occupied cell freed too fast"

    def test_occupied_cell_eventually_clears(self):
        # Damped, not frozen: enough consistent free observations clear it
        # (the removed obstacle must not block A* forever).
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _obstacle_frame())
        assert (occ.grid == OCCUPIED).any()

        for _ in range(10):
            _step(occ, _frame())
        assert not (occ.grid == OCCUPIED).any()


class TestShadowSuppression:
    def test_shadow_is_not_an_obstacle(self):
        # A shadowed patch of floor keeps its hue and only loses lightness;
        # classification leans on chroma, so it must not vote occupied.
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _shadow_frame())
        assert not (occ.grid == OCCUPIED).any(), "shadow voted as obstacle"

    def test_shadowed_patch_stays_free(self):
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _shadow_frame())
        assert (occ.grid == FREE).any()


class TestSilhouetteRing:
    def test_dim_interpolation_ring_votes_nothing(self):
        # The projection boundary is black unmapped -> a dim 1-2px
        # interpolation ring -> floor. The ring is not floor color; the
        # seen-mask erosion must drop it before classification or the whole
        # visible border votes occupied.
        occ = OccupancyMap()
        f = _frame()
        f[:2, :] = 0  # black unmapped outside the silhouette
        f[-2:, :] = 0
        f[:, :2] = 0
        f[:, -2:] = 0
        f[2:4, :] = 30  # dim interpolation ring (30 >= valid_min_brightness)
        f[-4:-2, :] = 30
        f[:, 2:4] = 30
        f[:, -4:-2] = 30
        _step(occ, _frame())
        _step(occ, f)
        assert not (occ.grid == OCCUPIED).any(), "silhouette ring voted as obstacle"


class TestLifecycle:
    def test_origin_set_from_first_pose(self):
        occ = OccupancyMap()
        assert occ.origin == (0.0, 0.0)
        occ.update(_frame(), BODY_X, BODY_Y, 120.0, -40.0, 0.0)
        assert occ.origin == (120.0, -40.0)

    def test_reset_clears_grid_and_floor_model(self):
        occ = OccupancyMap()
        _step(occ, _frame())
        _step(occ, _obstacle_frame())
        assert (occ.grid == OCCUPIED).any()
        assert occ.floor_model is not None

        occ.reset()
        assert (occ.grid == UNKNOWN).all()
        assert occ.floor_model is None
