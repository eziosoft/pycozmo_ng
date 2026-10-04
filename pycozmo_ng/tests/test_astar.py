"""Tests for A* pathfinding on the occupancy grid (pycozmo_ng/navigation/astar.py).

Key invariants guarded here:

- The planner must use the SAME centered grid indexing as
  ``OccupancyMap.update`` (``col = (wx - ox + half) / cell``). A missing
  ``half`` offset shifts every lookup by 20 cells / 1000 mm, so the planner
  sees a phantom grid where obstacles do not exist and line-of-sight always
  "succeeds" — driving straight through red cells.
- The map origin is set on the map's first update, so the planner must read
  it live, not cache it at construction.
- Paths must never contain a waypoint in an occupied cell.
"""

from pycozmo_ng.navigation.astar import AStarPlanner, PathPoint
from pycozmo_ng.vision.occupancy_map import FREE, OCCUPIED, OccupancyMap

# 2000 mm grid @ 50 mm/cell = 40x40, centered: world (0,0) is cell (20, 20).
HALF_MM = 1000.0
CELL_MM = 50.0


def _make_map(size_mm: float = 1000.0, cell_mm: float = 50.0) -> OccupancyMap:
    """Create an occupancy map with every cell FREE (origin at 0, 0)."""
    occ = OccupancyMap(grid_size_mm=size_mm, cell_size_mm=cell_mm)
    occ._grid[:] = FREE
    return occ


class TestGridIndexing:
    def test_world_zero_is_center_cell(self):
        """Regression: world (0,0) must map to the CENTER cell, not (0, 0)."""
        occ = OccupancyMap(grid_size_mm=2000.0, cell_size_mm=50.0)
        planner = AStarPlanner(occ)
        assert planner._world_to_grid(0.0, 0.0) == (20, 20)

    def test_matches_occupancy_map_update_indexing(self):
        """Planner indexing must equal OccupancyMap.update's formula."""
        occ = OccupancyMap(grid_size_mm=2000.0, cell_size_mm=50.0)
        planner = AStarPlanner(occ)
        for wx, wy in [(0, 0), (200, 150), (-300, 400), (950, -950), (-999, 999)]:
            expected = (
                int((wx - occ.origin[0] + HALF_MM) / CELL_MM),
                int((wy - occ.origin[1] + HALF_MM) / CELL_MM),
            )
            assert planner._world_to_grid(float(wx), float(wy)) == expected

    def test_grid_world_round_trip(self):
        occ = _make_map()
        planner = AStarPlanner(occ)
        for col, row in [(0, 0), (10, 10), (19, 5), (15, 18)]:
            wx = planner._grid_to_world_col(col)
            wy = planner._grid_to_world_row(row)
            assert planner._world_to_grid(wx, wy) == (col, row)

    def test_origin_read_live_not_cached(self):
        """The map origin is set on first update — after planner creation."""
        occ = OccupancyMap(grid_size_mm=2000.0, cell_size_mm=50.0)
        planner = AStarPlanner(occ)  # constructed while origin is still (0, 0)
        occ._origin_x = 100.0
        occ._origin_y = -50.0
        assert planner._world_to_grid(100.0, -50.0) == (20, 20)


class TestFindPath:
    def test_direct_when_clear(self):
        """Clear map: line-of-sight early exit returns a 2-point path."""
        occ = _make_map()
        planner = AStarPlanner(occ, allow_unknown=False)
        path = planner.find_path(0.0, 0.0, 400.0, 0.0)
        assert path is not None
        assert len(path) == 2
        assert path[-1].x == 400.0
        assert path[-1].y == 0.0

    def test_avoids_occupied_wall(self):
        """A wall between start and goal must be detoured around.

        Regression: with the missing ``half`` offset the planner indexed a
        grid region 1000 mm away from the real one, found only FREE/UNKNOWN
        cells, and returned a straight 2-point path through the wall.
        """
        occ = _make_map()  # 1000 mm, 50 mm/cell -> 20x20, half=500
        # Wall: column at world x ~100..150 (col 12), rows 5..14.
        occ._grid[5:15, 12] = OCCUPIED
        planner = AStarPlanner(occ, allow_unknown=False)

        # Start (0,0) -> cell (10, 10); goal (400, 0) -> cell (18, 10).
        path = planner.find_path(0.0, 0.0, 400.0, 0.0)
        assert path is not None, "path around a partial wall must exist"

        # No waypoint may sit in an occupied cell.
        for wp in path:
            col, row = planner._world_to_grid(wp.x, wp.y)
            assert planner._is_traversable(row, col), f"waypoint in obstacle: {wp}"

        # The path must actually deviate (not a straight line through row 10).
        detour_rows = {planner._world_to_grid(wp.x, wp.y)[1] for wp in path}
        assert detour_rows != {10}, "path went straight through the wall"

    def test_no_path_when_walled_off(self):
        occ = _make_map()
        occ._grid[:, 12] = OCCUPIED
        planner = AStarPlanner(occ, allow_unknown=False)
        assert planner.find_path(0.0, 0.0, 400.0, 0.0) is None

    def test_start_out_of_bounds_returns_none(self):
        occ = _make_map()
        planner = AStarPlanner(occ, allow_unknown=False)
        assert planner.find_path(-5000.0, 0.0, 0.0, 0.0) is None

    def test_goal_occupied_returns_none(self):
        occ = _make_map()
        # Goal (400, 0) -> cell (18, 10).
        occ._grid[10, 18] = OCCUPIED
        planner = AStarPlanner(occ, allow_unknown=False)
        assert planner.find_path(0.0, 0.0, 400.0, 0.0) is None


class TestPathPoint:
    def test_to_tuple(self):
        assert PathPoint(12.5, -3.0).to_tuple() == (12.5, -3.0)
