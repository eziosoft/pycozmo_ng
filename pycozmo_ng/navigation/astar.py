"""A* pathfinding on the occupancy grid.

Uses the world-frame occupancy grid from :class:`OccupancyMap` to find
a collision-free path between two points.  Returns waypoints in world
frame (mm, Anki convention: X forward, Y left) that can be fed to the
navigator for following.
"""

import heapq
import math

import numpy as np

from pycozmo_ng.vision.occupancy_map import OCCUPIED, UNKNOWN, OccupancyMap

__all__ = ["AStarPlanner", "PathPoint"]

# Heuristic weight multiplier.  Slightly inflated heuristics speed up
# search at the cost of exploring a few extra nodes.
HEURISTIC_WEIGHT = 1.0

# Diagonal movement cost.
COST_STRAIGHT = 1.0
COST_DIAGONAL = math.sqrt(2)


class PathPoint:
    """A single waypoint on an A* path."""

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"PathPoint({self.x:.1f}, {self.y:.1f})"

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


class AStarPlanner:
    """A* pathfinder operating on an :class:`OccupancyMap` grid.

    Parameters
    ----------
    occupancy_map:
        The occupancy map to search on.
    allow_unknown:
        If ``True`` (default), unknown cells are treated as
        traversable.  If ``False``, only cells classified as *free* are
        traversable.
    """

    def __init__(self, occupancy_map: OccupancyMap, allow_unknown: bool = True):
        self._occ = occupancy_map
        self._allow_unknown = allow_unknown
        self._cell_size: float = occupancy_map.cell_size_mm
        # The grid is CENTERED on the map origin (matches
        # OccupancyMap.update): world = origin + (col + 0.5) * cell - half.
        self._half: float = occupancy_map.grid_size_mm / 2.0

    @property
    def _grid(self) -> np.ndarray:
        return self._occ.grid

    @property
    def _grid_cells(self) -> int:
        return self._occ.grid.shape[0]

    # -- public API -------------------------------------------------------

    def find_path(
        self,
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
    ) -> list[PathPoint] | None:
        """Compute a path from *(start_x, start_y)* to *(goal_x, goal_y)*.

        Args:
            start_x, start_y: Start position in world frame (mm).
            goal_x, goal_y: Goal position in world frame (mm).

        Returns:
            List of ``PathPoint`` waypoints (world frame, mm) including
            the first point adjacent to the start, or ``None`` if no
            path exists.
        """
        start_col, start_row = self._world_to_grid(start_x, start_y)
        goal_col, goal_row = self._world_to_grid(goal_x, goal_y)

        if not self._in_bounds(start_col, start_row) or not self._in_bounds(
            goal_col, goal_row
        ):
            return None

        if not self._is_traversable(start_row, start_col) or not self._is_traversable(
            goal_row, goal_col
        ):
            return None

        # Early exit: direct line of sight
        if self._line_of_sight(start_row, start_col, goal_row, goal_col):
            return [
                PathPoint(
                    self._grid_to_world_col(start_col),
                    self._grid_to_world_row(start_row),
                ),
                PathPoint(goal_x, goal_y),
            ]

        path = self._search(start_col, start_row, goal_col, goal_row)
        if path is None:
            return None

        # Convert grid indices to world-frame waypoints.
        # Skip the start cell itself — the robot is already there.
        waypoints = []
        for col, row in path[1:]:
            wx = self._grid_to_world_col(col)
            wy = self._grid_to_world_row(row)
            waypoints.append(PathPoint(wx, wy))

        # Append the exact goal.
        waypoints.append(PathPoint(goal_x, goal_y))
        return waypoints

    def set_allow_unknown(self, allow: bool) -> None:
        """Toggle whether unknown cells are traversable."""
        self._allow_unknown = allow

    def is_traversable(self, wx: float, wy: float) -> bool:
        """Check whether world-frame *(wx, wy)* is traversable."""
        col, row = self._world_to_grid(wx, wy)
        if not (0 <= col < self._grid_cells and 0 <= row < self._grid_cells):
            return False
        val = self._grid[row, col]
        return not (val == OCCUPIED or (val == UNKNOWN and not self._allow_unknown))

    # -- grid helpers -----------------------------------------------------

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        """Convert world-frame (mm) to grid indices (col, row).

        Uses the same centered indexing as :meth:`OccupancyMap.update`
        (grid origin is the cell at ``half`` offsets, not cell 0).
        Origin is read live — it is set on the map's first update.
        """
        ox, oy = self._occ.origin
        col = int((wx - ox + self._half) / self._cell_size)
        row = int((wy - oy + self._half) / self._cell_size)
        return col, row

    def _grid_to_world_col(self, col: int) -> float:
        """Convert grid column to world X (mm) — cell center."""
        ox, _ = self._occ.origin
        return ox + (col + 0.5) * self._cell_size - self._half

    def _grid_to_world_row(self, row: int) -> float:
        """Convert grid row to world Y (mm) — cell center."""
        _, oy = self._occ.origin
        return oy + (row + 0.5) * self._cell_size - self._half

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self._grid_cells and 0 <= row < self._grid_cells

    def _is_traversable(self, row: int, col: int) -> bool:
        val = self._grid[row, col]
        return not (val == OCCUPIED or (val == UNKNOWN and not self._allow_unknown))

    # -- A* search --------------------------------------------------------

    def _search(
        self,
        start_col: int,
        start_row: int,
        goal_col: int,
        goal_row: int,
    ) -> list[tuple[int, int]] | None:
        """Run A* from *(start_col, start_row)* to *(goal_col, goal_row)*.

        Returns a list of (col, row) grid indices including start and
        goal, or ``None`` if no path exists.
        """
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        costs = [
            COST_DIAGONAL,
            COST_STRAIGHT,
            COST_DIAGONAL,
            COST_STRAIGHT,
            COST_STRAIGHT,
            COST_DIAGONAL,
            COST_STRAIGHT,
            COST_DIAGONAL,
        ]

        def h(c: int, r: int) -> float:
            dx = abs(c - goal_col)
            dy = abs(r - goal_row)
            return HEURISTIC_WEIGHT * (dx + dy) + (COST_DIAGONAL - 2) * min(dx, dy)

        open_set: list[tuple[float, float, int, int]] = []
        heapq.heappush(open_set, (0, 0, start_col, start_row))

        g_score: dict[tuple[int, int], float] = {(start_col, start_row): 0.0}
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {
            (start_col, start_row): None
        }
        closed: set[tuple[int, int]] = set()

        while open_set:
            _f, g, col, row = heapq.heappop(open_set)
            current = (col, row)

            if current in closed:
                continue
            closed.add(current)

            if col == goal_col and row == goal_row:
                return self._reconstruct(came_from, current)

            for (dc, dr), cost in zip(dirs, costs):
                ncol, nrow = col + dc, row + dr
                neighbor = (ncol, nrow)

                if not self._in_bounds(ncol, nrow):
                    continue
                if not self._is_traversable(nrow, ncol):
                    continue
                if neighbor in closed:
                    continue

                tentative_g = g + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + h(ncol, nrow)
                    heapq.heappush(open_set, (f_score, tentative_g, ncol, nrow))

        return None

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        current: tuple[int, int] | None,
    ) -> list[tuple[int, int]]:
        """Rebuild the path from start to *current*."""
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _line_of_sight(self, r0: int, c0: int, r1: int, c1: int) -> bool:
        """Bresenham line-of-sight check between two grid cells."""
        dx = abs(c1 - c0)
        dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy

        while True:
            if not self._is_traversable(r0, c0):
                return False
            if c0 == c1 and r0 == r1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                c0 += sx
            if e2 < dx:
                err += dx
                r0 += sy
        return True
