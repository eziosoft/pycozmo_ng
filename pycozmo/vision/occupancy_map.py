"""Occupancy map — accumulates body-frame floor observations into a world-frame grid.

Takes bird's-eye floor images from :class:`~pycozmo.vision.floor_projector.FloorProjector`
and robot pose, classifies free/occupied cells via brightness + edge detection,
and accumulates votes in a fixed-size world-frame grid centered on the robot's
starting position.
"""

import numpy as np
import cv2

from pycozmo import logger

__all__ = ["OccupancyMap"]

# Cell states
UNKNOWN = -1
OCCUPIED = 0
FREE = 1


class OccupancyMap:
    """Accumulates body-frame floor observations into a world-frame occupancy grid.

    Args:
        grid_size_mm: Physical extent of the grid in mm (square).
        cell_size_mm: Millimetres per grid cell.
        inflation_radius: Dilate edge mask by this many iterations.
        free_threshold: Brightness above this (and no edge) → free.
        edge_low: Canny low threshold.
        edge_high: Canny high threshold.
        vote_decay: Exponential decay applied to accumulated votes every frame
            (0.0-1.0). <1 makes the map react when the robot moves away or a
            region stops being observed; older observations fade out.
        valid_min_brightness: Grey value below which a floor pixel is treated
            as unseen (excludes the black unmapped border and the dim 1-2px
            interpolation ring along the projection boundary).
        valid_erosion: Erode the seen mask by this many pixels so the Canny
            edge ring at the projection boundary is not voted as occupied.
    """

    def __init__(
        self,
        grid_size_mm: float = 2000.0,
        cell_size_mm: float = 5.0,
        inflation_radius: int = 2,
        free_threshold: int = 60,
        edge_low: int = 50,
        edge_high: int = 150,
        vote_decay: float = 0.95,
        valid_min_brightness: int = 8,
        valid_erosion: int = 1,
    ):
        self._grid_size_mm = grid_size_mm
        self._cell_size_mm = cell_size_mm
        self._grid_cells = int(grid_size_mm / cell_size_mm)

        # Vote accumulators
        self._free_votes: np.ndarray = np.zeros(
            (self._grid_cells, self._grid_cells), dtype=np.float64
        )
        self._occupied_votes: np.ndarray = np.zeros(
            (self._grid_cells, self._grid_cells), dtype=np.float64
        )

        # Classified grid: -1 = unknown, 0 = occupied, 1 = free
        self._grid: np.ndarray = np.full(
            (self._grid_cells, self._grid_cells), UNKNOWN, dtype=np.int8
        )

        # World origin (set on first update from robot position)
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._origin_set: bool = False

        # Parameters
        self._inflation_radius = inflation_radius
        self._free_threshold = free_threshold
        self._edge_low = edge_low
        self._edge_high = edge_high
        self._vote_decay = float(min(1.0, max(0.0, vote_decay)))
        self._vote_epsilon = 1e-3  # below this a decayed vote is forgotten
        self._valid_min_brightness = valid_min_brightness
        self._valid_erosion = int(max(0, valid_erosion))

        logger.info(
            f"OccupancyMap: {self._grid_cells}x{self._grid_cells} cells, "
            f"{grid_size_mm}mm @ {cell_size_mm} mm/cell, "
            f"inflation={inflation_radius}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all votes and reset grid to unknown."""
        self._free_votes[:] = 0
        self._occupied_votes[:] = 0
        self._grid[:] = UNKNOWN
        self._origin_set = False
        logger.info("OccupancyMap reset")

    def update(
        self,
        floor_image: np.ndarray,
        body_x: np.ndarray,
        body_y: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_yaw_rad: float,
    ) -> None:
        """Process one frame: classify, accumulate, reclassify.

        Args:
            floor_image: Bird's-eye BGR image from FloorProjector.project().
            body_x: 1-D array of body-frame X coords (mm) for each pixel,
                same layout as the floor image (row-major, shape = H*W).
            body_y: 1-D array of body-frame Y coords (mm) for each pixel.
            robot_x: Robot world-frame X position (mm).
            robot_y: Robot world-frame Y position (mm).
            robot_yaw_rad: Robot yaw about world Z (radians).
        """
        if not self._origin_set:
            self._origin_x = robot_x
            self._origin_y = robot_y
            self._origin_set = True

        # --- Classify body-frame image -----------------------------------
        gray = cv2.cvtColor(floor_image, cv2.COLOR_BGR2GRAY)

        # Edge detection → obstacle boundaries
        edges = cv2.Canny(gray, self._edge_low, self._edge_high)
        if self._inflation_radius > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=self._inflation_radius)

        # Seen mask: drop black (unmapped) areas and the dim interpolation ring
        # along the projection boundary, then erode so the Canny edge at the
        # silhouette of the projection is not treated as an obstacle.
        seen = ((gray >= self._valid_min_brightness).astype(np.uint8)) * 255
        if self._valid_erosion > 0:
            seen = cv2.erode(
                seen, np.ones((3, 3), np.uint8), iterations=self._valid_erosion
            )

        # Masks (same shape as gray, flattened to 1-D)
        is_valid = (seen > 0).ravel()
        edge_flat = (edges > 0).ravel() & is_valid
        gray_flat = gray.ravel()

        is_occupied = is_valid & edge_flat
        is_free = is_valid & (gray_flat >= self._free_threshold) & ~edge_flat

        # --- Body → world transform -------------------------------------
        c, s = np.cos(robot_yaw_rad), np.sin(robot_yaw_rad)
        world_x = body_x * c - body_y * s + robot_x
        world_y = body_x * s + body_y * c + robot_y

        # --- World → grid indices ----------------------------------------
        half = self._grid_size_mm / 2.0
        col = ((world_x - self._origin_x + half) / self._cell_size_mm).astype(np.int32)
        row = ((world_y - self._origin_y + half) / self._cell_size_mm).astype(np.int32)

        in_bounds = (
            (col >= 0)
            & (col < self._grid_cells)
            & (row >= 0)
            & (row < self._grid_cells)
        )

        # --- Accumulate votes --------------------------------------------
        # Fade old votes so the map reacts when the robot moves away or a
        # region changes; otherwise a single stale vote sticks forever.
        # Fully-decayed votes are zeroed so cells return to UNKNOWN instead of
        # asymptoting above the vote threshold forever.
        if self._vote_decay < 1.0:
            self._free_votes *= self._vote_decay
            self._occupied_votes *= self._vote_decay
            self._free_votes[self._free_votes < self._vote_epsilon] = 0.0
            self._occupied_votes[self._occupied_votes < self._vote_epsilon] = 0.0

        np.add.at(
            self._free_votes,
            (row[in_bounds], col[in_bounds]),
            is_free[in_bounds].astype(np.float64),
        )
        np.add.at(
            self._occupied_votes,
            (row[in_bounds], col[in_bounds]),
            is_occupied[in_bounds].astype(np.float64),
        )

        # --- Reclassify --------------------------------------------------
        has_votes = (self._free_votes + self._occupied_votes) > 0
        self._grid[~has_votes] = UNKNOWN
        self._grid[has_votes & (self._occupied_votes > self._free_votes)] = OCCUPIED
        self._grid[has_votes & (self._free_votes >= self._occupied_votes)] = FREE

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def to_color_image(self) -> np.ndarray:
        """Convert grid to a BGR image for visualization / WebSocket transmission.

        Colors: unknown = dark gray (40), free = green, occupied = red.
        Transposed so image rows = world X (forward at top) and cols = world Y
        (left at left), matching FloorProjector's output convention and the
        scene plane orientation (local X=right, local Z=forward).
        """
        img = np.full((self._grid_cells, self._grid_cells, 3), 40, dtype=np.uint8)
        img[self._grid == FREE] = [0, 200, 0]
        img[self._grid == OCCUPIED] = [0, 0, 200]
        return np.flipud(np.fliplr(np.transpose(img, (1, 0, 2))))

    @property
    def grid(self) -> np.ndarray:
        """Classified occupancy grid (int8: -1=unknown, 0=occupied, 1=free)."""
        return self._grid

    @property
    def origin(self) -> tuple[float, float]:
        """World-frame origin of the grid (robot start position)."""
        return self._origin_x, self._origin_y

    @property
    def grid_size_mm(self) -> float:
        return self._grid_size_mm

    @property
    def cell_size_mm(self) -> float:
        return self._cell_size_mm
