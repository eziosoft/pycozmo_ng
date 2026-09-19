"""Occupancy map — accumulates body-frame floor observations into a world-frame grid.

Takes bird's-eye floor images from :class:`~pycozmo.vision.floor_projector.FloorProjector`
and robot pose, classifies free/occupied cells via brightness + edge detection,
and accumulates votes in a fixed-size world-frame grid centered on the robot's
starting position.
"""

import cv2
import numpy as np

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
        vote_cap: Saturation cap on each cell's vote accumulator. Votes are
            never decayed, so the map persists forever; each free/occupied
            observation moves a cell's accumulator toward ±cap, which bounds
            how long it takes to *flip* a cell (reacting to a new obstacle or
            a freed path) no matter how long the previous state was held.
        valid_min_brightness: Grey value below which a floor pixel is treated
            as unseen (excludes the black unmapped border and the dim 1-2px
            interpolation ring along the projection boundary).
        edge_margin_px: Erode the *edge* mask near the projection silhouette by
            this many pixels (dilated from the invalid/black region) so the
            Canny ring at the silhouette is not voted as occupied. Must be
            > inflation_radius or a residual ring remains. Only edges adjacent
            to the silhouette get suppressed; interior obstacle edges survive.
    """

    def __init__(
        self,
        grid_size_mm: float = 2000.0,
        cell_size_mm: float = 5.0,
        inflation_radius: int = 2,
        free_threshold: int = 60,
        edge_low: int = 50,
        edge_high: int = 150,
        vote_cap: int = 10,
        valid_min_brightness: int = 8,
        edge_margin_px: int = 3,
    ):
        self._grid_size_mm = grid_size_mm
        self._cell_size_mm = cell_size_mm
        self._grid_cells = int(grid_size_mm / cell_size_mm)

        # Vote accumulator: signed per-cell counter, clamped to [-cap, +cap].
        # positive = occupied, negative = free, 0 = unknown.
        self._votes: np.ndarray = np.zeros(
            (self._grid_cells, self._grid_cells), dtype=np.int32
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
        self._vote_cap = int(max(1, vote_cap))
        self._valid_min_brightness = valid_min_brightness
        self._edge_margin_px = int(max(0, edge_margin_px))

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
        self._votes[:] = 0
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

        # ---- Classify body-frame image ----------------------------------
        gray = cv2.cvtColor(floor_image, cv2.COLOR_BGR2GRAY)

        # Seen mask: black/dim areas (unmapped + interpolation ring along the
        # projection boundary) are not observations.
        seen = (gray >= self._valid_min_brightness).astype(np.uint8)

        # Edge detection → obstacle boundaries.
        edges = cv2.Canny(gray, self._edge_low, self._edge_high)

        # The projection silhouette is a hard black→floor transition, so Canny
        # fires all along it and would mark the whole border as an obstacle.
        # Mask out edges whose distance to the silhouette is <= edge_margin_px
        # (dilate the invalid region then zero edges under it) before inflation.
        # Interior object edges are untouched — they sit away from the silhouette.
        if self._edge_margin_px > 0:
            silhouette = (gray < self._valid_min_brightness).astype(np.uint8)
            silhouette = cv2.dilate(
                silhouette, np.ones((3, 3), np.uint8), iterations=self._edge_margin_px
            )
            edges[(silhouette > 0)] = 0

        if self._inflation_radius > 0:
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=self._inflation_radius)

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
        # Votes are never decayed — the map persists forever. Each observation
        # nudges the cell's signed counter toward ±cap; the cap is what bounds
        # flip latency (how soon a cell reacts to a new obstacle / freed path),
        # independent of how long the previous state was held.
        vote_delta = np.zeros_like(self._votes)
        np.add.at(
            vote_delta,
            (row[in_bounds], col[in_bounds]),
            is_occupied[in_bounds].astype(np.int32),
        )
        np.add.at(
            vote_delta,
            (row[in_bounds], col[in_bounds]),
            -is_free[in_bounds].astype(np.int32),
        )
        self._votes = np.clip(self._votes + vote_delta, -self._vote_cap, self._vote_cap)

        # --- Reclassify --------------------------------------------------
        self._grid[self._votes == 0] = UNKNOWN
        self._grid[self._votes > 0] = OCCUPIED
        self._grid[self._votes < 0] = FREE

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
