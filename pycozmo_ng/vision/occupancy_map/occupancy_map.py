"""Occupancy map — accumulates body-frame floor observations into a world-frame grid.

Takes bird's-eye floor images from :class:`~pycozmo_ng.vision.floor_projector.FloorProjector`
and robot pose, classifies each valid pixel as floor / non-floor against an
**adaptive floor color model**, extracts the **contact line** of every non-floor
region (where it touches the floor), and accumulates **log-odds** evidence in a
fixed-size world-frame grid centered on the robot's starting position.

Why not brightness + Canny edges (the previous method):

- Shadows and floor texture (tile grout, carpet pattern) produce edges → false
  obstacles; dark obstacles on a bright floor produce none → missed obstacles.
  Brightness heuristics fire on the wrong things in both directions.
- Every projected pixel is flattened to z=0, so a projected obstacle silhouette
  smears *past* its true base — voting the whole silhouette puts occupied
  cells too far away, at the worst-resolved (far) end of the grid.

Instead:

1. **Adaptive floor model** — the strip of floor directly in front of the
   robot (body X in ``strip_x_mm``, |Y| ≤ ``strip_y_max_mm``) is trusted floor:
   the robot just drove over it. Its per-channel Lab median (EMA-smoothed
   across frames) *is* the floor color, so any floor — bright, dark, wooden —
   works with no fixed brightness threshold.
2. **Shadow tolerance** — classification distance down-weights L
   (``floor_light_weight``) and leans on chroma (a, b): a shadow darkens the
   floor but leaves its hue, so it classifies as floor, not obstacle.
3. **Contact line** — of a non-floor region only its *near edge* (the pixel
   row closest to the robot) is where the object meets the floor, the one
   part of the projection that lands at the true ground position. Only that
   band (widened by ``inflation_radius``) votes occupied.
4. **Log-odds** — one symmetric evidence step per cell per frame:
   ``-logodds_miss`` for free floor, ``+logodds_hit`` for contact-line cells.
   Sign classifies (positive = occupied, negative = free, 0 = unknown);
   ``logodds_hit`` > several ``logodds_miss`` keeps the old damped-clearing
   behavior (a new obstacle appears in one frame, an established one survives
   several contradictory frames). Decay is optional and off by default (a
   decaying map fades to UNKNOWN outside the FOV — rejected live, see
   COZMO-DOC-9).

The public grid contract is unchanged: ``grid`` is int8 with
``FREE``/``OCCUPIED``/``UNKNOWN``, which is what ``AStarPlanner`` consumes.
"""

import cv2
import numpy as np

from pycozmo_ng.logger import logger

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
        inflation_radius: Dilate the contact line by this many 3x3 iterations
            (widens the occupied band at an obstacle's base — a safety margin
            for planning, not a detection change).
        logodds_hit: Occupied evidence added to each contact-line cell per
            frame. One frame classifies a cell occupied (positive); an
            established cell survives roughly ``ceil(logodds_hit /
            logodds_miss)`` consecutive free frames before it clears.
        logodds_miss: Free evidence subtracted from each free-floor cell per
            frame (at most one step per cell per frame, independent of pixel
            count) — the damping that keeps transient frames (blur, a passing
            shadow) from flipping the grid under A*.
        logodds_max: Clamp on the accumulator magnitude.
        logodds_decay: Per-frame multiplicative decay toward zero; 0 = never
            decay (default — the map persists outside the FOV).
        floor_distance_threshold: Lab distance above which a pixel is
            non-floor (obstacle candidate).
        floor_light_weight: Weight of the L (lightness) difference in that
            distance; < 1 buys shadow tolerance.
        floor_model_ema: Per-frame EMA blend of the sampled strip median into
            the floor model (1 = adopt the latest strip outright).
        strip_x_mm: Body-frame X band (forward, mm) sampled as trusted floor.
        strip_y_max_mm: Absolute body-frame Y limit for that strip.
        valid_min_brightness: Grey value below which a floor pixel is treated
            as unseen (excludes the black unmapped border and the dim 1-2px
            interpolation ring along the projection boundary).
        silhouette_margin_px: Erode the seen mask by this many pixels so the
            dim interpolation ring at the projection silhouette drops out
            before classification (it would otherwise classify as non-floor
            and vote a ring around the whole visible area). Interior obstacle
            boundaries are untouched — they sit away from the silhouette.
    """

    def __init__(
        self,
        grid_size_mm: float = 2000.0,
        cell_size_mm: float = 50.0,
        inflation_radius: int = 2,
        logodds_hit: float = 4.0,
        logodds_miss: float = 0.8,
        logodds_max: float = 5.0,
        logodds_decay: float = 0.0,
        floor_distance_threshold: float = 18.0,
        floor_light_weight: float = 0.25,
        floor_model_ema: float = 0.3,
        strip_x_mm: tuple[float, float] = (60.0, 300.0),
        strip_y_max_mm: float = 150.0,
        valid_min_brightness: int = 8,
        silhouette_margin_px: int = 3,
    ):
        self._grid_size_mm = grid_size_mm
        self._cell_size_mm = cell_size_mm
        self._grid_cells = int(grid_size_mm / cell_size_mm)

        # Log-odds accumulator: positive = occupied evidence, negative = free,
        # 0 = unknown (never observed).
        self._logodds: np.ndarray = np.zeros((self._grid_cells, self._grid_cells), dtype=np.float32)

        # Classified grid: -1 = unknown, 0 = occupied, 1 = free
        self._grid: np.ndarray = np.full((self._grid_cells, self._grid_cells), UNKNOWN, dtype=np.int8)

        # World origin (set on first update from robot position)
        self._origin_x: float = 0.0
        self._origin_y: float = 0.0
        self._origin_set: bool = False

        # Adaptive floor model (Lab, float32 (3,)); None until first strip sample.
        self._floor_lab: np.ndarray | None = None

        # Parameters
        self._inflation_radius = inflation_radius
        self._logodds_hit = float(logodds_hit)
        self._logodds_miss = float(logodds_miss)
        self._logodds_max = float(logodds_max)
        self._logodds_decay = float(logodds_decay)
        self._floor_distance_threshold = float(floor_distance_threshold)
        self._floor_light_weight = float(floor_light_weight)
        self._floor_model_ema = float(floor_model_ema)
        self._strip_x = (float(strip_x_mm[0]), float(strip_x_mm[1]))
        self._strip_y_max = float(strip_y_max_mm)
        self._valid_min_brightness = valid_min_brightness
        self._silhouette_margin_px = int(max(0, silhouette_margin_px))

        logger.info(
            f"OccupancyMap: {self._grid_cells}x{self._grid_cells} cells, "
            f"{grid_size_mm}mm @ {cell_size_mm} mm/cell, "
            f"inflation={inflation_radius}, "
            f"logodds hit/miss/max={logodds_hit}/{logodds_miss}/{logodds_max}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all evidence, the floor model, and the grid."""
        self._logodds[:] = 0.0
        self._grid[:] = UNKNOWN
        self._floor_lab = None
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
        """Process one frame: segment, extract contact lines, accumulate, reclassify.

        Args:
            floor_image: Bird's-eye BGR image from FloorProjector.project().
            body_x: 1-D array of body-frame X coords (mm) for each pixel,
                same layout as the floor image (row-major, shape = H*W).
            body_y: 1-D array of body-frame Y coords (mm).
            robot_x: Robot world-frame X position (mm).
            robot_y: Robot world-frame Y position (mm).
            robot_yaw_rad: Robot yaw about world Z (radians).
        """
        if not self._origin_set:
            self._origin_x = robot_x
            self._origin_y = robot_y
            self._origin_set = True

        h, w = floor_image.shape[:2]
        bx = body_x.reshape(h, w)
        by = body_y.reshape(h, w)

        # ---- Seen mask: black/dim areas (unmapped + interpolation ring) are
        # not observations. Eroding drops the dim ring at the silhouette so it
        # cannot classify as non-floor.
        gray = cv2.cvtColor(floor_image, cv2.COLOR_BGR2GRAY)
        seen: np.ndarray = (gray >= self._valid_min_brightness).astype(np.uint8)
        if self._silhouette_margin_px > 0:
            seen = cv2.erode(seen, np.ones((3, 3), np.uint8), iterations=self._silhouette_margin_px)

        is_floor = self._classify_floor(floor_image, seen, bx, by)

        # Free: every valid floor pixel. Occupied: only the contact line of
        # non-floor regions — their base, the one band that projects to the
        # true ground position (see _contact_line).
        non_floor = (seen > 0) & ~is_floor
        occupied = self._contact_line(non_floor) > 0

        # Masks (flattened, row-major like body_x / body_y)
        is_free = ((seen > 0) & is_floor).ravel()
        is_occupied = occupied.ravel()

        # --- Body → world transform -------------------------------------
        c, s = np.cos(robot_yaw_rad), np.sin(robot_yaw_rad)
        world_x = body_x * c - body_y * s + robot_x
        world_y = body_x * s + body_y * c + robot_y

        # --- World → grid indices ----------------------------------------
        half = self._grid_size_mm / 2.0
        col = ((world_x - self._origin_x + half) / self._cell_size_mm).astype(np.int32)
        row = ((world_y - self._origin_y + half) / self._cell_size_mm).astype(np.int32)

        in_bounds = (col >= 0) & (col < self._grid_cells) & (row >= 0) & (row < self._grid_cells)

        # --- Accumulate log-odds -----------------------------------------
        # One symmetric evidence step per cell per frame (not per pixel), so a
        # large free area does not out-vote a small obstacle, and evidence is
        # comparable in both directions.
        row_in = row[in_bounds]
        col_in = col[in_bounds]
        flat_idx = row_in * self._grid_cells + col_in

        delta = np.zeros(self._grid_cells * self._grid_cells, dtype=np.float32)
        occ_mask = is_occupied[in_bounds]
        free_mask = is_free[in_bounds]
        if np.any(occ_mask):
            occ_cells = np.bincount(flat_idx[occ_mask], minlength=delta.size)
            delta += (occ_cells > 0).astype(np.float32) * self._logodds_hit
        if np.any(free_mask):
            free_cells = np.bincount(flat_idx[free_mask], minlength=delta.size)
            delta -= (free_cells > 0).astype(np.float32) * self._logodds_miss

        if self._logodds_decay > 0.0:
            self._logodds *= self._logodds_decay
            # Epsilon reset: fully-decayed cells return to UNKNOWN instead of
            # asymptoting just above zero forever.
            self._logodds[np.abs(self._logodds) < 1e-3] = 0.0

        np.clip(
            self._logodds + delta.reshape(self._logodds.shape),
            -self._logodds_max,
            self._logodds_max,
            out=self._logodds,
        )

        # --- Reclassify (sign of the log-odds) ---------------------------
        self._grid[self._logodds == 0] = UNKNOWN
        self._grid[self._logodds > 0] = OCCUPIED
        self._grid[self._logodds < 0] = FREE

    # ------------------------------------------------------------------
    # Floor segmentation
    # ------------------------------------------------------------------

    def _classify_floor(
        self,
        floor_image: np.ndarray,
        seen: np.ndarray,
        bx: np.ndarray,
        by: np.ndarray,
    ) -> np.ndarray:
        """Floor / non-floor classification against the adaptive floor model.

        Refreshes the model from the trusted near-robot strip, then labels
        every pixel by its Lab distance to the model. Returns a boolean array
        (same shape as the image) of floor pixels; when no model exists yet
        (first frames, strip out of view) nothing is floor — no evidence is
        produced that frame.
        """
        lab = cv2.cvtColor(floor_image, cv2.COLOR_BGR2Lab).astype(np.float32)

        strip = (seen > 0) & (bx >= self._strip_x[0]) & (bx <= self._strip_x[1]) & (np.abs(by) <= self._strip_y_max)
        if np.any(strip):
            median = np.median(lab[strip], axis=0)
            if self._floor_lab is None:
                self._floor_lab = median
            else:
                a = self._floor_model_ema
                self._floor_lab = (1.0 - a) * self._floor_lab + a * median

        if self._floor_lab is None:
            return np.zeros(bx.shape, dtype=bool)

        d_l = (lab[:, :, 0] - self._floor_lab[0]) * self._floor_light_weight
        d_a = lab[:, :, 1] - self._floor_lab[1]
        d_b = lab[:, :, 2] - self._floor_lab[2]
        dist = np.sqrt(d_l * d_l + d_a * d_a + d_b * d_b)
        return dist <= self._floor_distance_threshold

    def _contact_line(self, non_floor: np.ndarray) -> np.ndarray:
        """Near-edge pixel band of every non-floor region (its floor contact).

        Rows increase toward the robot (row 0 = farthest forward), so the
        contact of a region is the last row of each column run: the pixel
        whose neighbor one row closer to the robot is floor. This is the only
        part of a projected obstacle that lands at its true ground position —
        the rest of the silhouette smears past the base (z is flattened to 0).
        """
        contact: np.ndarray = np.zeros_like(non_floor, dtype=np.uint8)
        contact[:-1, :] = non_floor[:-1, :] & ~non_floor[1:, :]
        contact[-1, :] = non_floor[-1, :]

        if self._inflation_radius > 0:
            contact = cv2.dilate(contact, np.ones((3, 3), np.uint8), iterations=self._inflation_radius)
        return contact

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
        # Single combined flip+transpose (avoids 3 intermediate arrays)
        return np.ascontiguousarray(img[::-1, ::-1].transpose(1, 0, 2))

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

    @property
    def floor_model(self) -> np.ndarray | None:
        """Current adaptive floor model as a Lab triplet, or None."""
        return self._floor_lab
