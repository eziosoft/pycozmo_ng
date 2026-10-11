"""Floor pipeline — orchestrates floor projection and occupancy mapping.

Wraps :class:`FloorProjector` and :class:`OccupancyMap` into a single
processing stage called by :class:`AnnotatedVisionProcessor`.
"""

import numpy as np

from pycozmo_ng.logger import logger
from pycozmo_ng.vision.calibration.camera_calibration import CameraCalibration
from pycozmo_ng.vision.occupancy_map import OccupancyMap
from pycozmo_ng.vision.occupancy_map.floor_projector import FloorProjector

__all__ = ["FloorPipeline"]


class FloorPipeline:
    """Orchestrates floor projection and occupancy map accumulation.

    Owns a :class:`FloorProjector` and an optional :class:`OccupancyMap`.
    Call :meth:`process` with the undistorted camera image to produce
    the bird's-eye floor view and world-frame occupancy grid.

    Args:
        camera_calibration: Camera intrinsics for the projector.
        enable_floor_projection: Create the floor projector.
        floor_center_mm: World-frame (x, y) centre of the floor grid (mm).
        floor_size_mm: (width, height) of the floor grid in mm.
        mm_per_pixel: Millimetres per output pixel.
        enable_occupancy_map: Enable world-frame occupancy accumulation.
        occupancy_grid_size_mm: Physical extent of the occupancy grid (mm).
        occupancy_cell_size_mm: Millimetres per occupancy grid cell.
        occupancy_inflation_radius: Dilate the obstacle contact line by this
            many 3x3 iterations.
        occupancy_logodds_hit: Occupied evidence per contact-line cell per
            frame (log-odds).
        occupancy_logodds_miss: Free evidence per free-floor cell per frame
            (log-odds).
        occupancy_logodds_max: Clamp on the per-cell log-odds magnitude.
        occupancy_logodds_decay: Per-frame multiplicative decay toward
            UNKNOWN; 0 = never (default).
        occupancy_floor_distance_threshold: Lab distance above which a pixel
            counts as non-floor (obstacle).
        occupancy_floor_light_weight: Weight of the lightness difference in
            that distance; below 1 the map tolerates shadows.
        occupancy_floor_model_ema: Per-frame EMA blend of the sampled
            near-robot strip into the adaptive floor model.
        occupancy_valid_min_brightness: Minimum brightness for a floor pixel
            to count as seen.
        occupancy_silhouette_margin_px: Erode the seen mask by this many
            pixels so the projection's interpolation ring cannot vote an
            obstacle ring.
        floor_edge_margin_px: Cut this many pixels off camera edges before
            projecting.
    """

    def __init__(
        self,
        camera_calibration: CameraCalibration,
        enable_floor_projection: bool = True,
        floor_center_mm: tuple[float, float] = (200.0, 0.0),
        floor_size_mm: tuple[float, float] = (600.0, 600.0),
        mm_per_pixel: float = 5.0,
        enable_occupancy_map: bool = True,
        occupancy_grid_size_mm: float = 2000.0,
        occupancy_cell_size_mm: float = 50.0,
        occupancy_inflation_radius: int = 2,
        occupancy_logodds_hit: float = 4.0,
        occupancy_logodds_miss: float = 0.8,
        occupancy_logodds_max: float = 5.0,
        occupancy_logodds_decay: float = 0.0,
        occupancy_floor_distance_threshold: float = 18.0,
        occupancy_floor_light_weight: float = 0.25,
        occupancy_floor_model_ema: float = 0.3,
        occupancy_valid_min_brightness: int = 8,
        occupancy_silhouette_margin_px: int = 3,
        floor_edge_margin_px: int = 4,
    ):
        # Floor projector
        self.floor_projector: FloorProjector | None = None
        if enable_floor_projection:
            self.floor_projector = FloorProjector(
                camera_calibration.camera_matrix,
                camera_calibration.image_width,
                camera_calibration.image_height,
                floor_center_mm=floor_center_mm,
                floor_size_mm=floor_size_mm,
                mm_per_pixel=mm_per_pixel,
                edge_margin_px=floor_edge_margin_px,
            )
            logger.info(f"Floor projection enabled: {floor_size_mm[0]}x{floor_size_mm[1]}mm @ {mm_per_pixel} mm/px")

        # Occupancy map
        self.occupancy_map: OccupancyMap | None = None
        if enable_floor_projection and enable_occupancy_map:
            self.occupancy_map = OccupancyMap(
                grid_size_mm=occupancy_grid_size_mm,
                cell_size_mm=occupancy_cell_size_mm,
                inflation_radius=occupancy_inflation_radius,
                logodds_hit=occupancy_logodds_hit,
                logodds_miss=occupancy_logodds_miss,
                logodds_max=occupancy_logodds_max,
                logodds_decay=occupancy_logodds_decay,
                floor_distance_threshold=occupancy_floor_distance_threshold,
                floor_light_weight=occupancy_floor_light_weight,
                floor_model_ema=occupancy_floor_model_ema,
                valid_min_brightness=occupancy_valid_min_brightness,
                silhouette_margin_px=occupancy_silhouette_margin_px,
            )

    def process(
        self,
        img_bgr: np.ndarray,
        head_angle_rad: float,
        robot_pose,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Project the camera image onto the floor and accumulate occupancy.

        Args:
            img_bgr: Camera image as BGR numpy array (already converted by caller).
            head_angle_rad: Current head tilt angle (rad).
            robot_pose: ``cli.pose`` — robot world-frame pose (may be ``None``).

        Returns:
            ``(floor_image, occupancy_grid, occupancy_map_image)`` — all may
            be ``None`` if the floor projector is disabled.
        """
        if self.floor_projector is None:
            return None, None, None

        floor_img = self.floor_projector.project(img_bgr, head_angle_rad)
        occ_grid = self.floor_projector.compute_occupancy_grid(floor_img)

        # Accumulate into world-frame occupancy map
        occ_map_image = None
        if self.occupancy_map is not None and robot_pose is not None:
            self.occupancy_map.update(
                floor_img,
                self.floor_projector.floor_x,
                self.floor_projector.floor_y,
                robot_pose.position.x,
                robot_pose.position.y,
                robot_pose.rotation.angle_z.radians,
            )
            occ_map_image = self.occupancy_map.to_color_image()

        return floor_img, occ_grid, occ_map_image

    @property
    def enabled(self) -> bool:
        """True if floor projection is enabled."""
        return self.floor_projector is not None

    @property
    def occupancy_enabled(self) -> bool:
        """True if occupancy mapping is enabled."""
        return self.occupancy_map is not None
