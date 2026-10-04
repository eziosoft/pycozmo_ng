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
        occupancy_inflation_radius: Dilate edge mask by this many iterations.
        occupancy_vote_cap: Saturation cap on each cell's vote accumulator.
        occupancy_valid_min_brightness: Minimum brightness for a floor pixel
            to count as seen.
        occupancy_edge_margin_px: Suppress Canny edges within this many
            pixels of the projection silhouette.
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
        occupancy_vote_cap: int = 10,
        occupancy_valid_min_brightness: int = 8,
        occupancy_edge_margin_px: int = 3,
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
                vote_cap=occupancy_vote_cap,
                valid_min_brightness=occupancy_valid_min_brightness,
                edge_margin_px=occupancy_edge_margin_px,
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
