"""
Floor projector — perspective-corrected top-down view from the robot camera.

Inversely projects each pixel of an output floor grid through the camera
intrinsic model and head-tilt rotation to find the corresponding source
pixel in the camera image.  A pre-computed lookup table (LUT) makes the
per-frame remapping fast (single ``cv2.remap`` call).

The output image is an orthographic top-down view of the floor, useful for
occupancy mapping and navigation.
"""

import cv2
import numpy as np

from pycozmo.logger import logger
from pycozmo.util.coords import CAMERA_TO_BODY_OFFSET, _head_tilt_matrix

__all__ = ["FloorProjector"]


class FloorProjector:
    """Projects camera images onto the floor plane (bird's-eye view).

    Args:
        camera_matrix: 3x3 intrinsic matrix K from CameraCalibration.
        image_width: Camera image width in pixels.
        image_height: Camera image height in pixels.
        floor_center_mm: World-frame (x, y) centre of the output grid (mm).
        floor_size_mm: (width, height) of the output grid in mm.
        mm_per_pixel: Millimetres per output pixel (lower = finer).
        edge_margin_px: Number of pixels to cut from the camera image edges.
            Edge pixels carry sensor/JPEG noise that gets magnified when
            re-sampled into the denser floor grid; masking them to black
            keeps the stretched-noise bands out of the occupancy map and the
            3D overlay (black is treated as transparent there).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        image_width: int = 320,
        image_height: int = 240,
        floor_center_mm: tuple[float, float] = (200.0, 0.0),
        floor_size_mm: tuple[float, float] = (600.0, 600.0),
        mm_per_pixel: float = 5.0,
        edge_margin_px: int = 4,
    ):
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self._K_inv = np.linalg.inv(self._K)
        self._image_width = image_width
        self._image_height = image_height

        self._floor_center = np.array(floor_center_mm, dtype=np.float64)
        self._floor_size = np.array(floor_size_mm, dtype=np.float64)
        self._mm_per_pixel = float(mm_per_pixel)
        self._edge_margin_px = int(max(0, edge_margin_px))

        self._out_w = int(np.ceil(self._floor_size[0] / self._mm_per_pixel))
        self._out_h = int(np.ceil(self._floor_size[1] / self._mm_per_pixel))

        # Pre-compute floor-grid coordinates in body frame (Z = 0).
        # Columns = body Y (left/right), rows = body X (forward/backward).
        # Forward at top (row 0), left at left (col 0).
        x_max = self._floor_center[0] + self._floor_size[0] / 2.0
        y_max = self._floor_center[1] + self._floor_size[1] / 2.0

        col_vals = y_max - (np.arange(self._out_w) + 0.5) * self._mm_per_pixel
        row_vals = x_max - (np.arange(self._out_h) + 0.5) * self._mm_per_pixel
        # meshgrid with 'ij' indexing: aa varies along rows, bb along columns
        bb, aa = np.meshgrid(col_vals, row_vals)  # both shape (out_h, out_w)

        self._floor_x = aa.ravel()  # body X (forward) — rows
        self._floor_y = bb.ravel()  # body Y (left) — columns
        self._floor_z = np.zeros_like(self._floor_x)  # body Z = 0

        self._remap_x: np.ndarray | None = None
        self._remap_y: np.ndarray | None = None
        self._last_head_angle: float | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def output_size(self) -> tuple[int, int]:
        """(width, height) of the floor-projected image in pixels."""
        return self._out_w, self._out_h

    @property
    def mm_per_pixel(self) -> float:
        return self._mm_per_pixel

    @property
    def floor_center(self) -> np.ndarray:
        """World-frame (x, y) centre of the floor grid (mm)."""
        return self._floor_center

    @property
    def floor_size(self) -> np.ndarray:
        """(width, height) of the floor grid in mm."""
        return self._floor_size

    @property
    def floor_x(self) -> np.ndarray:
        """Flattened body-X (forward) coordinates of the floor grid (mm)."""
        return self._floor_x

    @property
    def floor_y(self) -> np.ndarray:
        """Flattened body-Y (left) coordinates of the floor grid (mm)."""
        return self._floor_y

    # ------------------------------------------------------------------
    # LUT computation
    # ------------------------------------------------------------------

    def update_mapping(self, head_angle_rad: float) -> None:
        """(Re)compute the floor-to-camera remap LUT for the given head angle.

        The mapping depends only on the camera intrinsics and head tilt, both
        of which are constant for a given configuration, so this is typically
        called once at startup or whenever the head angle changes.
        """
        if (
            self._last_head_angle is not None
            and abs(head_angle_rad - self._last_head_angle) < 1e-4
        ):
            return  # no change

        R_body2cam = _head_tilt_matrix(head_angle_rad).T  # body -> camera

        # Camera origin in body frame
        cam_origin_body = CAMERA_TO_BODY_OFFSET  # [30, 0, 38]

        # Floor points relative to camera origin (body frame)
        dx = self._floor_x - cam_origin_body[0]
        dy = self._floor_y - cam_origin_body[1]
        dz = self._floor_z - cam_origin_body[2]  # always -38

        # Rotate into camera frame: P_cam = R^T @ (P_body - cam_origin)
        cam_x = R_body2cam[0, 0] * dx + R_body2cam[0, 1] * dy + R_body2cam[0, 2] * dz
        cam_y = R_body2cam[1, 0] * dx + R_body2cam[1, 1] * dy + R_body2cam[1, 2] * dz
        cam_z = R_body2cam[2, 0] * dx + R_body2cam[2, 1] * dy + R_body2cam[2, 2] * dz

        # Perspective projection (pixels).  Ignore points behind the camera.
        valid = cam_z > 1.0  # must be in front
        u = np.full(cam_z.shape, -1.0, dtype=np.float64)
        v = np.full(cam_z.shape, -1.0, dtype=np.float64)

        inv_z = np.where(valid, 1.0 / cam_z, 1.0)
        u[valid] = (self._K[0, 0] * cam_x[valid] * inv_z[valid]) + self._K[0, 2]
        v[valid] = (self._K[1, 1] * cam_y[valid] * inv_z[valid]) + self._K[1, 2]

        # Mark out-of-bounds or near-edge pixels (edge noise) as invalid
        m = self._edge_margin_px
        in_bounds = (
            valid
            & (u >= m)
            & (u < self._image_width - m)
            & (v >= m)
            & (v < self._image_height - m)
        )
        u[~in_bounds] = -1.0
        v[~in_bounds] = -1.0

        self._remap_x = u.reshape(self._out_h, self._out_w).astype(np.float32)
        self._remap_y = v.reshape(self._out_h, self._out_w).astype(np.float32)
        self._last_head_angle = head_angle_rad

        n_valid = int(np.sum(in_bounds))
        n_total = self._out_w * self._out_h
        logger.debug(
            f"Floor LUT updated: {self._out_w}x{self._out_h} "
            f"({n_valid}/{n_total} valid pixels, head={np.degrees(head_angle_rad):.1f} deg)"
        )

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(
        self,
        image: np.ndarray,
        head_angle_rad: float,
    ) -> np.ndarray:
        """Remap a camera image onto the floor plane.

        Args:
            image: Camera image as a numpy array (any channel count).
            head_angle_rad: Current head tilt angle (rad).

        Returns:
            Floor-projected image (BGR, same channel count as input).
        """
        self.update_mapping(head_angle_rad)

        if self._remap_x is None or self._remap_y is None:
            return np.zeros((self._out_h, self._out_w, 3), dtype=np.uint8)

        # cv2.remap needs float32 coordinate maps
        return cv2.remap(
            image,
            self._remap_x,
            self._remap_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    # ------------------------------------------------------------------
    # Coordinate helpers (for mapping world points ↔ floor pixels)
    # ------------------------------------------------------------------

    def world_to_floor_pixel(
        self,
        world_x: float,
        world_y: float,
        robot_x: float,
        robot_y: float,
        robot_yaw_rad: float,
    ) -> tuple[int, int] | None:
        """Convert a world-frame (X fwd, Y left) point to floor-pixel (col, row).

        Returns ``None`` if the point falls outside the floor grid.
        """
        # World -> body frame (translate + un-rotate)
        dx = world_x - robot_x
        dy = world_y - robot_y
        c, s = np.cos(-robot_yaw_rad), np.sin(-robot_yaw_rad)
        body_x = c * dx - s * dy
        body_y = s * dx + c * dy

        x_max = self._floor_center[0] + self._floor_size[0] / 2.0
        y_max = self._floor_center[1] + self._floor_size[1] / 2.0

        # col = body Y (left), row = body X reversed (forward = row 0)
        col = int((y_max - body_y) / self._mm_per_pixel)
        row = int((x_max - body_x) / self._mm_per_pixel)

        if 0 <= col < self._out_w and 0 <= row < self._out_h:
            return col, row
        return None

    def floor_pixel_to_body(self, col: int, row: int) -> tuple[float, float]:
        """Convert a floor-pixel (col, row) to body-frame (x, y) in mm."""
        x_max = self._floor_center[0] + self._floor_size[0] / 2.0
        y_max = self._floor_center[1] + self._floor_size[1] / 2.0
        bx = x_max - (row + 0.5) * self._mm_per_pixel
        by = y_max - (col + 0.5) * self._mm_per_pixel
        return bx, by

    # ------------------------------------------------------------------
    # Occupancy grid
    # ------------------------------------------------------------------

    def compute_occupancy_grid(
        self,
        floor_image: np.ndarray,
        free_threshold: int = 60,
    ) -> np.ndarray:
        """Derive a simple occupancy grid from a floor-projected image.

        Args:
            floor_image: BGR image from :meth:`project`.
            free_threshold: Average brightness above this → cell is free.

        Returns:
            2D int8 array: 0 = occupied, 1 = free, -1 = unknown (black/invalid).
        """
        gray = cv2.cvtColor(floor_image, cv2.COLOR_BGR2GRAY)
        grid = np.full(gray.shape, -1, dtype=np.int8)
        valid = gray > 0  # black pixels are unmapped
        grid[valid & (gray >= free_threshold)] = 1  # free
        grid[valid & (gray < free_threshold)] = 0  # occupied
        return grid
