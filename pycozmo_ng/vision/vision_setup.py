"""Vision setup options — the one declaration of what vision setup configures.

``Brain.setup_vision`` takes this object, and the public entry points
(``Cozmo.setup_vision`` and ``Cozmo.vision.start``) accept the same fields as
keyword arguments. Adding a vision option means adding a field here — there is
no second parameter list to keep in sync, so the public API cannot silently
grow shorter than the internal one.
"""

from __future__ import annotations

from dataclasses import dataclass

from pycozmo_ng.vision.calibration.camera_calibration import CameraCalibration

__all__ = ["VisionSetup"]


@dataclass
class VisionSetup:
    """Everything vision setup can configure (frame gating, floor, occupancy, faces)."""

    #: CameraCalibration instance; None = retrieve it from the robot.
    camera_calibration: CameraCalibration | None = None

    #: Seconds to wait for that retrieval.
    calibration_timeout: float = 10.0

    #: Process every Nth camera frame (1 = every frame).
    process_every_n_frames: int = 1

    #: Undistort frames with the stored distortion coefficients (off by default).
    undistort: bool = False

    #: Build the bird's-eye floor projection.
    enable_floor_projection: bool = True

    #: World-frame (x, y) centre of the floor grid in mm.
    floor_center_mm: tuple[float, float] = (200.0, 0.0)

    #: (width, height) of the floor grid in mm.
    floor_size_mm: tuple[float, float] = (600.0, 600.0)

    #: Millimetres per floor pixel.
    mm_per_pixel: float = 5.0

    #: Accumulate the world-frame occupancy grid.
    enable_occupancy_map: bool = True

    #: Physical extent of the occupancy grid in mm.
    occupancy_grid_size_mm: float = 2000.0

    #: Millimetres per occupancy grid cell.
    occupancy_cell_size_mm: float = 50.0

    #: Dilate the edge mask by this many iterations.
    occupancy_inflation_radius: int = 2

    #: Saturation cap on each cell's vote accumulator.
    occupancy_vote_cap: int = 10

    #: Minimum brightness for a floor pixel to count as free.
    occupancy_valid_min_brightness: int = 8

    #: Suppress Canny edges within this many pixels of the image border.
    occupancy_edge_margin_px: int = 3

    #: Cut this many pixels off camera edges before floor projection.
    floor_edge_margin_px: int = 4

    #: Haar cascade face detection.
    enable_face_detection: bool = True

    #: Run face detection every Nth processed frame (higher = less CPU).
    face_detect_every_n_frames: int = 1

    #: Detect the charger's rectangular marker as a first-class object
    #: (own detection path, never a cube ID — COZMO-65).
    enable_charger_detection: bool = True
