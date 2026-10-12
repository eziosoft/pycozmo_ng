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
    """Everything vision setup can configure (frame gating, floor, occupancy, faces, overlays)."""

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

    #: Body-frame (forward, left) centre of the floor grid in mm.
    #: Moves with the robot — the floor camera is on the robot.
    floor_center_mm: tuple[float, float] = (200.0, 0.0)

    #: (forward_extent, left_extent) of the floor grid in mm.
    floor_size_mm: tuple[float, float] = (600.0, 600.0)

    #: Millimetres per floor pixel.
    mm_per_pixel: float = 5.0

    #: Accumulate the world-frame occupancy grid.
    enable_occupancy_map: bool = True

    #: Physical extent of the occupancy grid in mm.
    occupancy_grid_size_mm: float = 2000.0

    #: Millimetres per occupancy grid cell.
    occupancy_cell_size_mm: float = 50.0

    #: Dilate the obstacle contact line by this many 3x3 iterations.
    occupancy_inflation_radius: int = 2

    #: Occupied evidence added to each contact-line cell per frame (log-odds).
    #: One frame establishes a cell; larger values make an established cell
    #: survive more contradictory free frames before it clears.
    occupancy_logodds_hit: float = 4.0

    #: Free evidence subtracted from each free-floor cell per frame (log-odds).
    occupancy_logodds_miss: float = 0.8

    #: Clamp on the per-cell log-odds magnitude.
    occupancy_logodds_max: float = 5.0

    #: Per-frame multiplicative decay toward UNKNOWN; 0 = never (default).
    occupancy_logodds_decay: float = 0.0

    #: Lab distance above which a pixel counts as non-floor (obstacle).
    occupancy_floor_distance_threshold: float = 18.0

    #: Weight of the lightness difference in that distance; below 1 the map
    #: tolerates shadows (they darken the floor without changing its hue).
    occupancy_floor_light_weight: float = 0.25

    #: Per-frame EMA blend of the sampled near-robot strip into the floor model.
    occupancy_floor_model_ema: float = 0.3

    #: Minimum brightness for a floor pixel to count as seen.
    occupancy_valid_min_brightness: int = 8

    #: Erode the seen mask by this many pixels so the dim interpolation ring
    #: at the projection silhouette cannot vote an obstacle ring.
    occupancy_silhouette_margin_px: int = 3

    #: Cut this many pixels off camera edges before floor projection.
    floor_edge_margin_px: int = 4

    #: Haar cascade face detection.
    enable_face_detection: bool = True

    #: Run face detection every Nth processed frame (higher = less CPU).
    face_detect_every_n_frames: int = 1

    #: Detect the charger's rectangular marker as a first-class object
    #: (own detection path, never a cube ID — COZMO-65).
    enable_charger_detection: bool = True

    #: Detect standard 4x4 ArUco markers (own detection path, marker ID is
    #: the identity — never a cube ID). Physical per-ID sizes come from the
    #: JSON config file (PYCOZMO_NG_ARUCO_CONFIG), not from here.
    enable_aruco_detection: bool = True

    #: Draw ground distance rings (range circles) in perspective on the
    #: annotated frame — the cue for a human viewer and for an LLM reading
    #: get_frame(annotated=True).
    enable_distance_rings: bool = True

    #: Band radii in mm, centred on the body-frame ground origin.
    #: Three geometric steps (250 / 500 / 1000) — enough range cues
    #: without crowding a 320×240 frame.
    distance_ring_radii_mm: tuple[int, ...] = (250, 500, 1000)
