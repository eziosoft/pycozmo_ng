"""
Coordinate-frame transforms for the world map.

Defines the single set of frame conventions and helpers used across the
localization stack (WorldMap, web viewer serialization). Keeping all frame
math here means the world→scene change of basis and the camera→body offset
live in exactly one place, so they can be verified and tuned on the robot once.

Frame conventions:

* **Camera** (OpenCV): X right, Y down, Z forward. Raw solvePnP output.
* **Body** (robot-relative, = Anki axes): X forward, Y left, Z up. Origin is
  the front-centre ground point directly under the camera.
* **World** (Anki): X forward, Y left, Z up, mm. The canonical storage frame —
  matches ``cli.pose`` (X = forward, Y = left, Z = up; yaw about Z, CCW positive).
  Origin = current Cozmo localization origin.
* **Scene** (Three.js): X right, Y up, Z forward. Only used for serializing to
  the web viewer, which is a pure display. The world→scene change of basis is a
  proper rotation (det = +1), so chirality is preserved and turns/circles render
  with the correct handedness.

> NOTE on the Anki convention: verified from ``pycozmo_ng/util.py`` — ``Matrix44``
> stores forward on the X row, left on the Y row, up on the Z row, and
> ``Pose.define_pose_relative_this`` applies ``cos`` to the X coordinate and
> ``sin`` to Y. So **X is forward, Y is left, Z is up** (NOT "X right, Y
> forward" as some docs say). Getting this wrong rotates everything 90°.
"""

import numpy as np

__all__ = [
    "CAMERA_TO_BODY_OFFSET",
    "WORLD_TO_SCENE_MATRIX",
    "R_yaw",
    "body_to_world",
    "camera_to_body",
    "camera_to_world",
    "world_to_scene_point",
    "world_to_scene_rotation",
    "world_yaw_to_scene",
]


# Camera-to-body offset in body frame (mm). Camera sits 30 mm back from
# the front edge, centred laterally, 38 mm above the ground. (COZMO-28,
# user-measured: 90 L x 56 W x 70 H, cam 38 mm high, 30 mm setback)
CAMERA_TO_BODY_OFFSET: np.ndarray = np.array([30.0, 0.0, 38.0], dtype=np.float64)

# World→scene change-of-basis matrix (a proper **rotation**, det = +1; see
# ``world_to_scene_point`` for why it must not be a reflection). Its columns
# are the scene coordinates of the world basis vectors:
#   world X (forward) -> scene Z (forward)   => col1 = (0, 0, 1)
#   world Y (left)    -> scene X (left)      => col2 = (1, 0, 0)
#   world Z (up)      -> scene Y (up)        => col3 = (0, 1, 0)
# So scene = M @ world = (y_left, z_up, x_forward). det = +1 (orientation
# preserving): mapping one right-handed frame to another right-handed frame
# must be a rotation, NOT a reflection. A reflection (det = -1) would flip
# chirality and make every turn/circle render mirrored (a real-world
# clockwise drive-out looks anticlockwise in the top-down 3D view).
WORLD_TO_SCENE_MATRIX: np.ndarray = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def R_yaw(yaw_rad: float) -> np.ndarray:
    """Rotation about world Z (up) by ``yaw_rad`` (Anki yaw, forward = +X)."""
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _head_tilt_matrix(head_angle_rad: float) -> np.ndarray:
    """Camera→body rotation including head pitch (Anki body axes).

    Columns are where each camera basis vector maps in body coordinates:

    * camera X (right)   -> body (0, -1, 0)            (body right = -Y)
    * camera Y (down)    -> body ( sinθ, 0, -cosθ)
    * camera Z (forward) -> body ( cosθ, 0,  sinθ)

    At level head (θ = 0) this maps camera-forward (+Zc) to body-forward (+X),
    camera-down (+Yc) to body-down (-Z), camera-right (+Xc) to body-right (-Y).
    As the head tilts up (positive θ) the optical axis gains an up component.
    """
    c = np.cos(head_angle_rad)
    s = np.sin(head_angle_rad)
    return np.array([[0.0, s, c], [-1.0, 0.0, 0.0], [0.0, -c, s]], dtype=np.float64)


def camera_to_body(
    position_cam: np.ndarray,
    rotation_cam: np.ndarray,
    head_angle_rad: float,
    camera_lateral_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a pose from camera frame to body frame.

    Applies the head-tilt rotation (camera axes -> body axes) then adds the
    camera-to-body offset. ``rotation_cam`` is the 3x3 rotation matrix that
    maps cube-frame basis vectors to camera-frame vectors; the returned
    rotation maps them to body-frame vectors.

    Args:
        position_cam: 3-vector position in camera frame (mm).
        rotation_cam: 3x3 rotation matrix in camera frame.
        head_angle_rad: head pitch (radians, positive = looking up).
        camera_lateral_offset_mm: Measured lateral offset of camera from
            robot centre (mm). Positive = camera is to the left of centre,
            so objects appear shifted right. Detected via robot center
            calibration (black circles).

    Returns:
        (position_body, rotation_body).
    """
    r_head = _head_tilt_matrix(head_angle_rad)
    position_body = r_head @ np.asarray(position_cam, dtype=np.float64)
    position_body = position_body + CAMERA_TO_BODY_OFFSET
    # Apply measured lateral correction (body Y axis = left)
    position_body[1] += camera_lateral_offset_mm
    rotation_body = r_head @ np.asarray(rotation_cam, dtype=np.float64)
    return position_body, rotation_body


def body_to_world(
    position_body: np.ndarray,
    rotation_body: np.ndarray,
    robot_position: np.ndarray,
    robot_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a pose from body frame to world frame.

    Rotates by the robot's yaw about world Z, then translates by the robot's
    world position.

    Args:
        position_body: 3-vector position in body frame (mm).
        rotation_body: 3x3 rotation matrix in body frame.
        robot_position: robot world position (mm, Anki frame).
        robot_yaw_rad: robot yaw about world Z (radians).

    Returns:
        (position_world, rotation_world).
    """
    r_yaw = R_yaw(robot_yaw_rad)
    position_world = r_yaw @ np.asarray(position_body, dtype=np.float64)
    position_world = position_world + np.asarray(robot_position, dtype=np.float64)
    rotation_world = r_yaw @ np.asarray(rotation_body, dtype=np.float64)
    return position_world, rotation_world


def camera_to_world(
    position_cam: np.ndarray,
    rotation_cam: np.ndarray,
    head_angle_rad: float,
    robot_position: np.ndarray,
    robot_yaw_rad: float,
    camera_lateral_offset_mm: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: camera frame -> world frame in one step."""
    p_body, r_body = camera_to_body(position_cam, rotation_cam, head_angle_rad, camera_lateral_offset_mm)
    return body_to_world(p_body, r_body, robot_position, robot_yaw_rad)


def world_to_scene_point(position_world: np.ndarray) -> np.ndarray:
    """Map a world-frame point (X fwd, Y left, Z up) to scene (X right, Y up, Z fwd)."""
    p = np.asarray(position_world, dtype=np.float64)
    return np.array([p[1], p[2], p[0]], dtype=np.float64)


def scene_to_world_point(position_scene: np.ndarray) -> np.ndarray:
    """Map a scene-frame point (X right, Y up, Z fwd) back to world (X fwd, Y left, Z up)."""
    p = np.asarray(position_scene, dtype=np.float64)
    return np.array([p[2], p[0], p[1]], dtype=np.float64)


def world_to_scene_rotation(rotation_world: np.ndarray) -> np.ndarray:
    """Map a world-frame 3x3 rotation to scene frame via change of basis."""
    m = WORLD_TO_SCENE_MATRIX
    return m @ np.asarray(rotation_world, dtype=np.float64) @ m


def world_yaw_to_scene(yaw_rad: float) -> float:
    """Map a world-frame yaw (about world Z) to a scene rotation.y.

    The robot's world forward at yaw ``ψ`` is ``(cos ψ, sin ψ, 0)`` (forward is
    +X, rotated about Z). Under the world→scene rotation (det = +1, see
    ``WORLD_TO_SCENE_MATRIX``) it becomes the scene vector ``(sin ψ, 0, cos ψ)``.
    A Three.js ``rotation.y = θ`` rotates +Z to ``(sin θ, 0, cos θ)``, so
    ``θ = ψ`` — the yaw passes through unchanged, because the basis change is
    orientation-preserving. (It is NOT negated.)
    """
    return float(yaw_rad)
