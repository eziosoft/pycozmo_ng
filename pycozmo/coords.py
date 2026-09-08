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
  the web viewer, which is a pure display.

> NOTE on the Anki convention: verified from ``pycozmo/util.py`` — ``Matrix44``
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


# Camera-to-body offset in body frame (mm). Camera sits on the front-centre,
# 40 mm above the ground, with no horizontal offset. (design decision)
CAMERA_TO_BODY_OFFSET: np.ndarray = np.array([0.0, 0.0, 40.0], dtype=np.float64)

# World→scene change-of-basis matrix (orthogonal). Its columns are the scene
# coordinates of the world basis vectors:
#   world X (forward) -> scene Z (forward)   => col1 = (0, 0, 1)
#   world Y (left)    -> scene -X (left)     => col2 = (-1, 0, 0)
#   world Z (up)      -> scene Y (up)        => col3 = (0, 1, 0)
# So scene = M @ world = (-y_left, z_up, x_forward). det = -1 (a reflection,
# as expected when mapping a right-handed frame onto the right-handed scene).
WORLD_TO_SCENE_MATRIX: np.ndarray = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def R_yaw(yaw_rad: float) -> np.ndarray:
    """Rotation about world Z (up) by ``yaw_rad`` (Anki yaw, forward = +X)."""
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _head_tilt_matrix(head_angle_rad: float) -> np.ndarray:
    """Camera→body rotation including head pitch (Anki body axes).

    Columns are where each camera basis vector maps in body coordinates:

    * camera X (right)   -> body (0, -1, 0)            (body right = -Y)
    * camera Y (down)    -> body (-sinθ, 0, -cosθ)
    * camera Z (forward) -> body ( cosθ, 0,  sinθ)

    At level head (θ = 0) this maps camera-forward (+Zc) to body-forward (+X),
    camera-down (+Yc) to body-down (-Z), camera-right (+Xc) to body-right (-Y).
    As the head tilts up (positive θ) the optical axis gains an up component.
    """
    c = np.cos(head_angle_rad)
    s = np.sin(head_angle_rad)
    return np.array([[0.0, -s, c], [-1.0, 0.0, 0.0], [0.0, -c, s]], dtype=np.float64)


def camera_to_body(
    position_cam: np.ndarray,
    rotation_cam: np.ndarray,
    head_angle_rad: float,
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

    Returns:
        (position_body, rotation_body).
    """
    r_head = _head_tilt_matrix(head_angle_rad)
    position_body = r_head @ np.asarray(position_cam, dtype=np.float64)
    position_body = position_body + CAMERA_TO_BODY_OFFSET
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
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: camera frame -> world frame in one step."""
    p_body, r_body = camera_to_body(position_cam, rotation_cam, head_angle_rad)
    return body_to_world(p_body, r_body, robot_position, robot_yaw_rad)


def world_to_scene_point(position_world: np.ndarray) -> np.ndarray:
    """Map a world-frame point (X fwd, Y left, Z up) to scene (X right, Y up, Z fwd)."""
    p = np.asarray(position_world, dtype=np.float64)
    return np.array([-p[1], p[2], p[0]], dtype=np.float64)


def world_to_scene_rotation(rotation_world: np.ndarray) -> np.ndarray:
    """Map a world-frame 3x3 rotation to scene frame via change of basis."""
    m = WORLD_TO_SCENE_MATRIX
    return m @ np.asarray(rotation_world, dtype=np.float64) @ m


def world_yaw_to_scene(yaw_rad: float) -> float:
    """Map a world-frame yaw (about world Z) to a scene rotation.y.

    The robot's world forward at yaw ``ψ`` is ``(cos ψ, sin ψ, 0)`` (forward is
    +X, rotated about Z). Under the world→scene change of basis it becomes the
    scene vector ``(-sin ψ, 0, cos ψ)``. A Three.js ``rotation.y = θ`` rotates
    +Z to ``(sin θ, 0, cos θ)``, so ``sin θ = -sin ψ``, ``cos θ = cos ψ``, i.e.
    ``θ = -ψ`` (the reflection mirrors the yaw).
    """
    return -float(yaw_rad)
