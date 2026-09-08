"""
Coordinate-frame transforms for the world map.

Defines the single set of frame conventions and helpers used across the
localization stack (WorldMap, web viewer serialization). Keeping all frame
math here means the world→scene reflection and the camera→body offset live in
exactly one place, so they can be verified and tuned on the robot once.

Frames (all right-handed unless noted):

* **Camera** (OpenCV): X right, Y down, Z forward. Raw solvePnP output.
* **Body**: X right, Y forward, Z up. Origin = front-centre ground point
  directly under the camera.
* **World** (Anki): X right, Y forward, Z up, mm. The canonical storage frame.
  Origin = current Cozmo localization origin.
* **Scene** (Three.js, right-handed): X right, Y up, Z forward. Only used for
  serializing to the web viewer, which is a pure display.

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

# World→scene change-of-basis matrix (orthogonal). Maps a world vector to a
# scene vector: scene_x = -world_x, scene_y = world_z, scene_z = world_y.
# The X negation is what makes the scene right-handed; it also mirrors yaw.
WORLD_TO_SCENE_MATRIX: np.ndarray = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def R_yaw(yaw_rad: float) -> np.ndarray:
    """Rotation about world Z (up) by ``yaw_rad`` (Anki yaw, forward = +Y)."""
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _head_tilt_matrix(head_angle_rad: float) -> np.ndarray:
    """Camera→body rotation including head pitch.

    Columns are where each camera basis vector maps in body coordinates:

    * camera X (right)   -> body X (right): (1, 0, 0)
    * camera Y (down)    -> body (0, sinθ, -cosθ)
    * camera Z (forward) -> body (0, cosθ,  sinθ)

    At level head (θ = 0) this is the +90° about X that takes camera-forward
    (+Zc) to body-forward (+Yb) and camera-down (+Yc) to body-down (-Zb). As
    the head tilts up (positive θ) the optical axis gains an up component.
    """
    c = np.cos(head_angle_rad)
    s = np.sin(head_angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, s, c], [0.0, -c, s]], dtype=np.float64)


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
    """Map a world-frame point to scene frame (no height offset)."""
    p = np.asarray(position_world, dtype=np.float64)
    return np.array([-p[0], p[2], p[1]], dtype=np.float64)


def world_to_scene_rotation(rotation_world: np.ndarray) -> np.ndarray:
    """Map a world-frame 3x3 rotation to scene frame via change of basis."""
    m = WORLD_TO_SCENE_MATRIX
    return m @ np.asarray(rotation_world, dtype=np.float64) @ m


def world_yaw_to_scene(yaw_rad: float) -> float:
    """Map a world-frame yaw (about world Z) to a scene rotation.y.

    The robot's world forward is ``(-sin ψ, cos ψ, 0)`` (body +Y at yaw ``ψ``).
    Under the world→scene change of basis (X→-X, Y→Z, Z→Y) it becomes the
    scene vector ``(sin ψ, 0, cos ψ)``. A Three.js ``rotation.y = θ`` rotates the
    +Z forward to ``(sin θ, 0, cos θ)``, so ``θ = ψ`` (the X negation cancels
    against body +Y pointing at +X in that basis — no mirroring).
    """
    return float(yaw_rad)
