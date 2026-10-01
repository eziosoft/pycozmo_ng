"""State serialization for the Three.js web viewer.

Pure functions that convert world-map data to JSON for the browser.
Extracted from WebViewerServer for testability and clarity.
"""

import json
import math
import time

from pycozmo import coords
from pycozmo.cubes.identity import cube_label

# How far (mm) to lift the rendered path trail above the scene ground plane
# (scene +Y). The trail is a zero-thickness line at ground level, so without a
# small offset it z-fights / clips the grid floor in the browser. This is a
# pure display artifact — it is NOT applied to world-frame data or cubes/robot.
PATH_SCENE_LIFT_MM: float = 3.0

# The world map can hold thousands of path points; serializing all of them on
# every broadcast makes each state message bigger as the session runs, which
# pushes the viewer further and further behind. Decimate before sending — the
# browser can't visually tell the difference above this many points.
MAX_PATH_POINTS_SENT: int = 800

# Occupancy changes incrementally (a few cells per frame); re-encoding and
# re-sending the full PNG that often is wasted bandwidth. Encode at most this
# often and only embed the PNG in a state message when it actually changed.
OCCUPANCY_SEND_INTERVAL_S: float = 0.2

# Size of a cube in scene units (mm) — mirrors CUBE_SIZE in index.html.
CUBE_SIZE_MM: float = 44.0


def encode_jpeg(image, quality: int = 70) -> bytes:
    """Encode a PIL image to JPEG bytes (runs on a worker thread).

    The image is the raw frame from Cozmo — no brightness/gamma
    post-processing. If you want to tune the look, send
    ``SetCameraParams`` to Cozmo (gain, exposure, auto-exposure) via
    ``cli.set_camera_params(...)`` instead. (COZMO-24)
    """
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def build_state_json(
    world_map,
    telemetry: dict,
    floor_size_mm: tuple[float, float],
    floor_center_mm: tuple[float, float],
    occupancy_png: str | None,
    occupancy_origin: list[float] | None,
    occupancy_seq: int,
    occupancy_seq_sent: int,
    start_time: float,
    force_occupancy: bool = False,
    camera_settings: dict | None = None,
    face_png: str | None = None,
) -> str:
    """Serialize world-map state to JSON for the browser.

    Pure function — no side effects, easy to test.

    Args:
        world_map: WorldMap instance to read state from.
        telemetry: Latest robot telemetry dict.
        floor_size_mm: (width, height) of floor grid in mm.
        floor_center_mm: (center_x, center_y) of floor grid in mm.
        occupancy_png: Base64-encoded occupancy PNG (or None).
        occupancy_origin: [origin_x, origin_y] of occupancy grid.
        occupancy_seq: Current occupancy sequence number.
        occupancy_seq_sent: Last occupancy sequence sent to browser.
        start_time: Server start time (for timestamp).
        force_occupancy: If True, include occupancy even if unchanged.
        camera_settings: Current camera settings dict (or None).

    Returns:
        JSON string for the browser.
    """
    world = world_map.get_world_state()

    # Robot: world frame -> scene frame, with yaw mapped to scene rotation.y.
    robot_pos = coords.world_to_scene_point(world["robot"]["position"])
    robot_heading = coords.world_yaw_to_scene(world["robot"]["yaw"])

    # Cubes: world frame -> scene frame (identity at the wire boundary).
    # `carried` marks the cube riding the forks: the world map only has its
    # last-seen (stale) pose, so the browser locks it to the robot instead.
    carry_state = telemetry.get("carry")
    carry_cube = telemetry.get("carry_cube")
    carried_id = (
        f"cube_{carry_cube}"
        if carry_state == "carrying" and carry_cube is not None
        else None
    )
    cube_list = []
    for cube_index, cube_state in world["cubes"].items():
        pos_gl = coords.world_to_scene_point(cube_state["position"])
        rot_gl = coords.world_to_scene_rotation(cube_state["rotation"])
        cube_id = cube_label(cube_index)
        cube_list.append(
            {
                "id": cube_id,
                "position": [float(v) for v in pos_gl],
                "rotation": [row.tolist() for row in rot_gl],
                "size": CUBE_SIZE_MM,
                "face_idx": cube_state.get("face_idx", -1),
                "face_rotation": cube_state.get("face_rotation", 0),
                "carried": cube_id == carried_id,
            }
        )

    if carried_id is not None and not any(c["id"] == carried_id for c in cube_list):
        # The carried cube left the camera frame, so the world map no
        # longer tracks it. Send a placeholder at the robot pose — the
        # browser re-locks the cube to the forks anyway (carried: true).
        cube_list.append(
            {
                "id": carried_id,
                "position": [float(v) for v in robot_pos],
                "rotation": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "size": CUBE_SIZE_MM,
                "face_idx": -1,
                "face_rotation": 0,
                "carried": True,
            }
        )

    # Faces: world frame -> scene frame.
    face_list = []
    for face_id, face_state in world.get("faces", {}).items():
        pos_gl = coords.world_to_scene_point(face_state["position"])
        face_list.append(
            {
                "id": f"face_{face_id}",
                "position": [float(v) for v in pos_gl],
                "confidence": face_state.get("confidence", 1.0),
                "label": face_state.get("label", "face"),
            }
        )

    # Path: world frame -> scene frame, lifted just above the ground so the
    # zero-thickness trail doesn't z-fight the grid floor. Decimated so the
    # payload size doesn't grow with session length (see
    # MAX_PATH_POINTS_SENT).
    path_world = world["path_points"]
    if len(path_world) > MAX_PATH_POINTS_SENT:
        step = math.ceil(len(path_world) / MAX_PATH_POINTS_SENT)
        # Align sampling to the END of the trail so the most recent
        # position (robot nose of the trail) is always included.
        offset = (len(path_world) - 1) % step
        path_world = path_world[offset::step]
    path_scene = [
        [fx, fy + PATH_SCENE_LIFT_MM, fz]
        for (fx, fy, fz) in (coords.world_to_scene_point(p) for p in path_world)
    ]

    # A* planned path: stored as 2-D world-frame [x, y] — convert to scene
    # frame (scene = [world_y, lift, world_x]) and lift above the ground so
    # the line doesn't z-fight the grid floor, same as the path trail.
    astar_path_scene = [
        [float(y), PATH_SCENE_LIFT_MM, float(x)]
        for (x, y) in world.get("astar_path", [])
    ]

    state = {
        "cubes": cube_list,
        "faces": face_list,
        "robot": {
            "position": [float(v) for v in robot_pos],
            "heading": float(robot_heading),
        },
        "path_points": path_scene,
        "astar_path": astar_path_scene,
        "telemetry": dict(telemetry),
        "floor_size": list(floor_size_mm),
        "floor_center": list(floor_center_mm),
        "timestamp": int((time.time() - start_time) * 1000),
    }

    if camera_settings is not None:
        state["camera_settings"] = dict(camera_settings)

    # Include procedural face image if available.
    if face_png is not None:
        state["face_png"] = face_png

    # Include occupancy grid PNG if available — and only when it actually
    # changed (or the receiver needs a full snapshot, e.g. on connect).
    if occupancy_png is not None and (
        force_occupancy or occupancy_seq != occupancy_seq_sent
    ):
        state["occupancy_png"] = occupancy_png
        state["occupancy_origin"] = occupancy_origin

    return json.dumps(state)
