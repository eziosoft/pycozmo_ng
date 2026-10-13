"""State serialization for the Three.js web viewer.

Pure functions that convert world-map data to JSON for the browser.
Extracted from WebViewerServer for testability and clarity.
"""

import json
import math
import time
from typing import Any

import numpy as np

from pycozmo_ng import coords
from pycozmo_ng.cubes.identity import cube_label

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

# Same reasoning for the motion-stereo point cloud: it is only serialized
# when it changed (the server's decision), but a single cloud can still be
# thousands of points, each a 3-number JSON entry. Stride-decimate above
# this count before sending.
MAX_POINT_CLOUD_SENT: int = 2000

# Size of a cube in scene units (mm) — mirrors CUBE_SIZE in index.html.
CUBE_SIZE_MM: float = 44.0

# Charger marker size in scene units (mm) — mirrors the CHARGER_* constants
# in index.html. The marker is 23 x 15 mm (COZMO-65).
CHARGER_WIDTH_MM: float = 23.0
CHARGER_HEIGHT_MM: float = 15.0


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
    start_time: float,
    occupancy: tuple[str, list[float]] | None = None,
    camera_settings: dict | None = None,
    face_png: str | None = None,
    point_cloud: np.ndarray | None = None,
) -> str:
    """Serialize world-map state to JSON for the browser.

    Pure function — no side effects, easy to test.

    Args:
        world_map: WorldMap instance to read state from.
        telemetry: Latest robot telemetry dict.
        floor_size_mm: (forward_extent, left_extent) of floor grid in mm.
        floor_center_mm: (forward, left) centre of floor grid in mm (body-frame).
        start_time: Server start time (for timestamp).
        occupancy: Payload from ``OccupancyDelivery.take()`` — ``(base64 PNG,
            [origin_x, origin_y])`` — or None when this message carries no
            grid. The change/send decision lives in that one call, not here.
        camera_settings: Current camera settings dict (or None).
        point_cloud: Motion-stereo cloud as an (N, 3) world-frame array
            (mm), or None when this message carries no cloud. World→scene
            conversion and decimation happen here.

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
    carried_id = f"cube_{carry_cube}" if carry_state == "carrying" and carry_cube is not None else None
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

    # Charger: one static landmark — null until the first fused sighting
    # (world map never forgets it, only reset_map / origin change clear it).
    # Rotation carries the dock facing (marker normal yaw) for alignment.
    charger_state = world.get("charger")
    charger = None
    if charger_state is not None:
        pos_gl = coords.world_to_scene_point(charger_state["position"])
        rot_gl = coords.world_to_scene_rotation(np.asarray(charger_state.get("rotation", np.eye(3)), dtype=np.float64))
        charger = {
            "position": [float(v) for v in pos_gl],
            "rotation": [row.tolist() for row in rot_gl],
            "confidence": float(charger_state.get("confidence", 1.0)),
            "width": CHARGER_WIDTH_MM,
            "height": CHARGER_HEIGHT_MM,
        }

    # ArUco markers: static printed landmarks keyed by dictionary ID.
    # size_mm (per-ID, from the JSON config) sizes the rendered plane; the
    # browser fetches each marker's bitmap once from /aruco-texture?id=N.
    aruco_list = []
    for marker_id, ms in world.get("aruco_markers", {}).items():
        pos_gl = coords.world_to_scene_point(ms["position"])
        rot_gl = coords.world_to_scene_rotation(np.asarray(ms.get("rotation", np.eye(3)), dtype=np.float64))
        aruco_list.append(
            {
                "id": int(marker_id),
                "position": [float(v) for v in pos_gl],
                "rotation": [row.tolist() for row in rot_gl],
                "confidence": float(ms.get("confidence", 1.0)),
                "size_mm": float(ms.get("size_mm", 0.0)),
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
        [fx, fy + PATH_SCENE_LIFT_MM, fz] for (fx, fy, fz) in (coords.world_to_scene_point(p) for p in path_world)
    ]

    # A* planned path: stored as 2-D world-frame [x, y] — convert to scene
    # frame (scene = [world_y, lift, world_x]) and lift above the ground so
    # the line doesn't z-fight the grid floor, same as the path trail.
    astar_path_scene = [[float(y), PATH_SCENE_LIFT_MM, float(x)] for (x, y) in world.get("astar_path", [])]

    # The JSON state bag: heterogeneous by design (lists, dicts, scalars) —
    # later keys (face_png, occupancy_*) are attached conditionally.
    state: dict[str, Any] = {
        "cubes": cube_list,
        "faces": face_list,
        "charger": charger,
        "aruco_markers": aruco_list,
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

    # Motion-stereo point cloud: world frame -> scene frame. Sent as a
    # flat [x, y, z, ...] list of ints (mm) — round-tripping sub-mm
    # precision would double the payload for no visible difference.
    if point_cloud is not None and len(point_cloud) > 0:
        cloud = np.asarray(point_cloud, dtype=np.float64)
        if len(cloud) > MAX_POINT_CLOUD_SENT:
            step = math.ceil(len(cloud) / MAX_POINT_CLOUD_SENT)
            cloud = cloud[::step]
        cloud_scene = [coords.world_to_scene_point(p) for p in cloud]
        state["point_cloud"] = [round(v) for p in cloud_scene for v in p]

    if camera_settings is not None:
        state["camera_settings"] = dict(camera_settings)

    # Include procedural face image if available.
    if face_png is not None:
        state["face_png"] = face_png

    # Include the occupancy grid when this message carries one (decided once,
    # by OccupancyDelivery.take()).
    if occupancy is not None:
        state["occupancy_png"] = occupancy[0]
        state["occupancy_origin"] = occupancy[1]

    return json.dumps(state)
