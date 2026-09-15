"""
World map — the single source of truth for Cozmo's world.

Holds Cozmo's pose, the light-cube poses, and the driven path, all in **world
frame** (Anki: X right, Y forward, Z up, mm). Receives raw inputs each tick —
the robot's fused pose from ``cli.pose`` and cube measurements from the vision
tracker (camera frame) — transforms them into world frame, and fuses repeated
cube measurements there so a stationary cube stays put while Cozmo drives.

The web viewer and web server are thin adapters over this class; it has no
dependency on the web stack, so it can be used standalone.

"""

import time
from collections import deque
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pycozmo import logger
from pycozmo.util.coords import camera_to_world

__all__ = [
    "CubeState",
    "WorldMap",
]


class CubeState:
    """Fused world-frame state of a single cube."""

    __slots__ = (
        "age",
        "confidence",
        "num_faces",
        "position",
        "rotation_matrix",
        "updated_at",
    )

    def __init__(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        confidence: float,
        num_faces: int = 1,
        age: int = 0,
        updated_at: float = 0.0,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        self.confidence = float(confidence)
        self.num_faces = int(num_faces)
        self.age = int(age)
        # Monotonic time (seconds) of the measurement that produced this
        # state. Lets callers (e.g. the navigator) detect that a fresh
        # camera measurement has been fused since a given instant.
        self.updated_at = float(updated_at)


class WorldMap:
    """Owns Cozmo, cube, and path state in world frame (mm).

    Thread-safe: mutated from the pycozmo event-dispatcher thread (robot state,
    camera frames) and read from the web-server thread. All public reads and
    writes take a single lock.
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.15,
        rotation_alpha: float = 0.12,
        confidence_floor: float = 0.5,
        max_path_points: int = 5000,
    ):
        """
        Args:
            smoothing_alpha: Exponential smoothing factor for cube position
                fusion (0 = no update, 1 = no smoothing).  Lower values
                suppress position jitter more aggressively — recommended range
                0.08–0.20 for cube-approach precision.
            rotation_alpha: Exponential smoothing factor for cube rotation
                (SLERP) fusion.  Lower values suppress attitude noise more
                aggressively — recommended range 0.05–0.20 for 10–20°
                per-frame jitter.  (0 = no update, 1 = no smoothing).
            confidence_floor: Cube measurements below this confidence are
                dropped entirely (not fused).
            max_path_points: Rolling cap on the number of path points kept.
        """
        self.smoothing_alpha = float(smoothing_alpha)
        self.rotation_alpha = float(rotation_alpha)
        self.confidence_floor = float(confidence_floor)
        self.max_path_points = int(max_path_points)

        self._lock = Lock()

        # Robot pose in world frame.
        self._robot_position: np.ndarray = np.zeros(3, dtype=np.float64)
        self._robot_yaw: float = 0.0
        self._robot_origin_id: int | None = None

        # Cubes: cube_id (0-indexed) -> fused world-frame state.
        self._cubes: dict[int, CubeState] = {}

        # Path: rolling buffer of world-frame positions (mm).
        self._path_points: deque[np.ndarray] = deque(maxlen=self.max_path_points)

    # -- robot pose ---------------------------------------------------------

    def update_robot_pose(
        self,
        position: np.ndarray,
        yaw_rad: float,
        origin_id: int | None = None,
        record_path: bool = True,
    ) -> None:
        """Update the robot's world pose from ``cli.pose``.

        Args:
            position: World position (mm, Anki frame).
            yaw_rad: Yaw about world Z (radians, ``cli.pose.rotation.angle_z``).
            origin_id: Localization origin id. When it changes (pick-up /
                put-down re-anchors the world frame) the map is reset.
            record_path: Append the position to the path trail.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        with self._lock:
            if (
                origin_id is not None
                and self._robot_origin_id is not None
                and origin_id != self._robot_origin_id
            ):
                logger.info("WorldMap: localization origin changed — resetting map")
                self._cubes.clear()
                self._path_points.clear()
            self._robot_origin_id = origin_id
            self._robot_position = pos
            self._robot_yaw = float(yaw_rad)
            if record_path:
                self._path_points.append(pos.copy())

    def reset_path(self) -> None:
        """Clear the path trail (e.g. on pick-up or a 'p' key press)."""
        with self._lock:
            self._path_points.clear()

    def add_path_point(self, position: np.ndarray) -> None:
        """Append a world-frame position (mm) to the path trail."""
        with self._lock:
            self._path_points.append(
                np.asarray(position, dtype=np.float64).reshape(3).copy()
            )

    def reset_map(self) -> None:
        """Clear all entities and the path (fresh world)."""
        with self._lock:
            self._cubes.clear()
            self._path_points.clear()
            self._robot_position = np.zeros(3, dtype=np.float64)
            self._robot_yaw = 0.0
            self._robot_origin_id = None

    # -- cubes --------------------------------------------------------------

    def update_cube_measurement(
        self,
        cube_id: int,
        position_cam: np.ndarray,
        rotation_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        num_faces: int | None = None,
        camera_lateral_offset_mm: float = 0.0,
    ) -> None:
        """Fuse a camera-frame cube measurement into the world map.

        The measurement is transformed camera → body → world using the current
        robot pose, gated on confidence, then fused (exponential smoothing on
        position, SLERP on rotation) against the previous world-frame state.

        Args:
            cube_id: 0-indexed cube id (0..2).
            position_cam: Cube center position in camera frame (mm).
            rotation_cam: 3x3 cube rotation matrix in camera frame.
            confidence: Detection confidence (0..1).
            head_angle_rad: Head pitch at measurement time (radians).
            num_faces: How many marker faces produced this measurement; keeps
                the fused state's ``num_faces`` current instead of forever
                keeping the first frame's value. (COZMO-36)
            camera_lateral_offset_mm: Measured lateral offset of camera from
                robot centre (mm). From robot center detection (black circles).
        """
        if confidence < self.confidence_floor:
            return

        with self._lock:
            pos_world, rot_world = camera_to_world(
                position_cam,
                rotation_cam,
                head_angle_rad,
                self._robot_position,
                self._robot_yaw,
                camera_lateral_offset_mm,
            )
            prev = self._cubes.get(cube_id)
            now = time.monotonic()
            if prev is None:
                self._cubes[cube_id] = CubeState(
                    position=pos_world,
                    rotation_matrix=rot_world,
                    confidence=confidence,
                    num_faces=int(num_faces) if num_faces is not None else 1,
                    updated_at=now,
                )
                return

            # Position: confidence-weighted exponential smoothing.
            pos_blend = self.smoothing_alpha * float(confidence)
            position = pos_blend * pos_world + (1.0 - pos_blend) * prev.position

            # Rotation: confidence-weighted SLERP with its own (lower) alpha
            # to suppress attitude noise more aggressively than position.
            rot_blend = self.rotation_alpha * float(confidence)
            prev_q = R.from_matrix(prev.rotation_matrix)
            new_q = R.from_matrix(rot_world)
            prev_quat = prev_q.as_quat()
            new_quat = new_q.as_quat()
            if np.dot(prev_quat, new_quat) < 0:
                new_quat = -new_quat
                new_q = R.from_quat(new_quat)
            key_rots = R.concatenate([prev_q, new_q])
            slerp = Slerp([0, 1], key_rots)
            rotation = slerp(rot_blend).as_matrix()

            self._cubes[cube_id] = CubeState(
                position=position,
                rotation_matrix=rotation,
                confidence=confidence,
                num_faces=int(num_faces) if num_faces is not None else prev.num_faces,
                age=0,
                updated_at=now,
            )

    def forget_cube(self, cube_id: int) -> None:
        """Drop a cube from the map.

        Called when the cube's IMU detects movement — the old position is stale
        and the cube will be re-placed on the next camera frame.
        """
        with self._lock:
            self._cubes.pop(cube_id, None)

    # -- read API -----------------------------------------------------------

    def get_robot_pose(self) -> tuple[np.ndarray, float]:
        """Return (world position mm, yaw rad)."""
        with self._lock:
            return self._robot_position.copy(), self._robot_yaw

    def get_cubes(self) -> dict[int, CubeState]:
        """Return a copy of the fused cube states (world frame)."""
        with self._lock:
            return {
                cube_id: CubeState(
                    position=cube_state.position.copy(),
                    rotation_matrix=cube_state.rotation_matrix.copy(),
                    confidence=cube_state.confidence,
                    num_faces=cube_state.num_faces,
                    age=cube_state.age,
                    updated_at=cube_state.updated_at,
                )
                for cube_id, cube_state in self._cubes.items()
            }

    def get_path_points(self) -> list:
        """Return a copy of the path points (world frame, mm)."""
        with self._lock:
            return [p.copy() for p in self._path_points]

    def get_world_state(self) -> dict:
        """Return the full world state as plain data (world frame).

        Keys: ``robot`` (position, yaw, origin_id), ``cubes`` (cube_id ->
        position/rotation/confidence), ``path_points``.
        """
        with self._lock:
            return {
                "robot": {
                    "position": self._robot_position.tolist(),
                    "yaw": self._robot_yaw,
                    "origin_id": self._robot_origin_id,
                },
                "cubes": {
                    cube_id: {
                        "position": cube_state.position.tolist(),
                        "rotation": cube_state.rotation_matrix.tolist(),
                        "confidence": cube_state.confidence,
                    }
                    for cube_id, cube_state in self._cubes.items()
                },
                "path_points": [p.tolist() for p in self._path_points],
            }
