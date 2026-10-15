"""
World map — the single source of truth for Cozmo's world.

Holds Cozmo's pose, the light-cube poses, and the driven path, all in **world
frame** (Anki: X right, Y forward, Z up, mm). Receives raw inputs each tick —
the robot's fused pose from ``cli.pose`` and cube measurements from the vision
tracker (camera frame) — transforms them into world frame, and fuses repeated
cube measurements there so a stationary cube stays put while Cozmo drives.

FACE INDEX MAPPING
-------------------
Each cube has 6 marker faces, indexed 0-5:

=======  ========  =========  ==========================================
face_idx  face_type  rotation  Marker file
=======  ========  =========  ==========================================
0         wall      0          cube{N}_face_0.jpg
1         wall      90         cube{N}_face_1.jpg
2         wall      180        cube{N}_face_2.jpg
3         wall      270        cube{N}_face_3.jpg
4         bottom    0          cube{N}_face_bottom.jpg
5         top       0          cube{N}_face_top.jpg
=======  ========  =========  ==========================================

The web viewer and web server are thin adapters over this class; it has no
dependency on the web stack, so it can be used standalone.

"""

import time
from collections import deque
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.logger import logger
from pycozmo_ng.util.coords import camera_to_world

__all__ = [
    "ArucoState",
    "CubeState",
    "FaceState",
    "WorldMap",
]

#: A measurement this far (mm) from the fused pose gets an info line:
#: it means the map was *stale* — an object re-sighted after leaving the
#: frame, or odometry drift — and is being corrected. Smaller deltas are
#: normal EMA convergence and stay out of the log. A stale map is what
#: makes the robot aim at where an object *was*, so the corrections are
#: exactly the positions worth having on record.
_LOG_CORRECTION_MM: float = 20.0


def _log_pose_correction(what: str, fused_xy, measured_xy) -> None:
    """Info line when a measurement pulls the fused pose by ``_LOG_CORRECTION_MM``+."""
    drift = float(np.hypot(float(measured_xy[0]) - float(fused_xy[0]), float(measured_xy[1]) - float(fused_xy[1])))
    if drift >= _LOG_CORRECTION_MM:
        logger.info(
            "WorldMap: %s measurement %.0f mm off the fused pose — correcting (%.0f, %.0f) → (%.0f, %.0f)",
            what,
            drift,
            float(fused_xy[0]),
            float(fused_xy[1]),
            float(measured_xy[0]),
            float(measured_xy[1]),
        )


class CubeState:
    """Fused world-frame state of a single cube."""

    __slots__ = (
        "age",
        "confidence",
        "face_idx",
        "face_rotation",
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
        face_idx: int = -1,
        face_rotation: int = 0,
        age: int = 0,
        updated_at: float = 0.0,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        self.confidence = float(confidence)
        self.num_faces = int(num_faces)
        self.face_idx = int(face_idx)
        self.face_rotation = int(face_rotation)
        self.age = int(age)
        # Monotonic time (seconds) of the measurement that produced this
        # state. Lets callers (e.g. the navigator) detect that a fresh
        # camera measurement has been fused since a given instant.
        self.updated_at = float(updated_at)


class FaceState:
    """Fused world-frame state of a detected face."""

    __slots__ = (
        "age",
        "confidence",
        "label",
        "position",
        "updated_at",
    )

    def __init__(
        self,
        position: np.ndarray,
        confidence: float,
        label: str = "face",
        age: int = 0,
        updated_at: float = 0.0,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.confidence = float(confidence)
        self.label = str(label)
        self.age = int(age)
        self.updated_at = float(updated_at)


class ChargerState:
    """Fused world-frame state of the charger marker (COZMO-65).

    The charger is a single static landmark — one slot, no per-object
    identity. Position and orientation are fused: the marker frame's
    world yaw (from the marker normal) is the dock's facing direction —
    the alignment reference for docking (COZMO-67).
    """

    __slots__ = (
        "age",
        "confidence",
        "position",
        "rotation_matrix",
        "updated_at",
    )

    def __init__(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        confidence: float,
        age: int = 0,
        updated_at: float = 0.0,
    ):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        self.confidence = float(confidence)
        self.age = int(age)
        self.updated_at = float(updated_at)


class ArucoState:
    """Fused world-frame state of one ArUco marker.

    Markers are printed static landmarks keyed by their dictionary ID —
    like the charger, there is no forgetting on visibility loss; the pose
    lives until ``reset_map`` or a localization-origin change.
    """

    __slots__ = (
        "age",
        "confidence",
        "marker_id",
        "position",
        "rotation_matrix",
        "size_mm",
        "updated_at",
    )

    def __init__(
        self,
        marker_id: int,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        confidence: float,
        size_mm: float,
        age: int = 0,
        updated_at: float = 0.0,
    ):
        self.marker_id = int(marker_id)
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        self.confidence = float(confidence)
        self.size_mm = float(size_mm)
        self.age = int(age)
        self.updated_at = float(updated_at)


# Face type to face_idx mapping (matches cube_detector face_type_lookup order)
_FACE_TYPE_TO_IDX = {
    "wall_0": 0,
    "wall_90": 1,
    "wall_180": 2,
    "wall_270": 3,
    "bottom_0": 4,
    "top_0": 5,
}


def _parse_rotation_source(rotation_source: str) -> tuple[int, int]:
    """Parse a rotation_source string like 'wall_90deg' into (face_idx, rotation_deg).

    Returns (-1, 0) if the string cannot be parsed.
    """
    try:
        parts = rotation_source.rsplit("deg", 1)
        key = parts[0]  # e.g. "wall_90"
        return _FACE_TYPE_TO_IDX.get(key, -1), int(key.split("_", 1)[1])
    except (ValueError, IndexError):
        return -1, 0


class WorldMap:
    """Owns Cozmo, cube, and path state in world frame (mm).

    Thread-safe: mutated from the pycozmo_ng event-dispatcher thread (robot state,
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

        # Cubes: cube_index (world index 0-2) -> fused world-frame state.
        self._cubes: dict[int, CubeState] = {}

        # Faces: face_id -> fused world-frame state.
        self._faces: dict[int, FaceState] = {}

        # Charger: single static landmark (None until first sighting).
        self._charger: ChargerState | None = None

        # ArUco markers: marker_id -> fused world-frame static landmark.
        self._aruco: dict[int, ArucoState] = {}

        # Path: rolling buffer of world-frame positions (mm).
        self._path_points: deque[np.ndarray] = deque(maxlen=self.max_path_points)

        # A* planned path (world-frame waypoints for the browser).
        self._astar_path: list[list[float]] = []

    def set_astar_path(self, waypoints: list[list[float]]) -> None:
        """Set the A* path waypoints for visualization.

        Each waypoint is [x, y] in world frame (mm, Anki convention).
        """
        with self._lock:
            self._astar_path = waypoints

    def clear_astar_path(self) -> None:
        """Clear the A* path."""
        with self._lock:
            self._astar_path = []

    def get_astar_path(self) -> list[list[float]]:
        """Return a copy of the A* path waypoints."""
        with self._lock:
            return [list(w) for w in self._astar_path]

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
            if origin_id is not None and self._robot_origin_id is not None and origin_id != self._robot_origin_id:
                # Positions are in the *old* frame — recording them lets a
                # log reader line the dropped world up against everything
                # logged before the reset.
                logger.info(
                    "WorldMap: localization origin changed — resetting map, dropping %s",
                    ", ".join(self._contents_locked()) or "empty map",
                )
                self._cubes.clear()
                self._charger = None
                self._aruco.clear()
                self._path_points.clear()
            self._robot_origin_id = origin_id
            self._robot_position = pos
            self._robot_yaw = float(yaw_rad)
            if record_path:
                self._path_points.append(pos.copy())

    def _contents_locked(self) -> list[str]:
        """Compact ``label (x, y)`` strings for every tracked object.

        Caller must hold ``self._lock``. Used by the origin-change log to
        record where the objects *were* in the frame being dropped, so a
        log reader can line the two worlds up.
        """
        items = [
            f"marker {mid} ({ms.position[0]:.0f}, {ms.position[1]:.0f})" for mid, ms in sorted(self._aruco.items())
        ]
        if self._charger is not None:
            items.append(f"charger ({self._charger.position[0]:.0f}, {self._charger.position[1]:.0f})")
        items += [
            f"cube {to_cube_id(ci)} ({cs.position[0]:.0f}, {cs.position[1]:.0f})"
            for ci, cs in sorted(self._cubes.items())
        ]
        return items

    def reset_path(self) -> None:
        """Clear the path trail (e.g. on pick-up or a 'p' key press)."""
        with self._lock:
            self._path_points.clear()

    def add_path_point(self, position: np.ndarray) -> None:
        """Append a world-frame position (mm) to the path trail."""
        with self._lock:
            self._path_points.append(np.asarray(position, dtype=np.float64).reshape(3).copy())

    def reset_map(self) -> None:
        """Clear all entities and the path (fresh world)."""
        with self._lock:
            self._cubes.clear()
            self._faces.clear()
            self._charger = None
            self._aruco.clear()
            self._path_points.clear()
            self._astar_path.clear()
            self._robot_position = np.zeros(3, dtype=np.float64)
            self._robot_yaw = 0.0
            self._robot_origin_id = None

    # -- cubes --------------------------------------------------------------

    def update_cube_measurement(
        self,
        cube_index: int,
        position_cam: np.ndarray,
        rotation_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        num_faces: int | None = None,
        camera_lateral_offset_mm: float = 0.0,
        rotation_source: str | None = None,
    ) -> None:
        """Fuse a camera-frame cube measurement into the world map.

        The measurement is transformed camera → body → world using the current
        robot pose, gated on confidence, then fused (exponential smoothing on
        position, SLERP on rotation) against the previous world-frame state.

        Args:
            cube_index: World-map index (0-2; public cube ID = index + 1,
                see ``pycozmo_ng.cubes.identity``).
            position_cam: Cube center position in camera frame (mm).
            rotation_cam: 3x3 cube rotation matrix in camera frame.
            confidence: Detection confidence (0..1).
            head_angle_rad: Head pitch at measurement time (radians).
            num_faces: How many marker faces produced this measurement; keeps
                the fused state's ``num_faces`` current instead of forever
                keeping the first frame's value. (COZMO-36)
            camera_lateral_offset_mm: Measured lateral offset of camera from
                robot centre (mm). From robot center detection (black circles).
            rotation_source: Which face determined the cube rotation, e.g.
                ``"wall_90deg"`` or ``"top_0deg"``. Parsed into face_idx and
                face_rotation for display in the 3D viewer.
        """
        if confidence < self.confidence_floor:
            return

        face_idx, face_rotation = _parse_rotation_source(rotation_source) if rotation_source else (-1, 0)

        with self._lock:
            pos_world, rot_world = camera_to_world(
                position_cam,
                rotation_cam,
                head_angle_rad,
                self._robot_position,
                self._robot_yaw,
                camera_lateral_offset_mm,
            )
            prev = self._cubes.get(cube_index)
            now = time.monotonic()
            if prev is None:
                self._cubes[cube_index] = CubeState(
                    position=pos_world,
                    rotation_matrix=rot_world,
                    confidence=confidence,
                    num_faces=int(num_faces) if num_faces is not None else 1,
                    face_idx=face_idx,
                    face_rotation=face_rotation,
                    updated_at=now,
                )
                logger.info(
                    "WorldMap: cube %d first seen at (%.0f, %.0f) — conf %.2f",
                    to_cube_id(cube_index),
                    pos_world[0],
                    pos_world[1],
                    confidence,
                )
                return

            _log_pose_correction(f"cube {to_cube_id(cube_index)}", prev.position, pos_world)

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

            self._cubes[cube_index] = CubeState(
                position=position,
                rotation_matrix=rotation,
                confidence=confidence,
                num_faces=int(num_faces) if num_faces is not None else prev.num_faces,
                face_idx=face_idx if face_idx >= 0 else prev.face_idx,
                face_rotation=face_rotation if face_idx >= 0 else prev.face_rotation,
                age=0,
                updated_at=now,
            )

    # -- faces --------------------------------------------------------------

    def update_face_measurement(
        self,
        face_id: int,
        position_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        camera_lateral_offset_mm: float = 0.0,
        label: str = "face",
    ) -> None:
        """Fuse a camera-frame face measurement into the world map.

        Transforms camera → body → world, then applies exponential smoothing
        against the previous world-frame state.

        Args:
            face_id: Face presence ID assigned by ``FaceIdentity`` (1+).
            position_cam: Face centre in camera frame (mm).
            confidence: Detection confidence (0..1).
            head_angle_rad: Head pitch at measurement time (radians).
            camera_lateral_offset_mm: Lateral offset of camera from robot centre.
            label: Human-readable label (e.g. "Face 1").
        """
        with self._lock:
            pos_world, _rot_world = camera_to_world(
                position_cam,
                np.eye(3),
                head_angle_rad,
                self._robot_position,
                self._robot_yaw,
                camera_lateral_offset_mm,
            )
            prev = self._faces.get(face_id)
            now = time.monotonic()
            if prev is None:
                self._faces[face_id] = FaceState(
                    position=pos_world,
                    confidence=confidence,
                    label=label,
                    updated_at=now,
                )
                # Debug: faces appear and vanish constantly; the presence
                # epochs are already surfaced by the face pipeline.
                logger.debug("WorldMap: %s first seen at (%.0f, %.0f)", label, pos_world[0], pos_world[1])
                return

            alpha = self.smoothing_alpha * float(confidence)
            position = alpha * pos_world + (1.0 - alpha) * prev.position

            self._faces[face_id] = FaceState(
                position=position,
                confidence=confidence,
                label=label,
                age=prev.age + 1,
                updated_at=now,
            )

    def forget_face(self, face_id: int) -> None:
        """Remove a tracked face from the map.

        Called by ``FrameStage`` when ``FaceIdentity`` reports that a
        presence ended (``FaceFrame.ended``).
        """
        with self._lock:
            self._faces.pop(face_id, None)

    def get_faces(self) -> dict[int, FaceState]:
        """Return a copy of the fused face states (world frame)."""
        with self._lock:
            return {
                face_id: FaceState(
                    position=fs.position.copy(),
                    confidence=fs.confidence,
                    label=fs.label,
                    age=fs.age,
                    updated_at=fs.updated_at,
                )
                for face_id, fs in self._faces.items()
            }

    def forget_cube(self, cube_index: int) -> bool:
        """Drop a cube's fused pose — it is no longer where we last saw it.

        A cube the robot moved (its accelerometer reported motion) or picked up
        may sit anywhere, so its pose is discarded: a forgotten cube reads as
        unknown and is rendered nowhere, while the next camera measurement
        starts a fresh pose with no stale history to blend against — unlike
        faces, cubes are never forgotten on mere loss of visibility.

        Args:
            cube_index: World-map index (0-2; public cube ID = index + 1).

        Returns:
            True if a pose was there to drop.
        """
        with self._lock:
            return self._cubes.pop(cube_index, None) is not None

    # -- charger ------------------------------------------------------------

    def update_charger_measurement(
        self,
        position_cam: np.ndarray,
        rotation_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        camera_lateral_offset_mm: float = 0.0,
    ) -> None:
        """Fuse a camera-frame charger measurement into the world map (COZMO-65).

        Transforms camera → body → world, then fuses against the previous
        world-frame state: exponential smoothing on position, confidence-
        weighted SLERP on rotation (the same rules as cubes). The marker
        frame's world yaw — the dock's facing direction — is the alignment
        reference for docking (COZMO-67).

        The charger is stationary, so there is no forgetting on visibility
        loss — the last fused pose stays until ``reset_map`` or a
        localization-origin change.

        Args:
            position_cam: Marker centre in camera frame (mm).
            rotation_cam: 3x3 marker rotation in camera frame — the marker
                plane basis from PnP (x/y in the sticker plane, z = normal).
            confidence: Detection confidence (0..1).
            head_angle_rad: Head pitch at measurement time (radians).
            camera_lateral_offset_mm: Lateral offset of camera from robot centre.
        """
        with self._lock:
            pos_world, rot_world = camera_to_world(
                position_cam,
                rotation_cam,
                head_angle_rad,
                self._robot_position,
                self._robot_yaw,
                camera_lateral_offset_mm,
            )
            prev = self._charger
            now = time.monotonic()
            if prev is None:
                self._charger = ChargerState(
                    position=pos_world,
                    rotation_matrix=rot_world,
                    confidence=confidence,
                    updated_at=now,
                )
                logger.info(
                    "WorldMap: charger marker first seen at (%.0f, %.0f) — conf %.2f",
                    pos_world[0],
                    pos_world[1],
                    confidence,
                )
                return

            _log_pose_correction("charger marker", prev.position, pos_world)

            alpha = self.smoothing_alpha * float(confidence)
            position = alpha * pos_world + (1.0 - alpha) * prev.position

            # Rotation: confidence-weighted SLERP with the cube rotation_alpha
            # so attitude noise is suppressed more aggressively than position.
            rot_blend = self.rotation_alpha * float(confidence)
            prev_q = R.from_matrix(prev.rotation_matrix)
            new_q = R.from_matrix(rot_world)
            prev_quat = prev_q.as_quat()
            new_quat = new_q.as_quat()
            if np.dot(prev_quat, new_quat) < 0:
                new_quat = -new_quat
                new_q = R.from_quat(new_quat)
            key_rots = R.concatenate([prev_q, new_q])
            rotation = Slerp([0, 1], key_rots)(rot_blend).as_matrix()

            self._charger = ChargerState(
                position=position,
                rotation_matrix=rotation,
                confidence=confidence,
                age=prev.age + 1,
                updated_at=now,
            )

    def get_charger(self) -> ChargerState | None:
        """Return a copy of the charger state (world frame), or None if unseen."""
        with self._lock:
            if self._charger is None:
                return None
            return ChargerState(
                position=self._charger.position.copy(),
                rotation_matrix=self._charger.rotation_matrix.copy(),
                confidence=self._charger.confidence,
                age=self._charger.age,
                updated_at=self._charger.updated_at,
            )

    # -- aruco ---------------------------------------------------------------

    def update_aruco_measurement(
        self,
        marker_id: int,
        position_cam: np.ndarray,
        rotation_cam: np.ndarray,
        confidence: float,
        head_angle_rad: float,
        size_mm: float,
        camera_lateral_offset_mm: float = 0.0,
    ) -> None:
        """Fuse a camera-frame ArUco measurement into the world map.

        Same fusion rules as the charger (exponential smoothing on position,
        confidence-weighted SLERP on rotation), but keyed by marker ID — each
        printed marker is its own static landmark. No forgetting on
        visibility loss; entries are cleared by ``reset_map`` or a
        localization-origin change.

        Args:
            marker_id: ArUco dictionary ID (the identity).
            position_cam: Marker centre in camera frame (mm).
            rotation_cam: 3x3 marker rotation in camera frame — the marker
                plane basis from PnP (x/y in the sticker plane, z = normal).
            confidence: Detection confidence (0..1).
            head_angle_rad: Head pitch at measurement time (radians).
            size_mm: Physical side length used for this marker's PnP solve.
            camera_lateral_offset_mm: Lateral offset of camera from robot centre.
        """
        with self._lock:
            pos_world, rot_world = camera_to_world(
                position_cam,
                rotation_cam,
                head_angle_rad,
                self._robot_position,
                self._robot_yaw,
                camera_lateral_offset_mm,
            )
            prev = self._aruco.get(marker_id)
            now = time.monotonic()
            if prev is None:
                self._aruco[marker_id] = ArucoState(
                    marker_id=marker_id,
                    position=pos_world,
                    rotation_matrix=rot_world,
                    confidence=confidence,
                    size_mm=size_mm,
                    updated_at=now,
                )
                logger.info(
                    "WorldMap: marker %d first seen at (%.0f, %.0f) — conf %.2f",
                    marker_id,
                    pos_world[0],
                    pos_world[1],
                    confidence,
                )
                return

            _log_pose_correction(f"marker {marker_id}", prev.position, pos_world)

            alpha = self.smoothing_alpha * float(confidence)
            position = alpha * pos_world + (1.0 - alpha) * prev.position

            rot_blend = self.rotation_alpha * float(confidence)
            prev_q = R.from_matrix(prev.rotation_matrix)
            new_q = R.from_matrix(rot_world)
            prev_quat = prev_q.as_quat()
            new_quat = new_q.as_quat()
            if np.dot(prev_quat, new_quat) < 0:
                new_quat = -new_quat
                new_q = R.from_quat(new_quat)
            key_rots = R.concatenate([prev_q, new_q])
            rotation = Slerp([0, 1], key_rots)(rot_blend).as_matrix()

            self._aruco[marker_id] = ArucoState(
                marker_id=marker_id,
                position=position,
                rotation_matrix=rotation,
                confidence=confidence,
                size_mm=size_mm,
                age=prev.age + 1,
                updated_at=now,
            )

    def set_aruco_world_pose(
        self,
        marker_id: int,
        position_world: np.ndarray,
        rotation_world: np.ndarray,
        size_mm: float = 22.5,
    ) -> None:
        """Hard-set a marker's world pose (overwrite, no fusion).

        Used after a successful dock: the robot is at the charger so the
        marker's exact position and orientation are known from geometry
        (e.g. 65 mm behind the camera).  The confidence is set to 1.0 so
        subsequent vision measurements converge toward the known landmark.
        """
        with self._lock:
            self._aruco[marker_id] = ArucoState(
                marker_id=marker_id,
                position=np.asarray(position_world, dtype=np.float64),
                rotation_matrix=np.asarray(rotation_world, dtype=np.float64),
                confidence=1.0,
                size_mm=float(size_mm),
                age=100,
                updated_at=time.monotonic(),
            )

    def get_aruco_markers(self) -> dict[int, ArucoState]:
        """Return a copy of the fused ArUco marker states, keyed by marker ID."""
        with self._lock:
            return {
                marker_id: ArucoState(
                    marker_id=ms.marker_id,
                    position=ms.position.copy(),
                    rotation_matrix=ms.rotation_matrix.copy(),
                    confidence=ms.confidence,
                    size_mm=ms.size_mm,
                    age=ms.age,
                    updated_at=ms.updated_at,
                )
                for marker_id, ms in self._aruco.items()
            }

    # -- read API -----------------------------------------------------------

    @staticmethod
    def _copy_cube_state(cube_state: CubeState) -> CubeState:
        """Return a deep copy of one cube state (arrays copied)."""
        return CubeState(
            position=cube_state.position.copy(),
            rotation_matrix=cube_state.rotation_matrix.copy(),
            confidence=cube_state.confidence,
            num_faces=cube_state.num_faces,
            face_idx=cube_state.face_idx,
            face_rotation=cube_state.face_rotation,
            age=cube_state.age,
            updated_at=cube_state.updated_at,
        )

    def get_robot_pose(self) -> tuple[np.ndarray, float]:
        """Return (world position mm, yaw rad)."""
        with self._lock:
            return self._robot_position.copy(), self._robot_yaw

    def get_cubes(self) -> dict[int, CubeState]:
        """Return a copy of the fused cube states, keyed by cube index (0-2)."""
        with self._lock:
            return {cube_index: self._copy_cube_state(cube_state) for cube_index, cube_state in self._cubes.items()}

    def get_cube(self, cube_index: int) -> CubeState | None:
        """Return a copy of one fused cube state, or None if unknown.

        ``cube_index`` is the world-map key (0-2; ``cube_index = cube ID - 1``,
        see ``pycozmo_ng.cubes.identity``). Single locked lookup — cheaper than
        :meth:`get_cubes` when only one cube is needed.
        """
        with self._lock:
            cube_state = self._cubes.get(cube_index)
            return self._copy_cube_state(cube_state) if cube_state else None

    def get_path_points(self) -> list:
        """Return a copy of the path points (world frame, mm)."""
        with self._lock:
            return [p.copy() for p in self._path_points]

    def get_world_state(self) -> dict:
        """Return the full world state as plain data (world frame).

        Keys: ``robot`` (position, yaw, origin_id), ``cubes`` (cube_index ->
        position/rotation/confidence), ``faces`` (face_id ->
        position/confidence/label), ``charger`` (position/rotation/confidence
        or None), ``aruco_markers`` (marker_id -> position/rotation/
        confidence/size_mm), ``path_points``.
        """
        with self._lock:
            robot_pos = self._robot_position.tolist()
            cubes = {
                cube_index: {
                    "position": cube_state.position.tolist(),
                    "rotation": cube_state.rotation_matrix.tolist(),
                    "confidence": cube_state.confidence,
                    "face_idx": cube_state.face_idx,
                    "face_rotation": cube_state.face_rotation,
                }
                for cube_index, cube_state in self._cubes.items()
            }
            faces = {
                face_id: {
                    "position": fs.position.tolist(),
                    "confidence": fs.confidence,
                    "label": fs.label,
                }
                for face_id, fs in self._faces.items()
            }
            path_points = [p.tolist() for p in self._path_points]
            astar_path = [list(w) for w in self._astar_path]
            charger = None
            if self._charger is not None:
                charger = {
                    "position": self._charger.position.tolist(),
                    "rotation": self._charger.rotation_matrix.tolist(),
                    "confidence": self._charger.confidence,
                }
            aruco_markers = {
                marker_id: {
                    "position": ms.position.tolist(),
                    "rotation": ms.rotation_matrix.tolist(),
                    "confidence": ms.confidence,
                    "size_mm": ms.size_mm,
                }
                for marker_id, ms in self._aruco.items()
            }
        return {
            "robot": {
                "position": robot_pos,
                "yaw": self._robot_yaw,
                "origin_id": self._robot_origin_id,
            },
            "cubes": cubes,
            "faces": faces,
            "charger": charger,
            "aruco_markers": aruco_markers,
            "path_points": path_points,
            "astar_path": astar_path,
        }
