"""CubeApproach — two-step standoff approach and grab alignment.

Extracted from Navigator._on_approach_cube (COZMO-40).  The class takes a
robot client, a cube-state reader, and two driving callbacks so it can be
unit-tested without live hardware.
"""

import math
import threading
import time

import numpy as np

import pycozmo
from pycozmo.navigation.steering import wrap_pi
from pycozmo.util import Angle, Pose


class CubeApproach:
    """Drive the robot to a cube, align with the best face, and grab it.

    Constructor args (dependency injection):
        cli             Robot client — pose, go_to_pose, drive_wheels,
                        set_lift_height, set_head_angle, stop_all_motors.
        get_cube_state  ``Callable[[int], CubeState | None]`` — returns the
                        fused world-frame state for a given cube index.
        drive_to_point  ``Callable[[tx, ty, *, stop_mm, timeout_s, ...], bool]``
                        Closed-loop drive to a point (Navigator._drive_to_point).
        move_forward    ``Callable[[dist_mm], bool]`` Straight-ahead creep
                        (Navigator._move_forward).
    """

    # ------------------------------------------------------------------
    # Tuning constants
    # ------------------------------------------------------------------
    STEP1_DIST_MM = 160.0
    STEP2_DIST_MM = 90.0
    BACK_AWAY_DIST_MM = 220.0
    TOO_CLOSE_MM = 120.0
    HEAD_TILT_DEG = -10.0
    NEAR_TILT_DIST_MM = 100.0
    NEAR_TILT_DEG = -20.0

    GRAB_FORWARD_MM = 20.0
    GRAB_RANGE_MM = (50.0, 150.0)
    GRAB_CENTER_TOL_MM = 30.0
    GRAB_HEADING_TOL_RAD = math.radians(5.0)
    GRAB_RETRY_ANGLE_RAD = math.radians(12.0)
    GRAB_RECHECK_MM = 10.0
    GRAB_MAX_RECHECKS = 3
    LIFT_UP_ACCEL = 3.0
    LIFT_UP_SPEED = 3.0
    FRESH_TOL_S = 1.5
    ARRIVE_POS_MM = 8.0
    ARRIVE_ANG_RAD = math.radians(2.0)

    def __init__(self, cli, get_cube_state, drive_to_point, move_forward):
        self.cli = cli
        self._get_cube_state = get_cube_state
        self._drive_to_point = drive_to_point
        self._move_forward = move_forward

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------

    @staticmethod
    def pick_face_normal(
        cube_pos: np.ndarray,
        cube_rot: np.ndarray,
        robot_pos: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float, float] | None:
        """Pick the cube face to approach and the heading that faces it.

        See ``Navigator._pick_face_normal`` for full docstring.
        """
        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        rx, ry = robot_pos
        dx, dy = rx - cx, ry - cy
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = dx / d, dy / d

        plan = []
        for col in range(3):
            nx, ny = float(cube_rot[0, col]), float(cube_rot[1, col])
            length = math.hypot(nx, ny)
            if length > 1e-3:
                plan.append((nx / length, ny / length, length))
        if not plan:
            return None
        plan.sort(key=lambda p: p[2], reverse=True)
        wall_axes = [p for p in plan if p[2] >= 0.5 * plan[0][2]][:2]

        normals: list[tuple[float, float]] = []
        for axis in wall_axes:
            normals.append((axis[0], axis[1]))
            normals.append((-axis[0], -axis[1]))

        best: tuple[float, float, float] | None = None
        best_key: tuple[float, float] | None = None
        for n in normals:
            nx, ny = n
            dot = nx * ux + ny * uy
            heading = math.atan2(-ny, -nx)
            turn = abs(wrap_pi(heading - robot_yaw))
            key = (-dot, turn)
            if best_key is None or key < best_key:
                best_key = key
                best = (nx, ny, heading)
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _standoff(
        self,
        state,
        dist_mm: float,
        n: tuple[float, float],
    ) -> tuple[float, float]:
        """Cube center + ``n * dist_mm`` from state (horizontal plane only)."""
        p = state.position
        return (
            float(p[0]) + n[0] * dist_mm,
            float(p[1]) + n[1] * dist_mm,
        )

    def _corrected_target(self, cube_idx: int, dist_mm: float):
        """Recompute approach line from corrected cube pose after step 1."""
        s = self._get_cube_state(cube_idx)
        rp = self.cli.pose
        if s is None:
            return None
        f = self.pick_face_normal(
            s.position,
            s.rotation_matrix,
            (rp.position.x, rp.position.y),
            rp.rotation.angle_z.radians,
        )
        if f is None:
            return None
        nx, ny, heading = f
        return (
            float(s.position[0]) + nx * dist_mm,
            float(s.position[1]) + ny * dist_mm,
            heading,
        )

    def _wait_for_fresh(self, cube_idx: int, timeout_s: float = 3.0) -> None:
        """Block until a camera measurement newer than now is fused."""
        before = self._get_cube_state(cube_idx)
        base_ts = before.updated_at if before is not None else None
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            s = self._get_cube_state(cube_idx)
            ts = s.updated_at if s is not None else None
            if ts is not None and (base_ts is None or ts > base_ts + 0.05):
                return
            time.sleep(0.05)
        print("Cube not re-detected at standoff - using last known pose")

    def _watch_near_tilt(self, cube_idx: int) -> None:
        """Drop head to ``NEAR_TILT_DEG`` once within proximity."""
        while True:
            s = self._get_cube_state(cube_idx)
            if s is not None:
                rp = self.cli.pose
                d = math.hypot(
                    s.position[0] - rp.position.x,
                    s.position[1] - rp.position.y,
                )
                if d < self.NEAR_TILT_DIST_MM:
                    self.cli.set_head_angle(
                        math.radians(self.NEAR_TILT_DEG),
                        accel=30.0,
                        max_speed=30.0,
                    )
                    print(f"Head tilt {self.NEAR_TILT_DEG:.0f} deg (dist {d:.0f}mm)")
                    return
            time.sleep(0.05)

    def _verify_grab_alignment(
        self,
        cube_idx: int,
    ) -> tuple[bool, list[str], float | None, float, float, bool, bool]:
        """Three checks that the robot is positioned to grab the cube.

        Returns ``(passed, messages, facing_err_rad, forward, lateral,
        centered, fresh)``.
        """
        s = self._get_cube_state(cube_idx)
        msgs: list[str] = []
        if s is None:
            return False, ["no cube in world map"], None, 0.0, 0.0, False, False
        rp = self.cli.pose
        rpos = (rp.position.x, rp.position.y)
        ryaw = rp.rotation.angle_z.radians

        age = time.monotonic() - s.updated_at
        fresh = age <= self.FRESH_TOL_S
        msgs.append(
            f"fresh={'ok' if fresh else 'STALE'} "
            f"(age {age:.1f}s, conf {s.confidence:.2f})"
        )

        f = self.pick_face_normal(s.position, s.rotation_matrix, rpos, ryaw)
        facing_err_rad = None if f is None else abs(wrap_pi(f[2] - ryaw))
        facing = (
            facing_err_rad is not None
            and facing_err_rad <= self.GRAB_HEADING_TOL_RAD
        )
        if facing_err_rad is None:
            msgs.append("facing=FAIL (no face normal resolved)")
        else:
            msgs.append(
                f"facing={'ok' if facing else 'WRONG'} "
                f"(err {math.degrees(facing_err_rad):.1f} deg)"
            )

        dx = s.position[0] - rpos[0]
        dy = s.position[1] - rpos[1]
        fwdx, fwdy = math.cos(ryaw), math.sin(ryaw)
        forward = dx * fwdx + dy * fwdy
        lateral = dx * -fwdy + dy * fwdx
        centered = abs(lateral) <= self.GRAB_CENTER_TOL_MM
        in_range = self.GRAB_RANGE_MM[0] <= forward <= self.GRAB_RANGE_MM[1]
        msgs.append(
            f"centered={'ok' if centered else 'OFF'} "
            f"(lat {lateral:.0f}mm <= {self.GRAB_CENTER_TOL_MM:.0f}mm), "
            f"range={'ok' if in_range else 'OUT'} "
            f"(fwd {forward:.0f}mm in "
            f"{self.GRAB_RANGE_MM[0]:.0f}-{self.GRAB_RANGE_MM[1]:.0f}mm)"
        )

        return (
            fresh and facing and centered and in_range,
            msgs,
            facing_err_rad,
            forward,
            lateral,
            centered,
            fresh,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def approach(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Two-step approach & grab for the given cube.

        Spawns a daemon thread so the caller (WebSocket handler) returns
        immediately.
        """
        t = threading.Thread(
            target=self._run, args=(cube_idx, cube_x, cube_y), daemon=True
        )
        t.start()

    def _run(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Blocking implementation — runs in a daemon thread."""
        try:
            state = self._get_cube_state(cube_idx)
            if state is None:
                print(f"Cube {cube_idx + 1} not in world map")
                return

            robot_pose = self.cli.pose
            rx, ry = robot_pose.position.x, robot_pose.position.y
            robot_yaw = robot_pose.rotation.angle_z.radians

            face = self.pick_face_normal(
                state.position, state.rotation_matrix, (rx, ry), robot_yaw
            )
            if face is None:
                print("Could not determine approach face")
                return
            nx, ny, heading = face

            print(
                f"Approaching cube {cube_idx + 1} at ({cube_x:.0f}, {cube_y:.0f}) "
                f"face normal=({nx:.2f}, {ny:.2f})"
            )

            # Forks down
            self.cli.set_lift_height(
                pycozmo.MIN_LIFT_HEIGHT.mm,
                accel=self.LIFT_UP_ACCEL,
                max_speed=self.LIFT_UP_SPEED,
            )
            print(f"Lift lowered ({pycozmo.MIN_LIFT_HEIGHT.mm:.0f}mm)")

            self.cli.set_head_angle(
                math.radians(self.HEAD_TILT_DEG), accel=30.0, max_speed=30.0
            )
            print(f"Head tilt {self.HEAD_TILT_DEG:.0f} deg")

            # Back away if too close
            dist_to_cube = math.hypot(cube_x - rx, cube_y - ry)
            if dist_to_cube < self.TOO_CLOSE_MM:
                bx = cube_x + (rx - cube_x) / dist_to_cube * self.BACK_AWAY_DIST_MM
                by = cube_y + (ry - cube_y) / dist_to_cube * self.BACK_AWAY_DIST_MM
                print(f"Too close ({dist_to_cube:.0f}mm) - backing away")
                self._drive_to_point(bx, by, timeout_s=8.0)

            time.sleep(0.3)

            threading.Thread(
                target=self._watch_near_tilt, args=(cube_idx,), daemon=True
            ).start()

            # Step 1: view standoff
            s1x, s1y = self._standoff(state, self.STEP1_DIST_MM, (nx, ny))
            print(f"Step 1: go_to_pose to view standoff ({s1x:.0f}, {s1y:.0f})")
            self.cli.go_to_pose(
                Pose(s1x, s1y, 0, angle_z=Angle(radians=heading)),
                relative_to_robot=False,
            )

            self._wait_for_fresh(cube_idx)

            # Step 2: corrected standoff
            target2 = self._corrected_target(cube_idx, self.STEP2_DIST_MM)
            if target2 is None:
                print("Cube lost after step 1 - keeping last standoff")
                s2x, s2y, heading2 = s1x, s1y, heading
            else:
                s2x, s2y, heading2 = target2
            print(
                f"Step 2: go_to_pose to {self.STEP2_DIST_MM:.0f}mm standoff "
                f"({s2x:.0f}, {s2y:.0f})"
            )
            self.cli.go_to_pose(
                Pose(s2x, s2y, 0, angle_z=Angle(radians=heading2)),
                relative_to_robot=False,
            )

            # Wait for arrival
            t_arrive = time.monotonic()
            while time.monotonic() - t_arrive < 10.0:
                rp = self.cli.pose
                pos_err = math.hypot(s2x - rp.position.x, s2y - rp.position.y)
                ang_err = abs(wrap_pi(heading2 - rp.rotation.angle_z.radians))
                if pos_err <= self.ARRIVE_POS_MM and ang_err <= self.ARRIVE_ANG_RAD:
                    break
                time.sleep(0.05)

            print(f"Aligned with cube {cube_idx + 1}")

            # Alignment verification with retry
            attempts = 0
            while True:
                (aligned, checks, facing_err_rad, forward, lateral, _, fresh) = (
                    self._verify_grab_alignment(cube_idx)
                )
                for msg in checks:
                    print(f"  check: {msg}")

                recoverable = (
                    not aligned
                    and fresh
                    and facing_err_rad is not None
                    and facing_err_rad <= self.GRAB_RETRY_ANGLE_RAD
                    and abs(lateral) <= self.GRAB_CENTER_TOL_MM
                    and forward - self.GRAB_RECHECK_MM >= self.GRAB_RANGE_MM[0]
                    and attempts < self.GRAB_MAX_RECHECKS
                )
                if not recoverable:
                    break

                attempts += 1
                print(
                    f"Misaligned by {math.degrees(facing_err_rad):.1f} deg "
                    f"- creeping {self.GRAB_RECHECK_MM:.0f}mm closer "
                    f"({attempts}/{self.GRAB_MAX_RECHECKS})"
                )
                self._move_forward(self.GRAB_RECHECK_MM)
                time.sleep(0.3)

            if not aligned:
                print("Alignment check FAILED - not grabbing")
                return

            print(
                f"Aligned - grabbing cube {cube_idx + 1}: "
                f"+{self.GRAB_FORWARD_MM:.0f}mm, lift to "
                f"{pycozmo.MAX_LIFT_HEIGHT.mm:.0f}mm"
            )
            self._move_forward(self.GRAB_FORWARD_MM)
            self.cli.set_lift_height(
                pycozmo.MAX_LIFT_HEIGHT.mm,
                accel=self.LIFT_UP_ACCEL,
                max_speed=self.LIFT_UP_SPEED,
            )
            print(f"Cube {cube_idx + 1} grabbed (lift up)")
        except Exception as e:  # noqa: BLE001
            print(f"Approach error: {e}")
