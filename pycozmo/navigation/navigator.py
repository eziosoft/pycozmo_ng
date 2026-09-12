"""Navigation — navigate, approach cube, dock, reset origin, charger handlers."""

import math
import threading
import time
from collections.abc import Callable

import numpy as np

import pycozmo
from pycozmo.navigation.steering import steer_velocities, wrap_pi
from pycozmo.util import Angle, Pose


class Navigator:
    """Robot navigation: point-to-point drive, cube approach, docking.

    References shared state on ``self.brain`` (the :class:`App` core).
    """

    def __init__(self, app):
        self.brain = app

    # ------------------------------------------------------------------
    # Charger / origin handlers (COZMO-28)
    # ------------------------------------------------------------------

    def _on_charger_change(self, cli, state: bool) -> None:
        """Record whether the robot is physically on the charger."""
        del cli
        with self.brain._dock_lock:
            self.brain._on_charger = state
        if state:
            print("On charger detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _on_charging_change(self, cli, state: bool) -> None:
        """Record whether the robot is actively charging."""
        del cli
        with self.brain._dock_lock:
            self.brain._charging = state
        if state:
            print("Charging detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _reset_pose_origin(self) -> None:
        """Re-anchor the localization origin at the current pose.

        Sends SetOrigin so the robot's current pose becomes the new
        world frame (0,0,0) and records it as the dock pose. (COZMO-28)
        """
        cli = self.brain.cli
        if cli is None:
            return
        with self.brain._dock_lock:
            if self.brain._origin_reset_origin_id == cli.pose.origin_id:
                return

        reset_pkt = pycozmo.protocol_encoder.SetOrigin(
            pose_frame_id=cli.pose_frame_id + 1,
            pose_origin_id=cli.pose.origin_id + 1,
        )
        cli.conn.send(reset_pkt)
        time.sleep(0.5)
        with self.brain._dock_lock:
            self.brain._origin_reset_origin_id = cli.pose.origin_id
            self.brain._dock_pose = cli.pose
        print("Pose origin reset - current pose is now origin")

    # ------------------------------------------------------------------
    # Click-to-navigate (COZMO-28 / COZMO-30)
    # ------------------------------------------------------------------

    def _on_navigate_to(self, world_forward: float, world_left: float) -> None:
        """Handle click-to-navigate from browser.

        world_forward = X (Anki frame), world_left = Y (Anki frame).
        """
        if world_forward is None:
            if self.brain.cli:
                self.brain.cli.stop_all_motors()
            print("Navigation cancelled")
            return

        if not self.brain.cli:
            print("Navigation: no robot connected")
            return

        SHORT_DIST_MM = 100.0

        robot_pose = self.brain.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        dx = world_forward - robot_x
        dy = world_left - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        print(
            f"Navigating to ({world_forward:.0f}, {world_left:.0f}) - dist={dist:.0f}mm"
        )

        def _nav_thread():
            try:
                if dist <= SHORT_DIST_MM:
                    self._drive_to_point(world_forward, world_left)
                else:
                    heading = math.atan2(dy, dx)
                    target_pose = Pose(
                        world_forward,
                        world_left,
                        0,
                        angle_z=Angle(radians=heading),
                    )
                    self.brain.cli.go_to_pose(target_pose, relative_to_robot=False)
                    print("Arrived at target")
            except Exception as e:
                print(f"Navigation error: {e}")

        t = threading.Thread(target=_nav_thread, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Closed-loop drive (COZMO-35)
    # ------------------------------------------------------------------

    def _drive_to_point(
        self,
        tx: float,
        ty: float,
        *,
        stop_mm: float = 20.0,
        timeout_s: float = 15.0,
        speed_mmps: float = 60.0,
        final_heading: float | None = None,
        align_tol_rad: float = 0.06,
        target_provider: Callable[[], tuple[float, float] | None] | None = None,
    ) -> bool:
        """Closed-loop differential drive toward a point, reversing if needed.

        The heading toward the target is re-aimed at the *current* robot pose
        every tick, so the robot converges onto the target (the old code
        steered along the line-of-sight captured at click time and never
        re-aimed, missing any target not exactly on that ray). With
        ``target_provider`` the goal is re-read each tick — used by the cube
        approach to follow the live fused cube pose. Ends by turning in place
        to ``final_heading`` (if given) once close.

        Args:
            tx, ty: Target position (mm, world frame). Ignored while
                ``target_provider`` is given.
            stop_mm: Stop when within this distance of the target.
            timeout_s: Overall drive timeout.
            speed_mmps: Cruise wheel speed.
            final_heading: World heading (rad) to align to after arrival
                (turning in place).
            align_tol_rad: Heading tolerance for the final alignment.
            target_provider: Optional ``() -> (tx, ty) | None`` callable
                re-read each tick; ``None`` keeps the last target.

        Returns:
            True if the target was reached (within tolerance), False on
            timeout.
        """
        cli = self.brain.cli
        if cli is None:
            print("Drive: no robot connected")
            return False

        target = (tx, ty)

        def _read_target() -> tuple[float, float]:
            nonlocal target
            if target_provider is not None:
                fresh = target_provider()
                if fresh is not None:
                    target = fresh
            return target

        t0 = time.monotonic()
        reached = False
        try:
            while time.monotonic() - t0 <= timeout_s:
                gx, gy = _read_target()
                pose = cli.pose
                ex, ey = gx - pose.position.x, gy - pose.position.y
                if math.hypot(ex, ey) <= stop_mm:
                    reached = True
                    break
                vl, vr = steer_velocities(
                    pose.position.x,
                    pose.position.y,
                    pose.rotation.angle_z.radians,
                    gx,
                    gy,
                    speed_mmps=speed_mmps,
                )
                cli.drive_wheels(lwheel_speed=vl, rwheel_speed=vr)
                time.sleep(0.05)

            if not reached and time.monotonic() - t0 > timeout_s:
                print("Nav timeout")
            if reached and final_heading is not None:
                self._align_heading(final_heading, align_tol_rad)
        finally:
            cli.stop_all_motors()

        if reached:
            print(f"Arrived at ({tx:.0f}, {ty:.0f})")

        return reached

    def _align_heading(
        self,
        target_heading: float,
        tol_rad: float = 0.06,
        timeout_s: float = 3.0,
        turn_speed_mmps: float = 45.0,
    ) -> None:
        """Turn in place to ``target_heading`` (world frame, radians)."""
        cli = self.brain.cli
        if cli is None:
            return

        t0 = time.monotonic()
        try:
            while time.monotonic() - t0 <= timeout_s:
                yaw = cli.pose.rotation.angle_z.radians
                err = wrap_pi(target_heading - yaw)
                if abs(err) < tol_rad:
                    break
                omega = max(-1.0, min(1.0, err * 2.2)) * turn_speed_mmps
                cli.drive_wheels(lwheel_speed=-omega, rwheel_speed=omega)
                time.sleep(0.03)
        finally:
            cli.stop_all_motors()

    # ------------------------------------------------------------------
    # Cube approach (COZMO-28)
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_face_normal(
        cube_pos: np.ndarray,
        cube_rot: np.ndarray,
        robot_pos: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float, float] | None:
        """Pick the cube face to approach and the heading that faces it.

        The four wall faces' outward normals are the cube's local ±X and ±Y
        axes expressed in world frame. The best face is the one whose normal
        most points toward the robot (so Cozmo sees the marker frontally).
        When the robot is roughly diagonal to the cube the two adjacent
        faces score about equally — ties are broken deterministically by the
        smallest heading change, so the choice no longer flips arbitrarily
        and the robot stops aiming "at the corner".

        Args:
            cube_pos: Cube center (mm, world frame; z ignored).
            cube_rot: 3x3 cube rotation matrix in world frame.
            robot_pos: Robot (x, y) in world frame (mm).
            robot_yaw: Robot heading (rad, world frame).

        Returns:
            ``(nx, ny, heading)``: the outward face normal (unit, 2D) and the
            world heading (rad) that faces the cube along that normal, or
            ``None`` if no face can be resolved (e.g. cube at the robot).
        """
        cx, cy = float(cube_pos[0]), float(cube_pos[1])
        rx, ry = robot_pos
        dx, dy = rx - cx, ry - cy
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = dx / d, dy / d

        normals = [
            (float(cube_rot[0, 0]), float(cube_rot[1, 0])),
            (-float(cube_rot[0, 0]), -float(cube_rot[1, 0])),
            (float(cube_rot[0, 1]), float(cube_rot[1, 1])),
            (-float(cube_rot[0, 1]), -float(cube_rot[1, 1])),
        ]

        best: tuple[float, float, float] | None = None
        best_key: tuple[float, float] | None = None
        for n in normals:
            nlen = math.hypot(*n)
            if nlen < 1e-3:
                continue
            nx, ny = n[0] / nlen, n[1] / nlen
            dot = nx * ux + ny * uy
            heading = math.atan2(-ny, -nx)  # robot at cube + n faces the cube
            turn = abs(wrap_pi(heading - robot_yaw))
            key = (-dot, turn)
            if best_key is None or key < best_key:
                best_key = key
                best = (nx, ny, heading)
        return best

    def _on_approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Handle cube click from browser — two-step approach & align.

        Step 1 drives to a standoff point along the best face normal, far
        enough that Cozmo sees the marker frontally and the world-map pose
        sharpens. Step 2 then approaches closer **re-aiming every tick at the
        live fused cube pose** (not the click-time snapshot), so the target
        tracks the constant stream of camera measurements instead of
        dead-reckoning to a stale position.
        """
        if not self.brain.cli:
            print("Approach: no robot connected")
            return

        STEP1_DIST = 160.0
        STEP2_DIST = 90.0
        BACK_AWAY_DIST = 220.0
        TOO_CLOSE = 120.0
        HEAD_TILT_DEG = -10.0

        world_map = self.brain.web_viewer.world_map

        def _cube_state(cube_idx: int):
            cubes = world_map.get_cubes()
            return cubes.get(cube_idx)

        state = _cube_state(cube_idx)
        if state is None:
            print(f"Cube {cube_idx + 1} not in world map")
            return

        robot_pose = self.brain.cli.pose
        rx, ry = robot_pose.position.x, robot_pose.position.y
        robot_yaw = robot_pose.rotation.angle_z.radians

        face = self._pick_face_normal(
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

        self.brain.cli.set_head_angle(
            math.radians(HEAD_TILT_DEG), accel=30.0, max_speed=30.0
        )
        print(f"Head tilt {HEAD_TILT_DEG:.0f} deg")

        # The approach face is fixed at step 1 (the robot stays on the same
        # side); only the cube POSITION is re-read live in step 2.
        face_xy = (nx, ny)

        def _standoff(dist_mm: float) -> tuple[float, float]:
            p = state.position
            if p is None:
                return (cube_x, cube_y)
            return (
                float(p[0]) + face_xy[0] * dist_mm,
                float(p[1]) + face_xy[1] * dist_mm,
            )

        def _live_standoff(dist_mm: float):
            s = _cube_state(cube_idx)
            if s is None:
                return None
            p = s.position
            return (
                float(p[0]) + face_xy[0] * dist_mm,
                float(p[1]) + face_xy[1] * dist_mm,
            )

        def _wait_for_fresh(timeout_s: float = 3.0) -> None:
            """Block until a camera measurement newer than now is fused."""
            before = _cube_state(cube_idx)
            base_ts = before.updated_at if before is not None else None
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                s = _cube_state(cube_idx)
                ts = s.updated_at if s is not None else None
                if ts is not None and (base_ts is None or ts > base_ts + 0.05):
                    return
                time.sleep(0.05)
            print("Cube not re-detected at standoff - using last known pose")

        def _nav_thread():
            try:
                dist_to_cube = math.hypot(cube_x - rx, cube_y - ry)
                if dist_to_cube < TOO_CLOSE:
                    bx = cube_x + (rx - cube_x) / dist_to_cube * BACK_AWAY_DIST
                    by = cube_y + (ry - cube_y) / dist_to_cube * BACK_AWAY_DIST
                    print(f"Too close ({dist_to_cube:.0f}mm) - backing away")
                    self._drive_to_point(bx, by, timeout_s=8.0)

                time.sleep(0.3)  # let the head settle before the first shot

                s1x, s1y = _standoff(STEP1_DIST)
                print(f"Step 1: drive to view standoff ({s1x:.0f}, {s1y:.0f})")
                self._drive_to_point(s1x, s1y, final_heading=heading, timeout_s=20.0)

                _wait_for_fresh()

                s2x, s2y = _live_standoff(STEP2_DIST) or (s1x, s1y)
                print(f"Step 2: approach & align at ({s2x:.0f}, {s2y:.0f})")
                self._drive_to_point(
                    s2x,
                    s2y,
                    stop_mm=25.0,
                    timeout_s=15.0,
                    final_heading=heading,
                    target_provider=lambda: _live_standoff(STEP2_DIST),
                )

                print(f"Aligned with cube {cube_idx + 1}")
            except Exception as e:
                print(f"Approach error: {e}")

        t = threading.Thread(target=_nav_thread, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Docking (COZMO-28)
    # ------------------------------------------------------------------

    def _on_dock(self) -> None:
        """Handle 'z' keypress - drive robot back into the dock."""
        if not self.brain.cli:
            print("Dock: no robot connected")
            return
        if self.brain._dock_pose is None:
            print("Dock: no start pose recorded")
            return

        DOCK_DRIVE_DIST = 150.0
        DOCK_REVERSE_SPEED = 80.0
        DOCK_AIM_BEHIND = 40.0
        DOCK_CONTACT_DIST = 15.0
        DOCK_MAX_RETRIES = 3
        DOCK_RETRY_SPEED = 60.0
        DOCK_RETRY_DRIVE = 40.0
        DOCK_STALL_ITERS = 20

        dock = self.brain._dock_pose
        dock_x = dock.position.x
        dock_y = dock.position.y
        dock_heading = dock.rotation.angle_z.radians

        turn_x = dock_x + DOCK_DRIVE_DIST * math.cos(dock_heading)
        turn_y = dock_y + DOCK_DRIVE_DIST * math.sin(dock_heading)

        aim_x = dock_x - DOCK_AIM_BEHIND * math.cos(dock_heading)
        aim_y = dock_y - DOCK_AIM_BEHIND * math.sin(dock_heading)

        def _is_charging() -> bool:
            with self.brain._dock_lock:
                return self.brain._on_charger or self.brain._charging

        def _dist_to_dock() -> float:
            rp = self.brain.cli.pose
            dx = aim_x - rp.position.x
            dy = aim_y - rp.position.y
            return math.sqrt(dx * dx + dy * dy)

        def _reverse_until_contact() -> bool:
            prev_dist = None
            stall = 0
            while True:
                if _is_charging():
                    return True

                dist = _dist_to_dock()

                if dist < DOCK_CONTACT_DIST:
                    return False

                if prev_dist is not None and dist >= prev_dist - 1.0:
                    stall += 1
                    if stall >= DOCK_STALL_ITERS:
                        return False
                else:
                    stall = 0
                prev_dist = dist

                self.brain.cli.drive_wheels(
                    lwheel_speed=-DOCK_REVERSE_SPEED,
                    rwheel_speed=-DOCK_REVERSE_SPEED,
                )
                time.sleep(0.05)

        def _dock_thread():
            try:
                turn_pose = Pose(turn_x, turn_y, 0, angle_z=Angle(radians=dock_heading))
                print("Driving to 150mm in front of dock")
                self.brain.cli.go_to_pose(turn_pose, relative_to_robot=False)
                time.sleep(0.3)

                docked = False
                for attempt in range(1, DOCK_MAX_RETRIES + 1):
                    print(f"Docking attempt {attempt}/{DOCK_MAX_RETRIES}")
                    if _reverse_until_contact():
                        docked = True
                        break

                    if _is_charging():
                        docked = True
                        break

                    if attempt < DOCK_MAX_RETRIES:
                        print("No charge at contact - backing away to retry")
                        self.brain.cli.drive_wheels(
                            lwheel_speed=DOCK_RETRY_SPEED,
                            rwheel_speed=DOCK_RETRY_SPEED,
                            duration=DOCK_RETRY_DRIVE / DOCK_RETRY_SPEED,
                        )
                        time.sleep(0.3)
                    else:
                        print(
                            f"Docking failed after {DOCK_MAX_RETRIES} attempts "
                            f"(dist={_dist_to_dock():.0f}mm, no charge)"
                        )

                self.brain.cli.stop_all_motors()

                if docked or _is_charging():
                    self._reset_pose_origin()
                    print("Docked and pose reset - dock is now origin")
                elif not docked:
                    print("Docking did not complete - no pose reset")
            except Exception as e:
                print(f"Dock error: {e}")

        t = threading.Thread(target=_dock_thread, daemon=True)
        t.start()
