"""Navigation — navigate, approach cube, dock, reset origin, charger handlers."""

import math
import threading
import time

import numpy as np

import pycozmo
from pycozmo.navigation.astar import AStarPlanner, PathPoint
from pycozmo.navigation.cube_approach import CubeApproach
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.steering import steer_velocities, wrap_pi
from pycozmo.util import Angle, Pose


class Navigator:
    """Robot navigation: point-to-point drive, cube approach, docking.

    Uses :class:`DockState` for charger/dock data. Receives ``cli`` and
    ``web_viewer`` via methods rather than holding a Brain reference.
    """

    def __init__(self, dock_state: DockState):
        self.ds = dock_state
        # Set by Brain after construction (cross-domain references)
        self.cli = None
        self.web_viewer = None
        self._test_mode = False
        self._test_mode_thread: threading.Thread | None = None
        self._test_mode_lock = threading.Lock()
        # A* pathfinding
        self.astar_enabled = False
        self.astar_planner: AStarPlanner | None = None
        self._path_waypoints: list[PathPoint] = []
        self._current_waypoint_idx = 0
        self._path_following = False
        self._path_follow_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Charger / origin handlers (COZMO-28)
    # ------------------------------------------------------------------

    def _on_charger_change(self, cli, state: bool) -> None:
        """Record whether the robot is physically on the charger."""
        del cli
        with self.ds.lock:
            self.ds.on_charger = state
        if state:
            print("On charger detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _on_charging_change(self, cli, state: bool) -> None:
        """Record whether the robot is actively charging."""
        del cli
        with self.ds.lock:
            self.ds.charging = state
        if state:
            print("Charging detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _reset_pose_origin(self) -> None:
        """Re-anchor the localization origin at the current pose.

        Sends SetOrigin so the robot's current pose becomes the new
        world frame (0,0,0) and records it as the dock pose. (COZMO-28)
        """
        cli = self.cli
        if cli is None:
            return
        with self.ds.lock:
            if self.ds.origin_reset_origin_id == cli.pose.origin_id:
                return

        reset_pkt = pycozmo.protocol_encoder.SetOrigin(
            pose_frame_id=cli.pose_frame_id + 1,
            pose_origin_id=cli.pose.origin_id + 1,
        )
        cli.conn.send(reset_pkt)
        time.sleep(0.5)
        with self.ds.lock:
            self.ds.origin_reset_origin_id = cli.pose.origin_id
            self.ds.dock_pose = cli.pose
        print("Pose origin reset - current pose is now origin")

    # ------------------------------------------------------------------
    # Click-to-navigate (COZMO-28 / COZMO-30)
    # ------------------------------------------------------------------

    def _on_navigate_to(self, world_forward: float, world_left: float) -> None:
        """Handle click-to-navigate from browser.

        world_forward = X (Anki frame), world_left = Y (Anki frame).
        When A* is enabled and a planner is set, computes an obstacle-
        avoiding path through the occupancy grid.
        """
        if world_forward is None:
            if self.cli:
                self.cli.stop_all_motors()
            self._path_following = False
            print("Navigation cancelled")
            return

        if not self.cli:
            print("Navigation: no robot connected")
            return

        SHORT_DIST_MM = 100.0

        robot_pose = self.cli.pose
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
                if self.astar_enabled and self.astar_planner is not None:
                    self._follow_astar_path(world_forward, world_left)
                elif dist <= SHORT_DIST_MM:
                    self._drive_to_point(world_forward, world_left)
                else:
                    heading = math.atan2(dy, dx)
                    target_pose = Pose(
                        world_forward,
                        world_left,
                        0,
                        angle_z=Angle(radians=heading),
                    )
                    self.cli.go_to_pose(target_pose, relative_to_robot=False)
                    print("Arrived at target")
            except Exception as e:  # noqa: BLE001
                print(f"Navigation error: {e}")
            finally:
                self._path_following = False

        t = threading.Thread(target=_nav_thread, daemon=True)
        t.start()

    def _follow_astar_path(self, goal_x: float, goal_y: float) -> None:
        """Compute and follow an A* path to *(goal_x, goal_y)*.

        Replans dynamically as the occupancy map updates — if the
        next waypoint becomes occupied or the path is exhausted,
        a new A* search is issued from the current robot pose.
        """
        if self.astar_planner is None:
            print("A*: no planner — falling back to direct nav")
            self._drive_to_point(goal_x, goal_y)
            return

        robot_pose = self.cli.pose
        start_x = robot_pose.position.x
        start_y = robot_pose.position.y

        with self._path_follow_lock:
            self._path_following = True
            self._current_waypoint_idx = 0
            self._path_waypoints = (
                self.astar_planner.find_path(start_x, start_y, goal_x, goal_y) or []
            )
            print(f"A*: computed {len(self._path_waypoints)} waypoints")

        if not self._path_waypoints:
            print("A*: no path found — falling back to direct nav")
            self._path_following = False
            self._drive_to_point(goal_x, goal_y)
            return

        REPLAN_WAYPOINTS = 3  # replan after consuming this many waypoints
        REPLAN_INTERVAL_S = 1.0  # or every N seconds

        t0 = time.monotonic()
        waypoints_consumed = 0

        while self._path_following and self.cli:
            if not self._path_waypoints:
                # Path exhausted — replan from current position
                if time.monotonic() - t0 < REPLAN_INTERVAL_S:
                    time.sleep(0.1)
                    continue
                robot_pose = self.cli.pose
                cx = robot_pose.position.x
                cy = robot_pose.position.y
                new_path = self.astar_planner.find_path(cx, cy, goal_x, goal_y)
                if not new_path:
                    print("A*: replan failed — no path to goal")
                    break
                with self._path_follow_lock:
                    self._path_waypoints = new_path
                waypoints_consumed = 0
                t0 = time.monotonic()
                print(f"A*: replanned — {len(new_path)} waypoints")
                continue

            # Check if we need to replan before the next waypoint
            if waypoints_consumed >= REPLAN_WAYPOINTS:
                robot_pose = self.cli.pose
                cx = robot_pose.position.x
                cy = robot_pose.position.y
                new_path = self.astar_planner.find_path(cx, cy, goal_x, goal_y)
                if new_path and len(new_path) > 0:
                    with self._path_follow_lock:
                        self._path_waypoints = new_path
                    waypoints_consumed = 0
                    t0 = time.monotonic()
                    print(f"A*: replanned — {len(new_path)} waypoints")
                else:
                    # Current path still OK, keep going
                    waypoints_consumed = 0

            with self._path_follow_lock:
                if not self._path_waypoints:
                    continue
                wp = self._path_waypoints.pop(0)

            if not self._path_following:
                print("A*: path following cancelled")
                return

            arrived = self._drive_to_point(wp.x, wp.y, stop_mm=15.0)
            waypoints_consumed += 1
            t0 = time.monotonic()

            if not arrived:
                print(f"A*: timed out at waypoint ({wp.x:.0f}, {wp.y:.0f})")
                break

            self.cli.stop_all_motors()
            time.sleep(0.15)

        if self._path_following:
            print("A*: reached goal")

        self._path_following = False

    def set_astar_planner(self, planner: AStarPlanner | None) -> None:
        """Inject or remove the A* planner."""
        self.astar_planner = planner

    def set_astar_enabled(self, enabled: bool) -> None:
        """Enable or disable A* pathfinding."""
        self.astar_enabled = enabled
        if not enabled:
            self._path_following = False
            print(f"A* {'disabled' if not enabled else 'enabled'}")

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
    ) -> bool:
        """Closed-loop differential drive toward a point, reversing if needed.

        The bearing to the target is re-aimed at the *current* robot pose every
        tick, so the robot converges onto the target (the old code steered
        along the line-of-sight captured at click time and never re-aimed,
        missing any target not exactly on that ray). Ends by turning in place
        to ``final_heading`` (if given) once close.

        Args:
            tx, ty: Target position (mm, world frame).
            stop_mm: Stop when within this distance of the target.
            timeout_s: Overall drive timeout.
            speed_mmps: Cruise wheel speed.
            final_heading: World heading (rad) to align to after arrival
                (turning in place).
            align_tol_rad: Heading tolerance for the final alignment.

        Returns:
            True if the target was reached (within tolerance), False on
            timeout.
        """
        cli = self.cli
        if cli is None:
            print("Drive: no robot connected")
            return False

        t0 = time.monotonic()
        reached = False
        try:
            while time.monotonic() - t0 <= timeout_s:
                pose = cli.pose
                ex, ey = tx - pose.position.x, ty - pose.position.y
                if math.hypot(ex, ey) <= stop_mm:
                    reached = True
                    break
                vl, vr = steer_velocities(
                    pose.position.x,
                    pose.position.y,
                    pose.rotation.angle_z.radians,
                    tx,
                    ty,
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
        cli = self.cli
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

    def _move_forward(self, dist_mm: float, timeout_s: float = 6.0) -> bool:
        """Drive straight ahead ``dist_mm`` using the onboard planner.

        ``go_to_pose`` (relative to the robot) is used instead of the Python
        closed loop (``_drive_to_point``) for these very short moves: right
        after a ``go_to_pose`` the robot can still be finalizing its path and
        ignores raw ``drive_wheels`` commands. (COZMO-37)

        Returns True if the robot actually arrived within ~4mm.
        """
        cli = self.cli
        if cli is None:
            return False
        start_x = cli.pose.position.x
        start_y = cli.pose.position.y
        start_yaw = cli.pose.rotation.angle_z.radians
        cli.go_to_pose(
            Pose(
                dist_mm,
                0,
                0,
                angle_z=Angle(radians=0.0),
            ),
            relative_to_robot=True,
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 <= timeout_s:
            rp = cli.pose
            moved = math.hypot(rp.position.x - start_x, rp.position.y - start_y)
            err = abs(wrap_pi(start_yaw - rp.rotation.angle_z.radians))
            if moved >= dist_mm - 4.0 and err <= math.radians(3.0):
                return True
            time.sleep(0.05)
        print(f"Forward move of {dist_mm:.0f}mm not reached")
        return False

    # ------------------------------------------------------------------
    # Cube approach (COZMO-28 / COZMO-40)
    # ------------------------------------------------------------------

    def _get_cube_approach(self) -> CubeApproach:
        """Lazily create the CubeApproach instance with DI callbacks."""
        if not hasattr(self, "_cube_approach"):
            world_map = self.web_viewer.world_map
            self._cube_approach = CubeApproach(
                cli=self.cli,
                get_cube_state=lambda idx: world_map.get_cubes().get(idx),
                drive_to_point=self._drive_to_point,
                move_forward=self._move_forward,
            )
        # Refresh cli reference (may reconnect)
        self._cube_approach.cli = self.cli
        return self._cube_approach

    @staticmethod
    def _pick_face_normal(
        cube_pos: np.ndarray,
        cube_rot: np.ndarray,
        robot_pos: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float, float] | None:
        """Pick the cube face to approach — delegates to CubeApproach."""
        return CubeApproach.pick_face_normal(cube_pos, cube_rot, robot_pos, robot_yaw)

    def _on_approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Handle cube click from browser — delegates to CubeApproach."""
        if not self.cli:
            print("Approach: no robot connected")
            return
        ca = self._get_cube_approach()
        ca.approach(cube_idx, cube_x, cube_y)

    # ------------------------------------------------------------------
    # Docking (COZMO-28)
    # ------------------------------------------------------------------

    def _on_dock(self) -> None:
        """Handle 'z' keypress - drive robot back into the dock."""
        if not self.cli:
            print("Dock: no robot connected")
            return
        if self.ds.dock_pose is None:
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

        dock = self.ds.dock_pose
        dock_x = dock.position.x
        dock_y = dock.position.y
        dock_heading = dock.rotation.angle_z.radians

        turn_x = dock_x + DOCK_DRIVE_DIST * math.cos(dock_heading)
        turn_y = dock_y + DOCK_DRIVE_DIST * math.sin(dock_heading)

        aim_x = dock_x - DOCK_AIM_BEHIND * math.cos(dock_heading)
        aim_y = dock_y - DOCK_AIM_BEHIND * math.sin(dock_heading)

        def _is_charging() -> bool:
            with self.ds.lock:
                return self.ds.on_charger or self.ds.charging

        def _dist_to_dock() -> float:
            rp = self.cli.pose
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

                self.cli.drive_wheels(
                    lwheel_speed=-DOCK_REVERSE_SPEED,
                    rwheel_speed=-DOCK_REVERSE_SPEED,
                )
                time.sleep(0.05)

        def _dock_thread():
            try:
                turn_pose = Pose(turn_x, turn_y, 0, angle_z=Angle(radians=dock_heading))
                print("Driving to 150mm in front of dock")
                self.cli.go_to_pose(turn_pose, relative_to_robot=False)
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
                        self.cli.drive_wheels(
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

                self.cli.stop_all_motors()

                if docked or _is_charging():
                    self._reset_pose_origin()
                    print("Docked and pose reset - dock is now origin")
                elif not docked:
                    print("Docking did not complete - no pose reset")
            except Exception as e:  # noqa: BLE001
                print(f"Dock error: {e}")

        t = threading.Thread(target=_dock_thread, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Test mode — loop approach on cubes with 3 sec pauses
    # ------------------------------------------------------------------

    def set_test_mode(self, enabled: bool) -> None:
        """Start or stop the test mode loop."""
        with self._test_mode_lock:
            if self._test_mode == enabled:
                return
            self._test_mode = enabled

        if enabled:
            self._test_mode_thread = threading.Thread(
                target=self._test_mode_loop, daemon=True
            )
            self._test_mode_thread.start()
            print("Test mode ON")
        else:
            print("Test mode OFF")

    def _test_mode_loop(self) -> None:
        """Continuously approach cubes with 3 second waits."""
        while True:
            with self._test_mode_lock:
                if not self._test_mode:
                    return
            if not self.cli or self.web_viewer is None:
                time.sleep(1)
                continue
            cubes = self.web_viewer.world_map.get_cubes()
            if not cubes:
                time.sleep(1)
                continue
            cube_idx = next(iter(cubes))
            cube = cubes[cube_idx]
            cx, cy, _ = cube.position
            print(f"Test mode: approaching cube {cube_idx + 1}")
            self._on_approach_cube(cube_idx, float(cx), float(cy))
            time.sleep(3.0)
