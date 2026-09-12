"""Navigation — navigate, approach cube, dock, reset origin, charger handlers."""

import math
import threading
import time

import numpy as np

import pycozmo
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
        NAV_SPEED_MMPS = 60.0
        NAV_STOP_MM = 8.0
        NAV_TURN_FRAC = 1.2
        NAV_TIMEOUT_S = 15.0

        robot_pose = self.brain.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        dx = world_forward - robot_x
        dy = world_left - robot_y
        heading = math.atan2(dy, dx)
        dist = math.sqrt(dx * dx + dy * dy)

        def _short_navigate(tx: float, ty: float) -> None:
            """Closed-loop drive to a close target, reversing if needed."""
            t0 = time.monotonic()
            while True:
                if time.monotonic() - t0 > NAV_TIMEOUT_S:
                    print("Short nav timeout")
                    break

                pose = self.brain.cli.pose
                cx, cy = pose.position.x, pose.position.y
                ex = tx - cx
                ey = ty - cy
                d = math.hypot(ex, ey)
                if d < NAV_STOP_MM:
                    break

                cur_h = pose.rotation.angle_z.radians
                err = heading - cur_h
                while err > math.pi:
                    err -= 2.0 * math.pi
                while err < -math.pi:
                    err += 2.0 * math.pi

                reverse = abs(err) > math.pi / 2.0
                if reverse:
                    if err > 0.0:
                        err -= math.pi
                    else:
                        err += math.pi
                    turn = err * NAV_TURN_FRAC * NAV_SPEED_MMPS
                    l_speed = -NAV_SPEED_MMPS - turn
                    r_speed = -NAV_SPEED_MMPS + turn
                else:
                    turn = err * NAV_TURN_FRAC * NAV_SPEED_MMPS
                    l_speed = NAV_SPEED_MMPS - turn
                    r_speed = NAV_SPEED_MMPS + turn

                slow = max(0.2, min(1.0, d / 40.0))
                self.brain.cli.drive_wheels(
                    lwheel_speed=l_speed * slow,
                    rwheel_speed=r_speed * slow,
                )
                time.sleep(0.05)

            self.brain.cli.stop_all_motors()
            print(f"Arrived at ({tx:.0f}, {ty:.0f})")

        print(
            f"Navigating to ({world_forward:.0f}, {world_left:.0f}) - "
            f"dist={dist:.0f}mm, heading={math.degrees(heading):.0f} deg"
        )

        def _nav_thread():
            try:
                if dist <= SHORT_DIST_MM:
                    _short_navigate(world_forward, world_left)
                else:
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
    # Cube approach (COZMO-28)
    # ------------------------------------------------------------------

    def _on_approach_cube(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Handle cube click from browser - approach cube along its face normal."""
        if not self.brain.cli:
            print("Approach: no robot connected")
            return

        STEP1_DIST = 120.0
        STEP2_DIST = 120.0
        BACK_AWAY_DIST = 150.0
        TOO_CLOSE = 100.0

        robot_pose = self.brain.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        dist_to_cube = math.sqrt((cube_x - robot_x) ** 2 + (cube_y - robot_y) ** 2)
        need_back_away = dist_to_cube < TOO_CLOSE

        world = self.brain.web_viewer.world_map.get_world_state()
        cubes = world.get("cubes", {})
        if cube_idx not in cubes:
            print(f"Cube {cube_idx + 1} not in world map")
            return

        cube_rot = np.array(cubes[cube_idx]["rotation"])

        local_x = cube_rot @ np.array([1.0, 0.0, 0.0])
        local_y = cube_rot @ np.array([0.0, 1.0, 0.0])

        face_normals = [local_x, -local_x, local_y, -local_y]

        to_robot = np.array([robot_x - cube_x, robot_y - cube_y, 0.0])
        to_robot_len = np.linalg.norm(to_robot)
        if to_robot_len < 1:
            print("Already at cube")
            return
        to_robot_unit = to_robot / to_robot_len

        best_face = None
        best_dot = -2.0
        for normal in face_normals:
            n2d = np.array([normal[0], normal[1]])
            n2d_len = np.linalg.norm(n2d)
            if n2d_len < 0.01:
                continue
            n2d = n2d / n2d_len
            dot = np.dot(n2d, to_robot_unit[:2])
            if dot > best_dot:
                best_dot = dot
                best_face = n2d

        if best_face is None:
            print("Could not determine approach face")
            return

        ux, uy = best_face[0], best_face[1]

        approach_x = cube_x + ux * STEP1_DIST
        approach_y = cube_y + uy * STEP1_DIST
        dist_robot_to_approach = math.sqrt(
            (approach_x - robot_x) ** 2 + (approach_y - robot_y) ** 2
        )
        dist_robot_to_cube = math.sqrt(
            (cube_x - robot_x) ** 2 + (cube_y - robot_y) ** 2
        )
        if dist_robot_to_approach > dist_robot_to_cube:
            ux, uy = -ux, -uy

        heading = math.atan2(-uy, -ux)

        step1_x = cube_x + ux * STEP1_DIST
        step1_y = cube_y + uy * STEP1_DIST

        print(
            f"Approaching cube {cube_idx + 1} at ({cube_x:.0f}, {cube_y:.0f}) "
            f"face normal=({ux:.2f}, {uy:.2f})"
        )

        HEAD_TILT_DEG = -10.0
        self.brain.cli.set_head_angle(
            math.radians(HEAD_TILT_DEG), accel=30.0, max_speed=30.0
        )
        print(f"Head tilt {HEAD_TILT_DEG:.0f} deg")

        def _nav_thread():
            try:
                if need_back_away:
                    back_heading = math.atan2(robot_y - cube_y, robot_x - cube_x)
                    back_x = cube_x + (robot_x - cube_x) / dist_to_cube * BACK_AWAY_DIST
                    back_y = cube_y + (robot_y - cube_y) / dist_to_cube * BACK_AWAY_DIST
                    pose_back = Pose(
                        back_x, back_y, 0, angle_z=Angle(radians=back_heading)
                    )
                    print(f"Backing away to ({back_x:.0f}, {back_y:.0f})")
                    self.brain.cli.go_to_pose(pose_back, relative_to_robot=False)
                    time.sleep(0.3)

                pose1 = Pose(step1_x, step1_y, 0, angle_z=Angle(radians=heading))
                print(f"Step 1: 120mm ({step1_x:.0f}, {step1_y:.0f})")
                self.brain.cli.go_to_pose(pose1, relative_to_robot=False)

                time.sleep(0.3)

                world2 = self.brain.web_viewer.world_map.get_world_state()
                cubes2 = world2.get("cubes", {})
                if cube_idx in cubes2:
                    cube_x2 = float(cubes2[cube_idx]["position"][0])
                    cube_y2 = float(cubes2[cube_idx]["position"][1])
                    cube_rot2 = np.array(cubes2[cube_idx]["rotation"])

                    local_x2 = cube_rot2 @ np.array([1.0, 0.0, 0.0])
                    local_y2 = cube_rot2 @ np.array([0.0, 1.0, 0.0])
                    normals2 = [local_x2, -local_x2, local_y2, -local_y2]

                    rp = self.brain.cli.pose
                    to_robot2 = np.array(
                        [rp.position.x - cube_x2, rp.position.y - cube_y2, 0.0]
                    )
                    to_robot2_len = np.linalg.norm(to_robot2)
                    if to_robot2_len > 1:
                        to_robot2_unit = to_robot2 / to_robot2_len
                        best2 = None
                        best_dot2 = -2.0
                        for n in normals2:
                            n2d = np.array([n[0], n[1]])
                            if np.linalg.norm(n2d) < 0.01:
                                continue
                            n2d = n2d / np.linalg.norm(n2d)
                            d = np.dot(n2d, to_robot2_unit[:2])
                            if d > best_dot2:
                                best_dot2 = d
                                best2 = n2d
                        if best2 is not None:
                            ux2, uy2 = best2[0], best2[1]
                            app2_x = cube_x2 + ux2 * STEP2_DIST
                            app2_y = cube_y2 + uy2 * STEP2_DIST
                            d_app = math.sqrt(
                                (app2_x - rp.position.x) ** 2
                                + (app2_y - rp.position.y) ** 2
                            )
                            d_cube = math.sqrt(
                                (cube_x2 - rp.position.x) ** 2
                                + (cube_y2 - rp.position.y) ** 2
                            )
                            if d_app > d_cube:
                                ux2, uy2 = -ux2, -uy2
                            step2_x = cube_x2 + ux2 * STEP2_DIST
                            step2_y = cube_y2 + uy2 * STEP2_DIST
                            heading2 = math.atan2(-uy2, -ux2)
                            print(
                                f"Updated: cube at ({cube_x2:.0f}, {cube_y2:.0f}) face=({ux2:.2f}, {uy2:.2f})"
                            )

                pose2 = Pose(step2_x, step2_y, 0, angle_z=Angle(radians=heading2))
                print(f"Step 2: 80mm ({step2_x:.0f}, {step2_y:.0f})")
                self.brain.cli.go_to_pose(pose2, relative_to_robot=False)

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
