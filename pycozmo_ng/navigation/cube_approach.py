"""CubeApproach — two-step standoff approach and grab alignment.

Extracted from Navigator.approach_cube_at (COZMO-40).  The class takes a
robot client, a cube-state reader, and two driving callbacks so it can be
unit-tested without live hardware.
"""

import math
import threading
import time

import numpy as np

import pycozmo_ng
from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.cube_visit import CubeVisit
from pycozmo_ng.navigation.motion_token import MotionToken
from pycozmo_ng.navigation.steering import wrap_pi
from pycozmo_ng.util import Angle, Pose


class CubeApproach:
    """Drive the robot to a cube, align with the best face, and grab it.

    Constructor args (dependency injection):
        cli             Robot client — pose, go_to_pose, drive_wheels,
                        set_lift_height, set_head_angle, stop_all_motors.
        get_cube_state  ``Callable[[int], CubeState | None]`` — returns the
                        fused world-frame state for a given cube index.
        drive_to_point  ``Callable[[tx, ty, *, stop_mm, timeout_s, ...], bool]``
                        Closed-loop drive to a point (Navigator.drive_to_point).
        move_forward    ``Callable[[dist_mm], bool]`` Straight-ahead creep
                        (Navigator._move_forward).
        visit           The shared :class:`CubeVisit` — starts the approach
                        on a background motion job (``MotionController.spawn``
                        behind it, which is what lets ``Navigator.stop``
                        cancel it) and serializes it against a stack.
                        Navigator builds one and hands the same instance to
                        :class:`CubeStack`, so the two can never interleave.
    """

    #: Motion job name this class registers while an approach runs.
    JOB_APPROACH: str = "approach"

    # ------------------------------------------------------------------
    # Tuning constants
    # ------------------------------------------------------------------
    STEP1_DIST_MM = 160.0
    STEP2_DIST_MM = 100.0
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
    #: How long to dwell at the step-1 standoff so the world-map EMA/SLERP
    #: filters converge to a stable cube position/attitude before step 2
    #: commits to the final approach line.
    STANDOFF_DWELL_S = 1.5
    ARRIVE_POS_MM = 8.0
    ARRIVE_ANG_RAD = math.radians(2.0)

    def __init__(self, cli, get_cube_state, drive_to_point, move_forward, visit: CubeVisit, carry=None):
        self.cli = cli
        self._get_cube_state = get_cube_state
        self._drive_to_point = drive_to_point
        self._move_forward = move_forward
        # The shared cube-visit seam: owns the "one visit at a time"
        # lock and starts the background job. Public (and shared with
        # CubeStack) so tests can ask whether a visit is in flight.
        self.visit = visit
        # Carry observation (Brain's CarryDetector) — cleared when the
        # forks go down, armed after a grab so "cube out of frame" means
        # "cube on the lift". Optional: a standalone approach works without.
        self.carry = carry

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

    def _dwell_for_smoothing(self, cube_idx: int, job: MotionToken) -> bool:
        """Dwell at the standoff so the world-map filters converge.

        The fused cube pose is updated once per camera frame; a single fresh
        frame barely moves the confidence-weighted EMA (position) / SLERP
        (rotation) estimate.  Waiting ``STANDOFF_DWELL_S`` lets dozens of
        frames accumulate, so step 2 corrects off a stable, converged pose
        instead of the first noisy frame after arrival.

        Returns True if the cube was re-detected during the dwell (a fresh
        measurement was fused), False otherwise — the caller should cancel
        the approach when the cube is not visible at the standoff. The dwell
        also ends early when ``job`` is cancelled.
        """
        before = self._get_cube_state(cube_idx)
        base_ts = before.updated_at if before is not None else None
        saw_fresh = False
        t0 = time.monotonic()
        while not job.cancelled and time.monotonic() - t0 < self.STANDOFF_DWELL_S:
            time.sleep(0.05)
            s = self._get_cube_state(cube_idx)
            ts = s.updated_at if s is not None else None
            if ts is not None and (base_ts is None or ts > base_ts + 0.05):
                saw_fresh = True
        return saw_fresh

    def _watch_near_tilt(self, cube_idx: int, job: MotionToken) -> None:
        """Drop head to ``NEAR_TILT_DEG`` once within proximity."""
        while not job.cancelled:
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
                    logger.info("Head tilt %.0f deg (dist %.0fmm)", self.NEAR_TILT_DEG, d)
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
        msgs.append(f"fresh={'ok' if fresh else 'STALE'} (age {age:.1f}s, conf {s.confidence:.2f})")

        f = self.pick_face_normal(s.position, s.rotation_matrix, rpos, ryaw)
        facing_err_rad = None if f is None else abs(wrap_pi(f[2] - ryaw))
        facing = facing_err_rad is not None and facing_err_rad <= self.GRAB_HEADING_TOL_RAD
        if facing_err_rad is None:
            msgs.append("facing=FAIL (no face normal resolved)")
        else:
            msgs.append(f"facing={'ok' if facing else 'WRONG'} (err {math.degrees(facing_err_rad):.1f} deg)")

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

    def approach(self, cube_idx: int, cube_x: float, cube_y: float) -> bool:
        """Two-step approach & grab for the given cube.

        Runs as one background ``approach`` motion job so the caller
        (WebSocket handler) returns immediately and ``Navigator.stop``
        cancels the whole sequence — including the drives nested inside
        it. Only one cube visit runs at a time: while a grab or a stack
        is in flight the request is ignored.

        Returns True when the job started, False when it was refused.
        """
        started = self.visit.begin(
            self.JOB_APPROACH,
            lambda job: self._run(job, cube_idx, cube_x, cube_y),
        )
        if not started:
            logger.warning("Approach busy - ignoring request for cube %d", to_cube_id(cube_idx))
        return started

    def _run(self, job: MotionToken, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Blocking implementation — runs inside the ``approach`` job."""
        try:
            aligned = self.drive_and_align(job, cube_idx, cube_x, cube_y, lower_forks=True)
            if job.cancelled:
                logger.info("Approach cancelled - not grabbing")
                self.cli.stop_all_motors()
                return
            if not aligned:
                return
            self._grab(cube_idx)
        except Exception as e:  # noqa: BLE001
            logger.exception("Approach error: %s", e)

    def drive_and_align(
        self,
        job: MotionToken,
        cube_idx: int,
        cube_x: float,
        cube_y: float,
        *,
        lower_forks: bool = True,
    ) -> bool:
        """Drive to the cube, align with its face, and verify the pose.

        The shared prelude of a grab (:class:`CubeApproach`) and a stack
        (:class:`CubeStack`), run inside the caller's motion job. With
        ``lower_forks=True`` the forks are dropped first (forgetting any
        carried cube) — the grab prelude; a stack passes ``False`` so its
        cube keeps riding the raised forks to the target.

        Returns True when aligned and ready for the caller's terminal
        action (grab or place); False when the cube is unknown, the
        drive was cancelled, or the alignment checks failed.
        """
        state = self._get_cube_state(cube_idx)
        if state is None:
            logger.warning("Cube %d not in world map", to_cube_id(cube_idx))
            return False

        robot_pose = self.cli.pose
        rx, ry = robot_pose.position.x, robot_pose.position.y
        robot_yaw = robot_pose.rotation.angle_z.radians

        face = self.pick_face_normal(state.position, state.rotation_matrix, (rx, ry), robot_yaw)
        if face is None:
            logger.warning("Could not determine approach face")
            return False
        nx, ny, heading = face

        logger.info(
            "Approaching cube %d at (%.0f, %.0f) face normal=(%.2f, %.2f)",
            to_cube_id(cube_idx),
            cube_x,
            cube_y,
            nx,
            ny,
        )

        if lower_forks:
            # Forks down
            self.cli.set_lift_height(
                pycozmo_ng.MIN_LIFT_HEIGHT.mm,
                accel=self.LIFT_UP_ACCEL,
                max_speed=self.LIFT_UP_SPEED,
            )
            logger.info("Lift lowered (%.0fmm)", pycozmo_ng.MIN_LIFT_HEIGHT.mm)
            if self.carry is not None:
                # Forks down: nothing rides them — forget any carry state.
                self.carry.clear()

        self.cli.set_head_angle(math.radians(self.HEAD_TILT_DEG), accel=30.0, max_speed=30.0)
        logger.info("Head tilt %.0f deg", self.HEAD_TILT_DEG)

        # Back away if too close
        dist_to_cube = math.hypot(cube_x - rx, cube_y - ry)
        if dist_to_cube < self.TOO_CLOSE_MM:
            bx = cube_x + (rx - cube_x) / dist_to_cube * self.BACK_AWAY_DIST_MM
            by = cube_y + (ry - cube_y) / dist_to_cube * self.BACK_AWAY_DIST_MM
            logger.info("Too close (%.0fmm) - backing away", dist_to_cube)
            self._drive_to_point(bx, by, timeout_s=8.0)

        time.sleep(0.3)
        if job.cancelled:
            logger.info("Approach cancelled")
            self.cli.stop_all_motors()
            return False

        threading.Thread(
            target=self._watch_near_tilt,
            args=(cube_idx, job),
            daemon=True,
        ).start()

        # Step 1: view standoff
        s1x, s1y = self._standoff(state, self.STEP1_DIST_MM, (nx, ny))
        logger.info("Step 1: go_to_pose to view standoff (%.0f, %.0f)", s1x, s1y)
        self.cli.go_to_pose(
            Pose(s1x, s1y, 0, angle_z=Angle(radians=heading)),
            relative_to_robot=False,
            should_interrupt=lambda: job.cancelled,
        )

        if job.cancelled:
            logger.info("Approach cancelled")
            self.cli.stop_all_motors()
            return False
        if not self._dwell_for_smoothing(cube_idx, job):
            self.cli.stop_all_motors()
            logger.warning("Cube not re-detected at standoff - cancelling approach")
            return False

        # Step 2: corrected standoff
        target2 = self._corrected_target(cube_idx, self.STEP2_DIST_MM)
        if target2 is None:
            logger.warning("Cube lost after step 1 - keeping last standoff")
            s2x, s2y, heading2 = s1x, s1y, heading
        else:
            s2x, s2y, heading2 = target2
        logger.info(
            "Step 2: go_to_pose to %.0fmm standoff (%.0f, %.0f)",
            self.STEP2_DIST_MM,
            s2x,
            s2y,
        )
        self.cli.go_to_pose(
            Pose(s2x, s2y, 0, angle_z=Angle(radians=heading2)),
            relative_to_robot=False,
            should_interrupt=lambda: job.cancelled,
        )

        # Wait for arrival
        t_arrive = time.monotonic()
        while not job.cancelled and time.monotonic() - t_arrive < 10.0:
            rp = self.cli.pose
            pos_err = math.hypot(s2x - rp.position.x, s2y - rp.position.y)
            ang_err = abs(wrap_pi(heading2 - rp.rotation.angle_z.radians))
            if pos_err <= self.ARRIVE_POS_MM and ang_err <= self.ARRIVE_ANG_RAD:
                break
            time.sleep(0.05)

        if job.cancelled:
            logger.info("Approach cancelled")
            self.cli.stop_all_motors()
            return False

        logger.info("Aligned with cube %d", to_cube_id(cube_idx))

        # Alignment verification with retry
        attempts = 0
        while not job.cancelled:
            (aligned, checks, facing_err_rad, forward, lateral, _, fresh) = self._verify_grab_alignment(cube_idx)
            for msg in checks:
                logger.info("check: %s", msg)

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
            assert facing_err_rad is not None  # guaranteed by recoverable
            logger.info(
                "Misaligned by %.1f deg - creeping %.0fmm closer (%d/%d)",
                math.degrees(facing_err_rad),
                self.GRAB_RECHECK_MM,
                attempts,
                self.GRAB_MAX_RECHECKS,
            )
            self._move_forward(self.GRAB_RECHECK_MM)
            time.sleep(0.3)

        if job.cancelled:
            logger.info("Approach cancelled")
            self.cli.stop_all_motors()
            return False
        if not aligned:
            logger.warning("Alignment check FAILED - not continuing")
            return False
        return True

    def _grab(self, cube_idx: int) -> None:
        """Terminal action of an approach: creep in, raise the forks, arm carry."""
        logger.info(
            "Aligned - grabbing cube %d: +%.0fmm, lift to %.0fmm",
            to_cube_id(cube_idx),
            self.GRAB_FORWARD_MM,
            pycozmo_ng.MAX_LIFT_HEIGHT.mm,
        )
        self._move_forward(self.GRAB_FORWARD_MM)
        self.cli.set_lift_height(
            pycozmo_ng.MAX_LIFT_HEIGHT.mm,
            accel=self.LIFT_UP_ACCEL,
            max_speed=self.LIFT_UP_SPEED,
        )
        logger.info("Cube %d grabbed (lift up)", to_cube_id(cube_idx))
        if self.carry is not None:
            # Arm the observation: while the forks are up the cube
            # should leave the camera frame; if it never does, the
            # grab missed and we are not carrying it.
            self.carry.note_grab(cube_idx)
