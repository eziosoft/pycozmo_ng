"""ArucoDocking — two-step standoff approach to the ArUco charger marker, then reverse in.

Mirrors :class:`~pycozmo_ng.navigation.cube_approach.CubeApproach`:
drive to a view standoff, dwell so the world-map fusion converges, then
drive to a corrected closer standoff, verify alignment, and *reverse*
into the charger until charging is detected (instead of grabbing the
cube with the forks).

On success the caller is expected to reset the pose origin (the docked
robot pose becomes the world origin — the point 65 mm in front of the
marker) and snap marker 0's world-map pose to its known docked position
(65 mm behind the camera).
"""

from __future__ import annotations

import math
import time

from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.motion_controller import MotionController
from pycozmo_ng.navigation.motion_token import MotionToken
from pycozmo_ng.navigation.steering import wrap_pi
from pycozmo_ng.util import Angle, Pose


class ArucoDocking:
    """Drive to the ArUco charger marker and reverse into the charger.

    Constructor args (dependency injection):

        cli:
            Robot client — ``pose``, ``go_to_pose``, ``drive_wheels``,
            ``stop_all_motors``.
        get_marker_state:
            ``Callable[[int], ArucoState | None]`` — returns the fused
            world-frame state for a given marker ID.
        dock_state:
            Shared :class:`DockState` — ``on_charger`` / ``charging``
            flags read during the reverse phase.
        motion:
            The shared :class:`MotionController` — spawns the dock job so
            :meth:`Navigator.stop` cancels it.
        move_forward:
            ``Callable[[dist_mm], bool]`` — straight-ahead creep for
            alignment rechecks (``Navigator._move_forward``).
    """

    JOB_ARUCO_DOCK: str = "aruco_dock"
    MARKER_ID: int = 0

    # -- tuning constants --------------------------------------------------
    STEP1_DIST_MM: float = 300.0  # view standoff in front of marker
    STEP2_DIST_MM: float = 150.0  # corrected standoff (user: 15 cm)
    STANDOFF_DWELL_S: float = 1.5

    ARRIVE_POS_MM: float = 30.0
    ARRIVE_ANG_RAD: float = math.radians(5.0)

    HEAD_TILT_DEG: float = -5.0

    # Alignment verification
    CENTER_TOL_MM: float = 30.0
    HEADING_TOL_RAD: float = math.radians(5.0)

    # Recheck creep
    RECHECK_MM: float = 10.0
    MAX_RECHECKS: int = 3

    # Reverse into charger
    REVERSE_SPEED_MMPS: float = -60.0  # negative = backward
    REVERSE_TIMEOUT_S: float = 15.0

    # Docked known offset
    DOCKED_MARKER_DIST_MM: float = 65.0

    def __init__(
        self,
        cli,
        get_marker_state,
        dock_state: DockState,
        motion: MotionController,
        move_forward,
    ):
        self.cli = cli
        self._get_marker_state = get_marker_state
        self.ds = dock_state
        self.motion = motion
        self._move_forward = move_forward

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def dock(self, marker_id: int = 0) -> bool:
        """Two-step standoff + reverse dock in the background.

        One dock job at a time; a new request supersedes the old one.
        Cancel via :meth:`motion.cancel` / ``Navigator.stop``.

        For post-dock actions (reset origin, snap marker pose) pass
        ``on_docked`` to the constructor; the callable runs on the same
        daemon thread that finishes the dock sequence.

        Returns:
            True when the dock job was started, False when the marker is
            not yet known (not in the world map).
        """
        if self._get_marker_state(marker_id) is None:
            return False
        self.motion.cancel(self.JOB_ARUCO_DOCK)
        self.motion.spawn(self.JOB_ARUCO_DOCK, lambda job: self._run(job, marker_id))
        return True

    def run(self, marker_id: int = 0) -> bool:
        """Blocking dock sequence for callers that need the result.

        Unlike :meth:`dock` (which spawns a background job), this method
        runs synchronously in the current thread — useful for wrapper
        code in ``Navigator`` that wants to act on the docked outcome
        (reset origin, snap marker pose) without a callback.

        Returns:
            True when the robot docked and detected charging,
            False on cancel, timeout, or alignment failure.
        """
        with self.motion.begin(f"{self.JOB_ARUCO_DOCK}_sync") as job:
            self._run(job, marker_id)
        return self._last_docked

    # ------------------------------------------------------------------
    # Job body
    # ------------------------------------------------------------------

    _last_docked: bool = False

    def _run(self, job: MotionToken, marker_id: int) -> None:
        """Blocking dock sequence — runs inside the dock motion job."""
        self._last_docked = False
        try:
            aligned = self._drive_and_align(job, marker_id)
            if job.cancelled or not aligned:
                return
            docked = self._reverse_until_charge(job, marker_id)
            if docked:
                self._last_docked = True
        except Exception:  # noqa: BLE001
            logger.exception("ArUco dock error")

    def _drive_and_align(self, job: MotionToken, marker_id: int) -> bool:
        """Drive to the marker, align with its normal, verify the pose.

        Returns True when aligned and ready for the reverse terminal
        action; False when the marker is lost, drive cancelled, or
        alignment checks failed.
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            logger.warning("ArUco dock: marker %d not in world map", marker_id)
            return False

        # Marker normal (z-axis = out of printed face toward robot)
        rot = state.rotation_matrix
        nx, ny = float(rot[0, 2]), float(rot[1, 2])
        n_len = math.hypot(nx, ny)
        if n_len < 1e-6:
            logger.warning("ArUco dock: marker normal has no horizontal component")
            return False
        nx /= n_len
        ny /= n_len

        # Heading toward the marker (robot faces marker at the standoff)
        heading = math.atan2(-ny, -nx)

        rp = self.cli.pose
        logger.info(
            "ArUco dock: marker %d at (%.0f, %.0f) normal=(%.2f, %.2f)",
            marker_id,
            state.position[0],
            state.position[1],
            nx,
            ny,
        )

        # Head tilt to help detection
        self.cli.set_head_angle(
            math.radians(self.HEAD_TILT_DEG),
            accel=30.0,
            max_speed=30.0,
        )

        # ----
        # Step 1: view standoff
        # ----
        cx, cy = float(state.position[0]), float(state.position[1])
        s1x = cx + nx * self.STEP1_DIST_MM
        s1y = cy + ny * self.STEP1_DIST_MM
        logger.info("ArUco dock step 1: go_to_pose to view standoff (%.0f, %.0f)", s1x, s1y)
        self.cli.go_to_pose(
            Pose(s1x, s1y, 0, angle_z=Angle(radians=heading)),
            relative_to_robot=False,
            should_interrupt=lambda: job.cancelled,
        )
        if job.cancelled:
            logger.info("ArUco dock cancelled")
            self.cli.stop_all_motors()
            return False

        # Dwell for world-map convergence
        if not self._dwell_for_smoothing(marker_id, job):
            self.cli.stop_all_motors()
            logger.warning("ArUco dock: marker not re-detected at standoff — cancelling")
            return False

        # ----
        # Step 2: corrected standoff
        # ----
        target2 = self._corrected_target(marker_id)
        if target2 is None:
            logger.warning("ArUco dock: marker lost after step 1 — keeping last standoff")
            s2x, s2y, heading2 = s1x, s1y, heading
        else:
            s2x, s2y, _nx, _ny, heading2 = target2

        logger.info("ArUco dock step 2: go_to_pose to %.0f mm (%.0f, %.0f)", self.STEP2_DIST_MM, s2x, s2y)
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
            logger.info("ArUco dock cancelled")
            self.cli.stop_all_motors()
            return False

        logger.info("ArUco dock: aligned with marker %d", marker_id)

        # ----
        # Alignment verification with creep-and-recheck
        # ----
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        attempts = 0
        while not job.cancelled:
            aligned, checks, facing_ok, centered_ok, forward_ok = self._verify_alignment(
                state,
                heading2,
                marker_id,
            )
            for msg in checks:
                logger.info("check: %s", msg)

            recoverable = (
                not aligned
                and facing_ok
                and centered_ok
                and forward_ok is not None
                and forward_ok is not False  # forward_ok may be None
                and attempts < self.MAX_RECHECKS
            )
            if not recoverable:
                break

            attempts += 1
            logger.info(
                "ArUco dock: creep %.0f mm forward for recheck (%d/%d)",
                self.RECHECK_MM,
                attempts,
                self.MAX_RECHECKS,
            )
            self._move_forward(self.RECHECK_MM)
            time.sleep(0.3)
            state = self._get_marker_state(marker_id)
            if state is None:
                return False

        if job.cancelled:
            self.cli.stop_all_motors()
            return False

        if not aligned:
            logger.warning("ArUco dock: alignment check FAILED — not continuing")
            return False

        return True

    # ------------------------------------------------------------------
    # Terminal action: reverse into the charger
    # ------------------------------------------------------------------

    def _reverse_until_charge(self, job: MotionToken, marker_id: int) -> bool:
        """Reverse slowly at the standoff until charging is detected.

        The robot faces the marker at the ~15 cm standoff.  We first turn
        it 180° so the rear faces the charger, then reverse (drive_wheels
        negative) until the charger contacts detect or a timeout fires.

        Returns True when charging was detected, False on cancel/timeout.

        .. note::

            During the reverse the marker is behind the robot and out of
            camera view — the sequence relies on the charging event, not
            vision, to stop.
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        rot = state.rotation_matrix
        nx, ny = float(rot[0, 2]), float(rot[1, 2])
        n_len = math.hypot(nx, ny)
        if n_len < 1e-6:
            return False
        nx /= n_len
        ny /= n_len

        # Turn to face AWAY from the marker (rear toward charger)
        away_heading = math.atan2(ny, nx)
        logger.info("ArUco dock: turning to face away (heading %.1f°)", math.degrees(away_heading))
        self.cli.go_to_pose(
            Pose(
                self.cli.pose.position.x,
                self.cli.pose.position.y,
                0,
                angle_z=Angle(radians=away_heading),
            ),
            relative_to_robot=False,
            should_interrupt=lambda: job.cancelled,
        )
        if job.cancelled:
            return False

        # Reverse until charging
        logger.info("ArUco dock: reversing into charger")
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.REVERSE_TIMEOUT_S:
            if job.cancelled:
                return False
            with self.ds.lock:
                on_contact = self.ds.on_charger or self.ds.charging
            if on_contact:
                logger.info("ArUco dock: charging detected!")
                return True
            self.cli.drive_wheels(
                lwheel_speed=self.REVERSE_SPEED_MMPS,
                rwheel_speed=self.REVERSE_SPEED_MMPS,
            )
            time.sleep(0.05)

        self.cli.stop_all_motors()
        # One last check
        with self.ds.lock:
            if self.ds.on_charger or self.ds.charging:
                logger.info("ArUco dock: charging (late)")
                return True
        logger.warning("ArUco dock: reverse timed out (%.0f s)", self.REVERSE_TIMEOUT_S)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _standoff(
        self,
        state,
        dist_mm: float,
        nx: float,
        ny: float,
    ) -> tuple[float, float]:
        r"""Marker centre + normal × *dist\ * in XY plane."""
        return (
            float(state.position[0]) + nx * dist_mm,
            float(state.position[1]) + ny * dist_mm,
        )

    def _corrected_target(self, marker_id: int):
        """Recompute approach line from the (now-fused) marker pose."""
        s = self._get_marker_state(marker_id)
        if s is None:
            return None
        rot = s.rotation_matrix
        nx = float(rot[0, 2])
        ny = float(rot[1, 2])
        n_len = math.hypot(nx, ny)
        if n_len < 1e-6:
            return None
        nx /= n_len
        ny /= n_len
        heading = math.atan2(-ny, -nx)
        sx, sy = self._standoff(s, self.STEP2_DIST_MM, nx, ny)
        return sx, sy, nx, ny, heading

    def _dwell_for_smoothing(self, marker_id: int, job: MotionToken) -> bool:
        """Dwell so the world-map filters converge on the marker pose.

        Returns True if the marker was re-detected during the dwell,
        False otherwise — the caller should cancel when the marker is
        not visible at the standoff.
        """
        before = self._get_marker_state(marker_id)
        base_ts = before.updated_at if before is not None else None
        saw_fresh = False
        t0 = time.monotonic()
        while not job.cancelled and time.monotonic() - t0 < self.STANDOFF_DWELL_S:
            time.sleep(0.05)
            s = self._get_marker_state(marker_id)
            ts = s.updated_at if s is not None else None
            if ts is not None and (base_ts is None or ts > base_ts + 0.05):
                saw_fresh = True
        return saw_fresh

    def _verify_alignment(self, state, expected_heading: float, marker_id: int):
        """Three checks: facing (heading err), centered (lateral), in range.

        Returns (passed, msgs, facing_ok, centered_ok, forward_ok_radius).
        """
        rp = self.cli.pose
        ryaw = rp.rotation.angle_z.radians
        msgs: list[str] = []

        facing_err = abs(wrap_pi(expected_heading - ryaw))
        facing_ok = facing_err <= self.HEADING_TOL_RAD
        msgs.append(f"facing={'ok' if facing_ok else 'WRONG'} ({math.degrees(facing_err):.1f}°)")

        dx = float(state.position[0]) - rp.position.x
        dy = float(state.position[1]) - rp.position.y
        fwdx, fwdy = math.cos(ryaw), math.sin(ryaw)
        forward = dx * fwdx + dy * fwdy
        lateral = dx * -fwdy + dy * fwdx
        centered_ok = abs(lateral) <= self.CENTER_TOL_MM
        msgs.append(f"centered={'ok' if centered_ok else 'OFF'} (lat {lateral:.0f} mm, fwd {forward:.0f} mm)")

        passed = facing_ok and centered_ok
        return passed, msgs, facing_ok, centered_ok, forward
