"""ArucoDocking — dock on the ArUco charger marker (marker 0).

The sequence, mirroring the cube approach:

1. Drive to a point in front of the marker, facing it — the camera sees
   the marker and the world map refreshes its pose.
2. Correct the approach point from the refreshed pose if it moved.
3. Turn 180° so the rear faces the charger and reverse in until the
   charging contacts detect (with a timeout).

On success ``on_docked`` runs — the caller resets the pose origin (the
docked robot pose becomes the world origin, 65 mm in front of the
marker) and snaps the marker to its known docked position.
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
            ``Callable[[int], ArucoState | None]`` — the fused world-frame
            marker state from the world map.
        dock_state:
            Shared :class:`DockState` — ``on_charger`` / ``charging``
            flags read during the reverse phase.
        motion:
            The shared :class:`MotionController` — spawns the dock job so
            ``Navigator.stop()`` cancels it.
        on_docked:
            Optional callback run once charging is detected (reset
            origin, snap marker pose).
    """

    JOB_ARUCO_DOCK: str = "aruco_dock"

    APPROACH_DIST_MM: float = 200.0  # standoff in front of the marker
    SETTLE_S: float = 0.5  # time for the camera to refresh the marker pose

    #: Below this, a move is a pure turn — never send a zero-length path
    #: segment: the firmware asserts ``y != 0 || x != 0`` in
    #: ``trig_fast.cpp`` (``atan2(0, 0)``) and crashes.
    MIN_SEG_MM: float = 15.0

    TURN_SPEED_MMPS: float = 60.0
    TURN_TOL_RAD: float = math.radians(4.0)
    TURN_TIMEOUT_S: float = 5.0

    REVERSE_SPEED_MMPS: float = -60.0  # negative = backward
    REVERSE_TIMEOUT_S: float = 15.0

    #: Marker 0 is this far behind the camera when the robot is docked —
    #: the offset the world origin and the post-dock marker snap use.
    DOCKED_MARKER_DIST_MM: float = 65.0

    def __init__(
        self,
        cli,
        get_marker_state,
        dock_state: DockState,
        motion: MotionController,
        on_docked=None,
    ):
        self.cli = cli
        self._get_marker_state = get_marker_state
        self.ds = dock_state
        self.motion = motion
        self._on_docked = on_docked

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def dock(self, marker_id: int = 0) -> bool:
        """Run the dock sequence in the background.

        A new request supersedes the old one; cancel via
        ``motion.cancel`` / ``Navigator.stop``.

        Returns:
            True when the dock job was started, False when the marker is
            not yet in the world map.
        """
        if self._get_marker_state(marker_id) is None:
            return False
        self.motion.cancel(self.JOB_ARUCO_DOCK)
        self.motion.spawn(self.JOB_ARUCO_DOCK, lambda job: self._run(job, marker_id))
        return True

    # ------------------------------------------------------------------
    # Job body
    # ------------------------------------------------------------------

    def _run(self, job: MotionToken, marker_id: int) -> None:
        """Dock sequence — runs inside the ``aruco_dock`` motion job."""
        try:
            if not self._approach(job, marker_id):
                return
            if not self._reverse_into_charger(job, marker_id):
                return
            logger.info("ArUco dock: docked")
            if self._on_docked is not None:
                self._on_docked()
        except Exception:  # noqa: BLE001
            logger.exception("ArUco dock error")

    # ------------------------------------------------------------------
    # 1-2. Come close, see the marker, update the approach point
    # ------------------------------------------------------------------

    def _approach(self, job: MotionToken, marker_id: int) -> bool:
        """Drive to the marker, let the camera refresh its pose, correct."""
        target = self._approach_target(marker_id)
        if target is None:
            logger.warning("ArUco dock: marker %d not in world map", marker_id)
            return False

        x, y, heading = target
        logger.info("ArUco dock: approaching marker %d at (%.0f, %.0f)", marker_id, x, y)
        if not self._go(job, x, y, heading):
            return False

        # Settle — the camera sees the marker and the world map refreshes
        # its fused pose.
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.SETTLE_S:
            if job.cancelled:
                self.cli.stop_all_motors()
                return False
            time.sleep(0.05)

        # If the refreshed pose moved the approach point, correct it
        # (``_go`` is a no-op drive + turn when already in place).
        corrected = self._approach_target(marker_id)
        if corrected is None:
            return False
        cx, cy, cheading = corrected
        if math.hypot(cx - x, cy - y) > self.MIN_SEG_MM:
            logger.info("ArUco dock: marker pose updated — correcting to (%.0f, %.0f)", cx, cy)
        return self._go(job, cx, cy, cheading)

    def _approach_target(self, marker_id: int):
        """(x, y, heading) of the point in front of the marker, facing it."""
        state = self._get_marker_state(marker_id)
        if state is None:
            return None
        nx, ny = self._normal(state)
        if nx is None:
            return None
        x = float(state.position[0]) + nx * self.APPROACH_DIST_MM
        y = float(state.position[1]) + ny * self.APPROACH_DIST_MM
        heading = math.atan2(-ny, -nx)  # face the marker
        return x, y, heading

    # ------------------------------------------------------------------
    # 3. Turn around and reverse into the marker
    # ------------------------------------------------------------------

    def _reverse_into_charger(self, job: MotionToken, marker_id: int) -> bool:
        """Turn 180° so the rear faces the charger, reverse until charging."""
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        nx, ny = self._normal(state)
        if nx is None:
            return False

        # Turn in place to face away from the marker — rear toward charger.
        pose = self.cli.pose
        if not self._go(job, pose.position.x, pose.position.y, math.atan2(ny, nx)):
            return False

        logger.info("ArUco dock: reversing into the charger")
        deadline = time.monotonic() + self.REVERSE_TIMEOUT_S
        while time.monotonic() < deadline:
            if job.cancelled:
                self.cli.stop_all_motors()
                return False
            if self._charging():
                self.cli.stop_all_motors()
                return True
            self.cli.drive_wheels(
                lwheel_speed=self.REVERSE_SPEED_MMPS,
                rwheel_speed=self.REVERSE_SPEED_MMPS,
            )
            time.sleep(0.05)

        self.cli.stop_all_motors()
        if self._charging():  # contacts landed as we stopped
            return True
        logger.warning("ArUco dock: reverse timed out (%.0f s)", self.REVERSE_TIMEOUT_S)
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normal(self, state):
        """Unit horizontal marker normal (z-axis = out of the printed face)."""
        rot = state.rotation_matrix
        nx, ny = float(rot[0, 2]), float(rot[1, 2])
        length = math.hypot(nx, ny)
        if length < 1e-6:
            logger.warning("ArUco dock: marker normal has no horizontal component")
            return None, None
        return nx / length, ny / length

    def _go(self, job: MotionToken, x: float, y: float, heading: float) -> bool:
        """Drive/turn to (*x*, *y*, *heading*) with cancellation.

        A zero-length path segment crashes the firmware (``atan2(0, 0)``
        → ``trig_fast.cpp`` assert), so when the robot is already at the
        target only the turn runs — via :meth:`_turn_to`, the same
        closed-loop ``drive_wheels`` turn ``Navigator._align_heading``
        uses.
        """
        rp = self.cli.pose
        if math.hypot(x - rp.position.x, y - rp.position.y) > self.MIN_SEG_MM:
            self.cli.go_to_pose(
                Pose(x, y, 0, angle_z=Angle(radians=heading)),
                relative_to_robot=False,
                should_interrupt=lambda: job.cancelled,
            )
            if job.cancelled:
                logger.info("ArUco dock cancelled")
                self.cli.stop_all_motors()
                return False
        return self._turn_to(job, heading)

    def _turn_to(self, job: MotionToken, heading: float) -> bool:
        """Turn in place to *heading* — closed loop on the yaw error.

        Returns False when cancelled or when the heading did not
        converge within the timeout.
        """
        cli = self.cli
        deadline = time.monotonic() + self.TURN_TIMEOUT_S
        while time.monotonic() < deadline:
            if job.cancelled:
                logger.info("ArUco dock cancelled")
                cli.stop_all_motors()
                return False
            err = wrap_pi(heading - cli.pose.rotation.angle_z.radians)
            if abs(err) < self.TURN_TOL_RAD:
                cli.stop_all_motors()
                return True
            omega = max(-1.0, min(1.0, err * 2.2)) * self.TURN_SPEED_MMPS
            cli.drive_wheels(lwheel_speed=-omega, rwheel_speed=omega)
            time.sleep(0.03)

        cli.stop_all_motors()
        logger.warning("ArUco dock: turn did not converge (timeout)")
        return False

    def _charging(self) -> bool:
        with self.ds.lock:
            return bool(self.ds.on_charger or self.ds.charging)
