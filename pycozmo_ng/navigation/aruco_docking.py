"""ArucoDocking — dock on the ArUco charger marker (marker 0).

The sequence, mirroring the cube approach:

1. Drive to a point in front of the marker, facing it — the camera sees
   the marker and the world map refreshes its pose.
2. Dwell until the fused marker pose has actually converged, then
   correct the approach point from it.
3. Verify the robot is centred on the marker's normal line — the
   reverse is open loop, so any lateral offset here is the offset it
   docks with.
4. Turn 180° so the rear is toward the charger and reverse in until the
   charging contacts detect or the robot has backed to the docked
   distance (with a timeout).

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
            Optional callback run once the robot is docked (contacts
            detect, or it has backed to the docked distance — reset
            origin, snap marker pose).
    """

    JOB_ARUCO_DOCK: str = "aruco_dock"

    APPROACH_DIST_MM: float = 200.0  # standoff in front of the marker

    #: Dwell at the standoff until the fused marker pose converges —
    #: mirrors ``CubeApproach.STANDOFF_DWELL_S``. The world-map EMA
    #: (alpha 0.15/frame) barely moves on the first frames after the
    #: approach drive; correcting off a still-moving pose leaves the
    #: reverse line off-centre (the intermittent ~3 cm dock error).
    STANDOFF_DWELL_S: float = 1.5
    #: The marker must have been re-detected this recently during the
    #: dwell (``ArucoState.updated_at`` advances per fused frame) — a
    #: stale pose is not a converged pose.
    FRESH_TOL_S: float = 1.0
    #: Convergence criterion: the fused position must move less than
    #: ``STABLE_MM`` between reads for ``STABLE_WINDOW_S``.
    STABLE_MM: float = 2.0
    STABLE_WINDOW_S: float = 0.3

    #: Below this, a move is a pure turn — never send a zero-length path
    #: segment: the firmware asserts ``y != 0 || x != 0`` in
    #: ``trig_fast.cpp`` (``atan2(0, 0)``) and crashes.
    MIN_SEG_MM: float = 15.0

    TURN_SPEED_MMPS: float = 60.0
    #: Tighter than the cube's 2° arrival angle because the reverse is
    #: open loop: a residual heading error ``e`` drags the line sideways
    #: by ~135 · sin(e) mm over the reverse (2° ≈ 5 mm, 4° ≈ 9 mm).
    TURN_TOL_RAD: float = math.radians(2.0)
    TURN_TIMEOUT_S: float = 5.0

    #: Post-``go_to_pose`` arrival wait — mirrors ``CubeApproach``'s
    #: ``ARRIVE_*`` pair. ``go_to_pose`` returns on its pose-proximity
    #: poll (15 mm / 0.15 rad) while the firmware path is still
    #: executing; starting the ``drive_wheels`` turn right away cuts the
    #: path short with a random stop error.
    ARRIVE_POS_MM: float = 8.0
    ARRIVE_ANG_RAD: float = math.radians(2.0)
    ARRIVE_TIMEOUT_S: float = 10.0

    #: Final centring tolerance before the 180° turn — mirrors
    #: ``CubeApproach._verify_grab_alignment``. Equal to ``MIN_SEG_MM``:
    #: a smaller correction could not be sent as a path segment anyway.
    LATERAL_TOL_MM: float = 15.0
    ALIGN_RETRIES: int = 2

    REVERSE_SPEED_MMPS: float = -60.0  # negative = backward
    REVERSE_TIMEOUT_S: float = 15.0

    #: After a distance-based stop, give the contacts this long to
    #: register before declaring the dock complete.
    CHARGE_GRACE_S: float = 0.5

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

        # Dwell until the fused marker pose converges — a fixed short
        # sleep corrects off a pose that is still moving.
        if not self._settle_marker(marker_id, job):
            return False

        # If the refreshed pose moved the approach point, correct it
        # (``_go`` is a no-op drive + turn when already in place).
        corrected = self._approach_target(marker_id)
        if corrected is None:
            return False
        cx, cy, cheading = corrected
        if math.hypot(cx - x, cy - y) > self.MIN_SEG_MM:
            logger.info("ArUco dock: marker pose updated — correcting to (%.0f, %.0f)", cx, cy)
        if not self._go(job, cx, cy, cheading):
            return False
        # The reverse below is open loop — commit only when the robot is
        # centred on the marker's normal line (the cube's verify step).
        return self._align_to_line(job, marker_id)

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

    def _settle_marker(self, marker_id: int, job: MotionToken) -> bool:
        """Dwell at the standoff until the fused marker pose converges.

        Mirrors ``CubeApproach._dwell_for_smoothing``: the world-map EMA
        (alpha 0.15/frame) needs dozens of frames to absorb the pose jump
        from the approach drive, so a fixed short sleep corrects off a
        still-moving pose. Ends early once the fused position stops
        moving *and* the marker has been re-detected during the dwell.

        Returns False when cancelled or when the marker was not
        re-detected (the caller must not dock on a stale pose).
        """
        deadline = time.monotonic() + self.STANDOFF_DWELL_S
        base_ts: float | None = None
        saw_fresh = False
        stable_since: float | None = None
        last: tuple[float, float] | None = None
        while time.monotonic() < deadline:
            if job.cancelled:
                self.cli.stop_all_motors()
                return False
            state = self._get_marker_state(marker_id)
            if state is None:
                time.sleep(0.05)
                continue
            if base_ts is None:
                base_ts = state.updated_at
            if state.updated_at > base_ts + 0.05:
                saw_fresh = True
            pos = (float(state.position[0]), float(state.position[1]))
            if last is not None:
                if math.hypot(pos[0] - last[0], pos[1] - last[1]) >= self.STABLE_MM:
                    stable_since = None
                elif stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= self.STABLE_WINDOW_S and saw_fresh:
                    return True
            last = pos
            time.sleep(0.05)
        if not saw_fresh:
            logger.warning("ArUco dock: marker %d not re-detected at the standoff", marker_id)
            return False
        return True

    def _align_to_line(self, job: MotionToken, marker_id: int) -> bool:
        """Verify (and correct) that the robot sits on the marker's normal line.

        The reverse phase drives straight with no feedback, so the
        lateral offset measured here is the offset the robot docks with
        — mirroring ``CubeApproach._verify_grab_alignment``, which reads
        a fresh fused pose and re-drives when the target line moved.

        Returns False when cancelled, when the marker is unknown, or
        when the robot could not be centred within ``ALIGN_RETRIES``
        corrections (the caller must not reverse blind).
        """
        lateral = float("inf")
        for attempt in range(self.ALIGN_RETRIES + 1):
            state = self._get_marker_state(marker_id)
            if state is None:
                return False
            nx, ny = self._normal(state)
            if nx is None:
                return False
            mx, my = float(state.position[0]), float(state.position[1])
            rp = self.cli.pose
            dx, dy = rp.position.x - mx, rp.position.y - my
            along = dx * nx + dy * ny  # distance in front of the marker
            lateral = abs(dx * -ny + dy * nx)  # offset perpendicular to the line
            if lateral <= self.LATERAL_TOL_MM:
                return True
            if attempt == self.ALIGN_RETRIES:
                break
            if along <= 0.0:
                logger.warning("ArUco dock: robot is behind the marker — cannot centre")
                return False
            logger.info(
                "ArUco dock: %.0f mm off the marker line — centring (%d/%d)",
                lateral,
                attempt + 1,
                self.ALIGN_RETRIES,
            )
            # Drive to the point on the normal line at the same standoff.
            if not self._go(job, mx + nx * along, my + ny * along, math.atan2(-ny, -nx)):
                return False
        logger.warning("ArUco dock: %.0f mm off the marker line — not reversing", lateral)
        return False

    # ------------------------------------------------------------------
    # 3. Turn around and reverse into the marker
    # ------------------------------------------------------------------

    def _reverse_into_charger(self, job: MotionToken, marker_id: int) -> bool:
        """Turn 180° and reverse until contact or the docked distance.

        Two stop conditions: the charging contacts (checked first) or the
        robot having backed to ``DOCKED_MARKER_DIST_MM`` along the marker
        normal — so the wheels never keep driving into the charger while
        waiting for the charging flag to land.
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        nx, ny = self._normal(state)
        if nx is None:
            return False
        mx, my = float(state.position[0]), float(state.position[1])

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
            if self._at_dock(mx, my, nx, ny):
                self.cli.stop_all_motors()
                return self._wait_charging()
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
            # ``go_to_pose`` returns on its 15 mm / 0.15 rad proximity
            # poll while the firmware path may still be executing — let
            # it finish before taking the wheels over with ``_turn_to``.
            if not self._wait_arrival(job, x, y, heading):
                return False
        return self._turn_to(job, heading)

    def _wait_arrival(self, job: MotionToken, x: float, y: float, heading: float) -> bool:
        """Wait until the robot actually reaches (*x*, *y*, *heading*).

        Mirrors ``CubeApproach``'s post-``go_to_pose`` arrival wait: the
        firmware path keeps driving after the proximity poll returned,
        and interrupting it with wheel commands stops the robot short
        with a random error.

        Returns False when cancelled (motors already stopped); True on
        arrival or after ``ARRIVE_TIMEOUT_S`` (the alignment check runs
        downstream either way).
        """
        deadline = time.monotonic() + self.ARRIVE_TIMEOUT_S
        while time.monotonic() < deadline:
            if job.cancelled:
                logger.info("ArUco dock cancelled")
                self.cli.stop_all_motors()
                return False
            rp = self.cli.pose
            if (
                math.hypot(x - rp.position.x, y - rp.position.y) <= self.ARRIVE_POS_MM
                and abs(wrap_pi(heading - rp.rotation.angle_z.radians)) <= self.ARRIVE_ANG_RAD
            ):
                return True
            time.sleep(0.05)
        logger.warning("ArUco dock: target not reached within %.0f s — continuing", self.ARRIVE_TIMEOUT_S)
        return True

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

    def _at_dock(self, mx: float, my: float, nx: float, ny: float) -> bool:
        """True once the robot has backed to the docked distance.

        Distance is measured along the marker normal (the reverse
        direction): the marker sits ``DOCKED_MARKER_DIST_MM`` behind the
        robot when docked. Acts as a geometric backstop so the robot
        stops there instead of driving on while the charging flag lags.
        """
        rp = self.cli.pose
        along = (rp.position.x - mx) * nx + (rp.position.y - my) * ny
        return along <= self.DOCKED_MARKER_DIST_MM

    def _wait_charging(self) -> bool:
        """After a distance stop, give the contacts time to register.

        Always returns True: the robot is at the docked geometry with a
        verified line-up, so the dock counts even if the charging flag
        lands a moment later (Brain polls it for the HUD anyway).
        """
        deadline = time.monotonic() + self.CHARGE_GRACE_S
        while time.monotonic() < deadline and not self._charging():
            time.sleep(0.05)
        if not self._charging():
            logger.warning("ArUco dock: at docked distance — charging not detected yet")
        return True

    def _charging(self) -> bool:
        with self.ds.lock:
            return bool(self.ds.on_charger or self.ds.charging)
