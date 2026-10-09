"""ArucoDocking — dock on the ArUco charger marker (marker 0).

The sequence, mirroring the cube approach:

1. Sidestep to a **vantage** offset from the marker's normal and look back
   at it, so the marker's face is seen obliquely.
2. Dwell and read the marker's normal *there* — an oblique view is what
   makes the normal measurable at all (see
   :mod:`pycozmo_ng.navigation.marker_observation`).
3. Drive to the standoff on the normal, facing the marker *centre*.
4. Verify the robot is centred on the normal line, then turn 180° and
   reverse in until the charging contacts detect — the real stop.

Why step 1 exists
-----------------
Driving straight at the marker — the obvious approach — puts the camera
fronto-parallel to the marker's face, which is the worst-conditioned
geometry for recovering its normal: the horizontal component that defines
the normal's *bearing* collapses toward zero, so normalising it amplifies
pixel noise without bound (measured: ~43 deg of bearing error, ~150 mm of
lateral error at a 200 mm standoff, against a ~15 mm capture width).
Step 4's reverse line runs on that bearing, so the robot aimed off the
normal line by metres.

Averaging does not fix this — the error is a systematic bias of the
ill-conditioned solve, so more frames and a second symmetric vantage both
converge to the same offset. Only the viewing angle moves it. Hence one
sideways look, and the normal is then used for *aiming only*.

Where the normal is deliberately **not** used
--------------------------------------------
* Lateral centring is measured against the robot's own heading, not the
  normal: "the marker is 12 mm to my left" is well conditioned and directly
  actionable, where re-deriving a normal from the same bad rotation is not.
* The reverse heading is ``heading + pi`` — the robot is already facing the
  marker centre, so turning about its own axis needs no landmark estimate.

What remains open-loop is the reverse itself: once the robot turns away the
marker is out of view, so the line it reverses along is fixed by the pose it
turned from. That is why step 4 is verified rather than trusted.

On success ``on_docked`` runs — the caller resets the pose origin (the
docked robot pose becomes the world origin, 65 mm in front of the
marker) and snaps the marker to its known docked position.
"""

from __future__ import annotations

import math
import time

from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.marker_observation import (
    MIN_USABLE_OBLIQUITY_DEG,
    TARGET_OBLIQUITY_DEG,
    NormalEstimate,
    estimate_normal,
    vantage_target,
)
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
            detect, or it has overshot the docked distance — reset
            origin, snap marker pose).
    """

    JOB_ARUCO_DOCK: str = "aruco_dock"

    APPROACH_DIST_MM: float = 200.0  # standoff in front of the marker

    #: Distance held from the marker while measuring its normal. Long
    #: enough that the marker clears the robot's own body when it sidesteps
    #: (and short enough that the position estimate stays sub-2 mm).
    VANTAGE_DIST_MM: float = 200.0
    #: Dwell at the vantage while the marker is seen obliquely.
    #: Mirrors ``CubeApproach.STANDOFF_DWELL_S``: the world-map EMA (alpha
    #: 0.15/frame) needs dozens of frames to absorb the pose jump from the
    #: sidestep, and correcting off a still-moving pose defeats the point.
    VANTAGE_DWELL_S: float = 3.0
    #: Extra window, after the settle, over which the fused marker state is
    #: sampled for the median. Runs once per vantage attempt.
    SAMPLE_WINDOW_S: float = 0.5
    #: Re-vantage this many times if the view comes back too
    #: fronto-parallel to aim from (see ``marker_observation``). Each retry
    #: sidesteps further, so a couple of attempts cover a robot that
    #: overshot the requested offset.
    VANTAGE_ATTEMPTS: int = 2
    #: Extra obliquity per retry, so attempt 2 looks from ~45 deg — the
    #: regime where the bearing error drops under 1 deg.
    VANTAGE_ESCALATION_DEG: float = 15.0

    #: Dwell at the standoff until the fused marker pose converges.
    STANDOFF_DWELL_S: float = 3.0
    #: Convergence criterion: the fused position must move less than
    #: ``STABLE_MM`` between reads for ``STABLE_WINDOW_S``.
    STABLE_MM: float = 2.0
    STABLE_WINDOW_S: float = 0.3

    #: Below this, a move is a pure turn — never send a zero-length path
    #: segment: the firmware asserts ``y != 0 || x != 0`` in
    #: ``trig_fast.cpp`` (``atan2(0, 0)``) and crashes.
    MIN_SEG_MM: float = 15.0

    TURN_SPEED_MMPS: float = 60.0
    #: Tighter than the cube's 2 deg arrival angle because the reverse is
    #: open loop: a residual heading error ``e`` drags the line sideways
    #: by ~135 · sin(e) mm over the reverse (2 deg ≈ 5 mm, 4 deg ≈ 9 mm).
    TURN_TOL_RAD: float = math.radians(2.0)
    TURN_TIMEOUT_S: float = 5.0

    #: Post-``go_to_pose`` arrival wait — mirrors ``CubeApproach``'s
    #: ``ARRIVE_*`` pair. After ``go_to_pose`` returns (its proximity
    #: poll can fire at 15 mm / 0.15 rad while the path is still driving,
    #: or when the firmware died short with the event omitted — COZMO-53),
    #: wait for the robot to land within these tolerances before taking
    #: the wheels over with ``_turn_to``. Short bound: when it fails, the
    #: residual is logged and the alignment check downstream recovers.
    ARRIVE_POS_MM: float = 8.0
    ARRIVE_ANG_RAD: float = math.radians(2.0)
    ARRIVE_TIMEOUT_S: float = 5.0

    #: Final centring tolerance before the 180 deg turn — measured against
    #: the robot's own heading (see the module docstring), so this is a
    #: genuine millimetre quantity rather than a normal-derived fudge.
    #: Equal to ``MIN_SEG_MM``: a smaller correction could not be sent as a
    #: path segment anyway.
    LATERAL_TOL_MM: float = 15.0
    ALIGN_RETRIES: int = 3
    #: Short settle before each centring measurement so the line being
    #: corrected against is the converged one, not a moving EMA.
    ALIGN_DWELL_S: float = 1.0

    REVERSE_SPEED_MMPS: float = -60.0  # negative = backward
    REVERSE_TIMEOUT_S: float = 15.0

    #: After a distance-based stop, give the contacts this long to
    #: register before declaring the dock complete.
    CHARGE_GRACE_S: float = 0.5

    #: Marker 0 is this far behind the camera when the robot is docked —
    #: the offset the world origin and the post-dock marker snap use.
    #: The reverse deliberately drives past it (``REVERSE_OVERSHOOT_MM``)
    #: so the robot actually bumps the charger.
    DOCKED_MARKER_DIST_MM: float = 65.0
    #: Deliberate overshoot: the geometric stop fires this far *past* the
    #: docked geometry, so the robot drives too much and bumps the back
    #: of the charger instead of stopping short when the fused pose runs
    #: optimistic. Charging contact remains the real stop (checked first
    #: every tick); ``DOCKED_MARKER_DIST_MM`` still describes where the
    #: robot rests when docked — the pose snap keeps using it.
    REVERSE_OVERSHOOT_MM: float = 50.0

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
    # 1-3. Measure the normal obliquely, drive the standoff, re-aim
    # ------------------------------------------------------------------

    def _approach(self, job: MotionToken, marker_id: int) -> bool:
        """Sidestep, read the normal obliquely, then drive the standoff.

        The vantage exists purely to make the normal measurable — see the
        module docstring. Everything after it is ordinary closed-loop
        driving.
        """
        estimate = self._measure_normal(job, marker_id)
        if estimate is None:
            logger.warning("ArUco dock: no usable view of marker %d", marker_id)
            return False
        if not estimate.usable:
            # Degenerate view: aiming a reverse from this would put the
            # robot far off the marker's normal line. Refuse rather than
            # dock on a bad estimate — the caller can re-run from a new
            # robot pose, which changes the geometry anyway.
            logger.warning(
                "ArUco dock: marker %d view too head-on (%.0f deg obliquity, need %.0f) — not docking",
                marker_id,
                estimate.obliquity_deg,
                MIN_USABLE_OBLIQUITY_DEG,
            )
            return False

        standoff_x, standoff_y = estimate.at_range(self.APPROACH_DIST_MM)
        heading = self._bearing_to_marker(estimate)
        logger.info(
            "ArUco dock: standoff (%.0f, %.0f) from marker %d normal — %s",
            standoff_x,
            standoff_y,
            marker_id,
            estimate.describe(),
        )
        if not self._go(job, standoff_x, standoff_y, heading):
            return False

        # The reverse below is open loop — commit only once the robot's rear
        # axis provably points at the marker.
        return self._align_to_line(job, marker_id)

    def _measure_normal(self, job: MotionToken, marker_id: int) -> NormalEstimate | None:
        """Sidestep to an oblique vantage and read the marker's normal there.

        Retries from a progressively wider angle when the view comes back
        too fronto-parallel to aim from (the robot may overshoot the
        requested offset, or the marker may sit where one sidestep does not
        clear the line).

        Returns None when the marker was never seen or the job was
        cancelled.
        """
        for attempt in range(self.VANTAGE_ATTEMPTS + 1):
            state = self._get_marker_state(marker_id)
            if state is None:
                logger.warning("ArUco dock: marker %d not in world map", marker_id)
                return None
            robot = self.cli.pose
            seed = estimate_normal([state], (robot.position.x, robot.position.y))
            if seed is None:
                logger.warning("ArUco dock: marker %d has no usable normal to plan a vantage", marker_id)
                return None

            obliquity = TARGET_OBLIQUITY_DEG + attempt * self.VANTAGE_ESCALATION_DEG
            vx, vy, vheading = vantage_target(
                seed,
                (robot.position.x, robot.position.y),
                self.VANTAGE_DIST_MM,
                obliquity,
            )
            if attempt == 0:
                logger.info(
                    "ArUco dock: viewing marker %d from (%.0f, %.0f) — %.0f deg off its normal",
                    marker_id,
                    vx,
                    vy,
                    obliquity,
                )
            if not self._go(job, vx, vy, vheading):
                return None
            if not self._settle_marker(marker_id, job, self.VANTAGE_DWELL_S):
                return None

            estimate = estimate_normal(self._collect_samples(marker_id), self._robot_xy())
            if estimate is None:
                logger.warning("ArUco dock: marker %d lost at the vantage", marker_id)
                return None
            if estimate.usable or attempt == self.VANTAGE_ATTEMPTS:
                logger.info("ArUco dock: marker %d measured — %s", marker_id, estimate.describe())
                return estimate
            logger.info(
                "ArUco dock: view still head-on (%.0f deg) — retrying from a wider angle",
                estimate.obliquity_deg,
            )
        return None

    def _collect_samples(self, marker_id: int) -> list:
        """Snapshot the fused marker state over one settle window.

        The world map already smooths each update; reading the state once
        more per poll tick widens the sample set so
        :func:`~pycozmo_ng.navigation.marker_observation.estimate_normal`
        can take a median and drop an outlier a single frame would
        otherwise dominate. This screens outliers — it does not reduce the
        systematic error of a head-on view (see that module's docstring).
        """
        samples = []
        deadline = time.monotonic() + self.SAMPLE_WINDOW_S
        while time.monotonic() < deadline:
            state = self._get_marker_state(marker_id)
            if state is not None:
                samples.append(state)
            time.sleep(0.02)
        return samples

    def _robot_xy(self) -> tuple[float, float]:
        """The robot's XY position (mm, world frame)."""
        pose = self.cli.pose
        return pose.position.x, pose.position.y

    def _bearing_to_marker(self, estimate: NormalEstimate) -> float:
        """Heading from the robot's current pose to the marker centre.

        Derived from the marker's *position*, not its normal — position is
        measured ~20x better than rotation, and facing the centre is what
        makes the standoff reachable without trusting the noisy angle.
        """
        robot = self.cli.pose
        return math.atan2(estimate.y - robot.position.y, estimate.x - robot.position.x)

    def _settle_marker(self, marker_id: int, job: MotionToken, dwell_s: float | None = None) -> bool:
        """Dwell until the fused marker pose converges (``dwell_s`` deadline).

        Mirrors ``CubeApproach._dwell_for_smoothing``: the world-map EMA
        (alpha 0.15/frame) needs dozens of frames to absorb the pose jump
        from the approach drive, so a fixed short sleep corrects off a
        still-moving pose. Ends early once the fused position stops
        moving *and* the marker has been re-detected during the dwell.

        Returns False when cancelled or when the marker was not
        re-detected (the caller must not dock on a stale pose).
        """
        if dwell_s is None:
            dwell_s = self.STANDOFF_DWELL_S
        deadline = time.monotonic() + dwell_s
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
            logger.warning("ArUco dock: marker %d not re-detected during settle", marker_id)
            return False
        return True

    def _align_to_line(self, job: MotionToken, marker_id: int) -> bool:
        """Verify (and correct) that the robot's rear axis points at the marker.

        The reverse phase drives straight with no feedback, so whatever
        misalignment exists here is the misalignment it docks with.

        The check is the perpendicular distance from the marker to the
        robot's own centreline — how far off its centreline the marker
        sits. That is measured against the **robot's heading**, not the
        marker's normal: the marker position is ~20x better measured than
        the marker's rotation, so this inherits none of the normal's error.
        Re-deriving a normal here would reintroduce exactly the
        ill-conditioned estimate the vantage step exists to avoid.

        The correction is a *turn* onto the bearing to the marker: aiming
        the robot at the marker is what zeroes the offset (sliding
        sideways would only move the offset around). Turning holds the
        standoff distance exactly, so the reverse distance stays known.

        Returns False when cancelled, when the marker is unknown, or
        when the robot could not be centred within ``ALIGN_RETRIES``
        corrections (the caller must not reverse blind).
        """
        lateral = float("inf")
        for attempt in range(self.ALIGN_RETRIES + 1):
            # Settle first: the correction target must come off the
            # converged line, not a moving EMA (live log showed the line
            # drifting ~17 mm between the approach and the measurement).
            if not self._settle_marker(marker_id, job, self.ALIGN_DWELL_S):
                return False
            state = self._get_marker_state(marker_id)
            if state is None:
                return False
            rp = self.cli.pose
            yaw = rp.rotation.angle_z.radians
            dx, dy = float(state.position[0]) - rp.position.x, float(state.position[1]) - rp.position.y
            along = dx * math.cos(yaw) + dy * math.sin(yaw)  # range along the centreline
            if along <= 0.0:
                logger.warning("ArUco dock: robot is behind the marker — cannot centre")
                return False
            # Signed perpendicular offset: how far the marker sits off the
            # robot's centreline, in millimetres.
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if abs(lateral) <= self.LATERAL_TOL_MM:
                return True
            if attempt == self.ALIGN_RETRIES:
                break
            logger.info(
                "ArUco dock: %.0f mm off the marker line — re-aiming (%d/%d)",
                abs(lateral),
                attempt + 1,
                self.ALIGN_RETRIES,
            )
            # Turn onto the bearing to the marker centre. This is the
            # correction that actually zeroes the offset, it holds the
            # standoff exactly, and it needs no normal to express.
            if not self._go(job, rp.position.x, rp.position.y, math.atan2(dy, dx)):
                return False
        logger.warning("ArUco dock: %.0f mm off the marker line — not reversing", abs(lateral))
        return False

    # ------------------------------------------------------------------
    # 3. Turn around and reverse into the marker
    # ------------------------------------------------------------------

    def _reverse_into_charger(self, job: MotionToken, marker_id: int) -> bool:
        """Turn about the robot's own axis and reverse until contact.

        The robot is already facing the marker centre, so the reverse
        heading is just ``yaw + pi``: turning away needs no landmark
        estimate at all, and it keeps the reverse line on the line the
        centring step verified.

        Two stop conditions: the charging contacts (checked first — the
        real stop) or the robot having driven past the docked distance
        by ``REVERSE_OVERSHOOT_MM`` — a deliberate overshoot so it bumps
        the back of the charger instead of stopping short, while still
        bounding how long the wheels can push with a lagging flag.
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        mx, my = float(state.position[0]), float(state.position[1])

        # Turn in place: rear toward the charger.
        pose = self.cli.pose
        if not self._go(job, pose.position.x, pose.position.y, wrap_pi(pose.rotation.angle_z.radians + math.pi)):
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
            if self._at_dock(mx, my):
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
            # ``go_to_pose`` can return while the path is still driving
            # (15 mm / 0.15 rad proximity poll) — let it land before
            # taking the wheels over with ``_turn_to``.
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
        rp = self.cli.pose
        dist_err = math.hypot(x - rp.position.x, y - rp.position.y)
        ang_err = abs(wrap_pi(heading - rp.rotation.angle_z.radians))
        logger.warning(
            "ArUco dock: target not reached within %.0f s — %.0f mm / %.1f deg off; continuing",
            self.ARRIVE_TIMEOUT_S,
            dist_err,
            math.degrees(ang_err),
        )
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

    def _at_dock(self, mx: float, my: float) -> bool:
        """True once the robot has deliberately overshot the docked distance.

        The stop fires when the robot is within
        ``DOCKED_MARKER_DIST_MM - REVERSE_OVERSHOOT_MM`` of the marker — 5 cm
        past the resting geometry — so the robot bumps the back of the
        charger instead of stopping short when the fused pose runs
        optimistic. A geometric backstop for the lagging charging flag,
        which remains the real stop.

        A plain range check: the marker position is measured ~20x better
        than its rotation, and by this point the robot is on the normal
        line, so range and "distance along the normal" agree.
        """
        rp = self.cli.pose
        return math.hypot(rp.position.x - mx, rp.position.y - my) <= (
            self.DOCKED_MARKER_DIST_MM - self.REVERSE_OVERSHOOT_MM
        )

    def _wait_charging(self) -> bool:
        """After a distance stop, give the contacts time to register.

        Returns the real charging state. The geometric backstop only says
        the robot *reached* the docked distance — a dock that bumped the
        charger off-centre gets there too, and reporting that as success
        would let ``on_docked`` re-anchor the world origin to a pose the
        robot is not in.
        """
        deadline = time.monotonic() + self.CHARGE_GRACE_S
        while time.monotonic() < deadline and not self._charging():
            time.sleep(0.05)
        if not self._charging():
            logger.warning("ArUco dock: reached the docked distance but the contacts did not register — not docked")
            return False
        return True

    def _charging(self) -> bool:
        with self.ds.lock:
            return bool(self.ds.on_charger or self.ds.charging)
