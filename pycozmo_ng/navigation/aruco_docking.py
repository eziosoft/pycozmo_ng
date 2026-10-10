"""ArucoDocking — dock on the ArUco charger marker (marker 0).

The sequence, mirroring the cube approach:

1. Drive to a **standoff** on the line of sight to the marker — the ray
   from the marker through the robot, held at ``APPROACH_DIST_MM`` — so
   the camera keeps the marker in view the whole way in.
2. Re-aim onto the marker and settle until the fused pose converges.
3. Verify the robot is centred on the marker, then turn 180° and reverse
   in until the charging contacts detect — the real stop.

Why there is no sideways vantage
--------------------------------
An earlier revision sidestepped ~20° off the marker's normal to measure
that normal from an oblique view, because a planar marker viewed head-on
yields a normal whose *bearing* is meaningless (~43° of error at 2° of
obliquity, which a 250 mm standoff turns into hundreds of millimetres of
lateral error at the charger — the reason the vantage existed at all).

That cannot work on this dock. **The marker sits on a post at the back
end of the charging station, inside its rim.** From the side it is
heavily foreshortened and rides the edge of the frame, where the
detector drops any quad that touches the image border. Live failure:

    ArUco dock: view still head-on (12 deg) — retrying from a wider angle
    ArUco dock: marker 0 not re-detected during settle — robot (162, -129)
    ArUco dock: no usable view of marker 0

The retry had pushed the robot 32° off the normal, where the marker sat
at the frame edge and no detection landed for a whole dwell — so the
dock aborted before it ever tried to reverse. Measured afterwards on the
robot: marker 0 is detected at every angle from −40° to +30° **as long
as the robot is pointed at it**; the vantage lost it by pointing the
camera from a stale plan instead of at the marker.

So the normal is not used anywhere, and nothing about the dock depends
on the marker being seen from the side.

Where the accuracy actually comes from
--------------------------------------
The standoff is placed on the **line of sight**, which is computed from
two *positions* — the fused marker and the robot pose — measured ~20×
better than the marker's rotation. Facing the marker from it is exact by
construction, so the camera is on the marker at every settle and a
settle can never fail for lack of a detection.

The reverse line is fixed by the robot's own heading (``yaw + π``) and
the centring check is measured against that heading — "the marker is
27 mm off my centreline" is well conditioned and directly actionable.
Landing the rear on the marker's centre is what matters, because the
marker is the dock's centre; the reverse needs no landmark estimate
beyond that.

The one quantity the line of sight cannot give is how far the robot is
turned away from the dock's own axis: the reverse inherits whatever
obliquity the robot started with, because centring only ever points the
robot *at* the marker — never *along* the dock. There is deliberately no
guard on that angle. The obvious signal would be the marker's fused
rotation, but that is the ill-conditioned normal again: at a head-on
start it is off by tens of degrees (the same ~43 deg the vantage existed
to avoid), so gating on it would refuse the poses that dock best. In
practice the angle is bounded anyway — a flat marker at the end of the
dock is not detectable from far off its own axis, so if the dock could
be planned at all the robot was already facing roughly the right way.

What remains open-loop is the reverse itself: once the robot turns away
the marker is out of view, so the line it reverses along is fixed by the
pose it turned from. That is why the centring step is verified rather
than trusted.

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
            detect, or it has overshot the docked distance — reset
            origin, snap marker pose).
    """

    JOB_ARUCO_DOCK: str = "aruco_dock"

    #: Standoff in front of the marker, held on the line of sight. 250 mm
    #: — far enough that the reverse covers the distance in a straight
    #: line, near enough that the marker is large and unambiguous in the
    #: frame (~20 px wide at QVGA).
    APPROACH_DIST_MM: float = 250.0
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
    #: Tightened from 15 mm to 10 mm after live testing: the old
    #: tolerance let the robot pass without correction at offsets
    #: that matched the charger capture width. 10 mm forces the
    #: heading correction for any material offset while still
    #: accommodating the 2° turn-tolerance residual (at 120 mm
    #: standoff, 2° → 4.2 mm).
    LATERAL_TOL_MM: float = 10.0
    ALIGN_RETRIES: int = 3
    #: Short settle before each centring measurement so the line being
    #: corrected against is the converged one, not a moving EMA.
    ALIGN_DWELL_S: float = 1.0

    #: Heading sweep for a marker that is out of view at the standoff.
    #: The fused marker position carries tens of millimetres of error,
    #: which at standoff range is ~10 deg of bearing — enough to park the
    #: marker outside the camera's view (live log: "10 degrees more to
    #: the left and it will see it"). Steps of ``SEARCH_STEP_DEG`` out to
    #: ``SEARCH_MAX_OFFSET_DEG`` cover it with margin for the 2 deg turn
    #: tolerance; ``SEARCH_DWELL_S`` per step is a few detection cycles
    #: at the camera's frame rate.
    SEARCH_STEP_DEG: float = 5.0
    SEARCH_MAX_OFFSET_DEG: float = 15.0
    SEARCH_DWELL_S: float = 0.4
    #: A detection this recent counts as "the marker is in view now" —
    #: comfortably above one camera frame, so a healthy detection stream
    #: never reads stale, while a marker that left the frame ages out
    #: after a couple of frames.
    DETECTION_FRESH_S: float = 0.25

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
            logger.info("ArUco dock: docked — %s", self._scene(marker_id))
            if self._on_docked is not None:
                self._on_docked()
        except Exception:  # noqa: BLE001
            logger.exception("ArUco dock error")

    # ------------------------------------------------------------------
    # 1-2. Drive the line of sight, settle, re-aim
    # ------------------------------------------------------------------

    def _approach(self, job: MotionToken, marker_id: int) -> bool:
        """Drive the standoff on the line of sight, then centre on the marker.

        The standoff is computed from positions alone (see the module
        docstring), so the marker stays in view on the way in and no
        measurement of the marker's rotation is ever needed.
        """
        logger.info("ArUco dock: start — %s", self._scene(marker_id))
        standoff_x, standoff_y, heading = self._standoff_target(marker_id)
        if standoff_x is None:
            logger.warning("ArUco dock: no usable view of marker %d — %s", marker_id, self._scene(marker_id))
            return False
        logger.info(
            "ArUco dock: standoff (%.0f, %.0f) from marker %d, heading %.1f deg — %s",
            standoff_x,
            standoff_y,
            marker_id,
            math.degrees(heading),
            self._scene(marker_id),
        )
        if not self._go(job, standoff_x, standoff_y, heading):
            return False
        logger.info("ArUco dock: at standoff — %s", self._scene(marker_id))

        # Let the world-map EMA absorb the drive before anything measures
        # off it: the alignment below reads the fused pose, and a settle
        # that still has the marker moving corrects against a line the
        # robot is not actually on.
        if not self._settle_or_search(marker_id, job, self.STANDOFF_DWELL_S):
            return False

        # The reverse below is open loop — commit only once the robot's rear
        # axis provably points at the marker.
        return self._align_to_line(job, marker_id)

    def _standoff_target(self, marker_id: int) -> tuple[float | None, float, float]:
        """Plan the standoff pose from the line of sight to the marker.

        The standoff sits on the ray from the marker through the robot, so
        the drive keeps the camera pointed at the marker and the final
        heading is the bearing back to it. Both come from *positions*,
        which are measured ~20× better than the marker's rotation — there
        is no ill-conditioned normal anywhere in the estimate.

        Returns:
            ``(x, y, heading)``, or ``(None, 0, 0)`` when the marker is
            unknown or the robot is on top of it.
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            logger.warning("ArUco dock: marker %d not in world map — %s", marker_id, self._scene(marker_id))
            return None, 0.0, 0.0
        mx, my = float(state.position[0]), float(state.position[1])
        rx, ry = self._robot_xy()
        dx, dy = rx - mx, ry - my
        rng = math.hypot(dx, dy)
        if rng < 1e-6:
            logger.warning("ArUco dock: robot is on marker %d — %s", marker_id, self._scene(marker_id))
            return None, 0.0, 0.0
        ux, uy = dx / rng, dy / rng
        standoff_x = mx + ux * self.APPROACH_DIST_MM
        standoff_y = my + uy * self.APPROACH_DIST_MM
        return standoff_x, standoff_y, math.atan2(my - standoff_y, mx - standoff_x)

    def _robot_xy(self) -> tuple[float, float]:
        """The robot's XY position (mm, world frame)."""
        pose = self.cli.pose
        return pose.position.x, pose.position.y

    @staticmethod
    def _bearing_to(rp, mx: float, my: float) -> float:
        """Bearing from the robot pose *rp* to a world-frame point.

        Derived from *positions* — they are measured ~20x better than the
        marker's rotation, so facing the centre is what makes the standoff
        reachable without trusting the noisy angle.
        """
        return math.atan2(my - rp.position.y, mx - rp.position.x)

    def _scene(self, marker_id: int) -> str:
        """One-line geometry snapshot: robot pose, marker pose, and the relation.

        Every dock decision — aim, settle, sweep, reverse — is a statement
        about where the robot and the marker are, so each milestone and
        each failure appends this and the log alone reconstructs the
        attempt (world frame, mm; yaw/bearing in degrees). ``aim`` is
        bearing minus yaw: how far the camera points off the marker, with
        positive meaning the marker sits left of the heading.
        """
        rp = self.cli.pose
        yaw = rp.rotation.angle_z.radians
        head = f"robot ({rp.position.x:.0f}, {rp.position.y:.0f}, {math.degrees(yaw):.1f} deg)"
        state = self._get_marker_state(marker_id)
        if state is None:
            return f"{head}; marker {marker_id} not in world map"
        mx, my = float(state.position[0]), float(state.position[1])
        dx, dy = mx - rp.position.x, my - rp.position.y
        bearing = math.atan2(dy, dx)
        return (
            f"{head}; marker {marker_id} ({mx:.0f}, {my:.0f}); range {math.hypot(dx, dy):.0f} mm; "
            f"bearing {math.degrees(bearing):.1f} deg; aim {math.degrees(wrap_pi(bearing - yaw)):+.1f} deg"
        )

    def _settle_marker(self, marker_id: int, job: MotionToken, dwell_s: float) -> bool:
        """Dwell until the fused marker pose converges (``dwell_s`` deadline).

        Mirrors ``CubeApproach._dwell_for_smoothing``: the world-map EMA
        (alpha 0.15/frame) needs dozens of frames to absorb the pose jump
        from the approach drive, so a fixed short sleep corrects off a
        still-moving pose. Ends early once the fused position stops
        moving *and* the marker has been re-detected during the dwell.

        Returns False when cancelled or when the marker was not
        re-detected (the caller must not dock on a stale pose).
        """
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
                    return self._settled(marker_id)
            last = pos
            time.sleep(0.05)
        if not saw_fresh:
            logger.warning(
                "ArUco dock: marker %d not re-detected during settle — %s", marker_id, self._scene(marker_id)
            )
            return False
        return self._settled(marker_id)

    def _settled(self, marker_id: int) -> bool:
        """Debug record of where the marker converged — always returns True."""
        state = self._get_marker_state(marker_id)
        if state is not None:
            logger.debug(
                "ArUco dock: marker %d settled at (%.0f, %.0f)",
                marker_id,
                float(state.position[0]),
                float(state.position[1]),
            )
        return True

    def _settle_or_search(self, marker_id: int, job: MotionToken, dwell_s: float) -> bool:
        """Settle — sweeping the heading for the marker when it is out of view.

        The settle needs fresh detections to converge on; when the camera
        is pointed away from the marker there are none, and aborting the
        whole dock over it throws away a pose that is one small turn from
        usable. So the heading sweeps out to ``SEARCH_MAX_OFFSET_DEG``
        either side of the computed bearing until a detection lands, then
        settles for real.

        Returns False when cancelled, when the marker is unknown, or when
        the sweep found nothing within the window.
        """
        if self._settle_marker(marker_id, job, dwell_s):
            return True
        if job.cancelled:
            return False
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        bearing = self._bearing_to(self.cli.pose, float(state.position[0]), float(state.position[1]))
        logger.info(
            "ArUco dock: marker %d out of view — sweeping ±%.0f deg about %.1f deg — %s",
            marker_id,
            self.SEARCH_MAX_OFFSET_DEG,
            math.degrees(bearing),
            self._scene(marker_id),
        )
        if not self._search_marker(job, marker_id, bearing):
            logger.warning(
                "ArUco dock: marker %d not found within ±%.0f deg — %s",
                marker_id,
                self.SEARCH_MAX_OFFSET_DEG,
                self._scene(marker_id),
            )
            return False
        return self._settle_marker(marker_id, job, dwell_s)

    def _search_marker(self, job: MotionToken, marker_id: int, bearing: float) -> bool:
        """Sweep the heading about *bearing* until a fresh detection lands.

        The heading at *bearing* itself was just covered by the failed
        settle, so the sweep starts at the first step off it, alternating
        sides — the observed miss was to one side, but the sign of the
        fused-position error is not known.

        Returns False when cancelled or when nothing was seen within
        ``SEARCH_MAX_OFFSET_DEG``.
        """
        step = math.radians(self.SEARCH_STEP_DEG)
        steps = round(self.SEARCH_MAX_OFFSET_DEG / self.SEARCH_STEP_DEG)
        for k in range(1, steps + 1):
            for offset in (k * step, -k * step):
                if not self._turn_to(job, wrap_pi(bearing + offset)):
                    return False
                if self._marker_seen(marker_id, job):
                    logger.info(
                        "ArUco dock: marker %d found %.0f deg off the computed bearing — %s",
                        marker_id,
                        math.degrees(offset),
                        self._scene(marker_id),
                    )
                    return True
        return False

    def _marker_seen(self, marker_id: int, job: MotionToken) -> bool:
        """True when a detection lands within ``SEARCH_DWELL_S``.

        Freshness is judged by age rather than by an advancing timestamp:
        the first read after the turn may already carry a detection that
        arrived during it, and requiring a *second* update would miss it.
        """
        deadline = time.monotonic() + self.SEARCH_DWELL_S
        while time.monotonic() < deadline:
            if job.cancelled:
                self.cli.stop_all_motors()
                return False
            state = self._get_marker_state(marker_id)
            if state is not None and time.monotonic() - float(state.updated_at) <= self.DETECTION_FRESH_S:
                return True
            time.sleep(0.05)
        return False

    def _align_to_line(self, job: MotionToken, marker_id: int) -> bool:
        """Verify (and correct) that the robot's rear axis points at the marker.

        The reverse phase drives straight with no feedback, so whatever
        misalignment exists here is the misalignment it docks with.

        The check is the perpendicular distance from the marker to the
        robot's own centreline — how far off its centreline the marker
        sits. That is measured against the **robot's heading**, not the
        marker's normal: the marker position is ~20x better measured than
        the marker's rotation, so this inherits none of a rotation's error.
        Re-deriving a normal here would reintroduce exactly the
        ill-conditioned estimate the line-of-sight approach exists to
        avoid.

        The correction is a *turn* onto the bearing to the marker: aiming
        the robot at the marker is what zeroes the offset (sliding
        sideways would only move the offset around). Turning holds the
        standoff distance exactly, so the reverse distance stays known.

        Two steps run before that check, because a settle needs a marker
        the camera can actually see:

        * A re-aim onto the *current* bearing — the arrival heading came
          off the pre-drive plan, and the fused marker position keeps
          moving while the path executes, so the camera can land pointed
          away from the marker.
        * A heading sweep when the settle still comes back with no fresh
          detection: the fused position carries tens of millimetres of
          error, which at standoff range is ~10 deg of bearing (live log:
          "10 degrees more to the left and it will see it"). The heading
          that finds the marker is also the correct aim — it is where the
          marker really is.

        Returns False when cancelled, when the marker is unknown, or
        when the robot could not be centred within ``ALIGN_RETRIES``
        corrections (the caller must not reverse blind).
        """
        state = self._get_marker_state(marker_id)
        if state is None:
            return False
        rp = self.cli.pose
        if not self._turn_to(job, self._bearing_to(rp, float(state.position[0]), float(state.position[1]))):
            return False
        lateral = float("inf")
        for attempt in range(self.ALIGN_RETRIES + 1):
            # Settle first: the correction target must come off the
            # converged line, not a moving EMA (live log showed the line
            # drifting ~17 mm between the approach and the measurement).
            if not self._settle_or_search(marker_id, job, self.ALIGN_DWELL_S):
                return False
            state = self._get_marker_state(marker_id)
            if state is None:
                return False
            rp = self.cli.pose
            yaw = rp.rotation.angle_z.radians
            dx, dy = float(state.position[0]) - rp.position.x, float(state.position[1]) - rp.position.y
            along = dx * math.cos(yaw) + dy * math.sin(yaw)  # range along the centreline
            if along <= 0.0:
                logger.warning("ArUco dock: robot is behind the marker — cannot centre — %s", self._scene(marker_id))
                return False
            # Signed perpendicular offset: how far the marker sits off the
            # robot's centreline, in millimetres.
            lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if abs(lateral) <= self.LATERAL_TOL_MM:
                return True
            if attempt == self.ALIGN_RETRIES:
                break
            logger.info(
                "ArUco dock: %.0f mm off the marker line — re-aiming (%d/%d) — %s",
                abs(lateral),
                attempt + 1,
                self.ALIGN_RETRIES,
                self._scene(marker_id),
            )
            # Turn onto the bearing to the marker centre. This is the
            # correction that actually zeroes the offset, it holds the
            # standoff exactly, and it needs no normal to express.
            if not self._go(job, rp.position.x, rp.position.y, math.atan2(dy, dx)):
                return False
        logger.warning(
            "ArUco dock: %.0f mm off the marker line — not reversing — %s", abs(lateral), self._scene(marker_id)
        )
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
            logger.warning("ArUco dock: marker %d gone before the reverse — %s", marker_id, self._scene(marker_id))
            return False
        mx, my = float(state.position[0]), float(state.position[1])

        # Turn in place: rear toward the charger.
        pose = self.cli.pose
        if not self._go(job, pose.position.x, pose.position.y, wrap_pi(pose.rotation.angle_z.radians + math.pi)):
            return False

        logger.info(
            "ArUco dock: reversing into the charger (stop at range ≤ %.0f mm or contacts) — %s",
            self.DOCKED_MARKER_DIST_MM - self.REVERSE_OVERSHOOT_MM,
            self._scene(marker_id),
        )
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
                logger.info("ArUco dock: reached the docked distance — %s", self._scene(marker_id))
                return self._wait_charging(marker_id)
            self.cli.drive_wheels(
                lwheel_speed=self.REVERSE_SPEED_MMPS,
                rwheel_speed=self.REVERSE_SPEED_MMPS,
            )
            time.sleep(0.05)

        self.cli.stop_all_motors()
        if self._charging():  # contacts landed as we stopped
            return True
        logger.warning(
            "ArUco dock: reverse timed out (%.0f s) — %s",
            self.REVERSE_TIMEOUT_S,
            self._scene(marker_id),
        )
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
            "ArUco dock: target (%.0f, %.0f) at %.1f deg not reached within %.0f s — %.0f mm / %.1f deg off, "
            "continuing; robot (%.0f, %.0f, %.1f deg)",
            x,
            y,
            math.degrees(heading),
            self.ARRIVE_TIMEOUT_S,
            dist_err,
            math.degrees(ang_err),
            rp.position.x,
            rp.position.y,
            math.degrees(rp.rotation.angle_z.radians),
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
        logger.warning(
            "ArUco dock: turn did not converge (timeout) — target %.1f deg, actual %.1f deg",
            math.degrees(heading),
            math.degrees(cli.pose.rotation.angle_z.radians),
        )
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
        than its rotation, and by this point the robot is on the marker's
        centreline, so range and "distance along the centreline" agree.
        """
        rp = self.cli.pose
        return math.hypot(rp.position.x - mx, rp.position.y - my) <= (
            self.DOCKED_MARKER_DIST_MM - self.REVERSE_OVERSHOOT_MM
        )

    def _wait_charging(self, marker_id: int) -> bool:
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
            logger.warning(
                "ArUco dock: reached the docked distance but the contacts did not register — not docked — %s",
                self._scene(marker_id),
            )
            return False
        return True

    def _charging(self) -> bool:
        with self.ds.lock:
            return bool(self.ds.on_charger or self.ds.charging)
