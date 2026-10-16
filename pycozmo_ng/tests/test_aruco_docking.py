"""Tests for ArucoDocking — the point in front of the charger, centring, reverse into the charger.

``ArucoDocking`` is constructible from plain dependencies (mock client,
marker feed, dock state, motion controller) so the whole sequence runs
without a robot: the guard when the marker is unknown, the head level,
the drive to the point in front of the charger -> settle -> replan ->
centre -> reverse flow, the refusal to dock when the charger's facing axis
is unresolved, the ``on_docked`` callback that anchors the world origin,
the refusal to anchor after a failed dock, the timeout that gives up
without docking, and cancellation.

The sequence drives to the point in front of the charger — the charger's
own pose plus ``APPROACH_DIST_MM`` along its facing axis — and never
estimates the marker's bearing from its rotation: the standoff heading
comes from positions (the bearing back to the marker), so facing the
marker from the standoff is exact by construction. The marker sits at the
back end of the dock, on its centreline, so that point is on the charger's
centreline and the open-loop reverse runs straight in. When the first
close look at the marker improves its pose, the point moves with it and
the approach runs again (once).

The last class covers the browser wiring: the Z key must reach
``Navigator.dock_on_aruco`` — not the legacy pose-based dock.
"""

import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from pycozmo_ng.navigation.aruco_docking import ArucoDocking
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.motion_controller import MotionController
from pycozmo_ng.navigation.steering import wrap_pi
from pycozmo_ng.navigation.world_map import ArucoState


def _pose(x: float, y: float, yaw: float = 0.0):
    return SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=yaw)),
    )


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class _FakeCli:
    """Minimal client: ``pose`` follows ``go_to_pose``; ``drive_wheels``
    applies crude differential-drive kinematics so closed-loop turns
    converge; ``set_head_angle`` moves ``head_angle`` immediately."""

    def __init__(self, x: float = 500.0, y: float = 0.0, yaw: float = 0.0):
        self.pose = _pose(x, y, yaw)
        self.head_angle = SimpleNamespace(radians=math.radians(-25.0))
        self.go_to_pose = MagicMock(side_effect=self._go_to_pose)
        self.drive_wheels = MagicMock(side_effect=self._drive_wheels)
        self.stop_all_motors = MagicMock()
        self.set_head_angle = MagicMock(side_effect=self._set_head_angle)

    def _set_head_angle(self, radians: float, **kwargs):
        del kwargs
        self.head_angle = SimpleNamespace(radians=radians)

    def _go_to_pose(self, pose, **kwargs):
        del kwargs
        self.pose = _pose(pose.position.x, pose.position.y, pose.rotation.angle_z.radians)
        return True

    def _drive_wheels(self, lwheel_speed: float, rwheel_speed: float, **kwargs):
        del kwargs
        # Time-consistent differential drive: one 30 ms tick, wheelbase
        # ~65 mm → yaw rate = (r - l) / wheelbase. Big enough to converge
        # fast, small enough never to overshoot the proportional command.
        yaw = self.pose.rotation.angle_z.radians + 0.002 * (rwheel_speed - lwheel_speed)
        # Straight motion advances along the heading (mean wheel speed,
        # ~0.15 s effective tick) so the reverse actually covers ground;
        # pure turns have a zero mean → no translation, like the real robot.
        dist = 0.15 * (lwheel_speed + rwheel_speed) / 2.0
        x = self.pose.position.x + dist * math.cos(yaw)
        y = self.pose.position.y + dist * math.sin(yaw)
        self.pose = _pose(x, y, yaw)


class _MarkerFeed:
    """Serves the marker state; every read is a fresh detection.

    ``refresh_xy`` moves the marker after ``refresh_after`` reads —
    mimicking the fused pose shifting once the camera really sees it
    (the "marker pose updated — correcting to ..." case).
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        normal_yaw: float = 0.0,
        refresh_xy: tuple[float, float] | None = None,
        refresh_after: int = 2,
    ):
        z = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
        y_up = np.array([0.0, 0.0, 1.0])
        self._xy = (x, y)
        self._refresh_xy = refresh_xy
        self._refresh_after = refresh_after
        self._state = ArucoState(
            marker_id=0,
            position=np.array([x, y, 0.0]),
            rotation_matrix=np.column_stack([np.cross(y_up, z), y_up, z]),
            confidence=1.0,
            size_mm=22.5,
            age=10,
        )
        self._reads = 0

    def get(self, marker_id: int):
        if marker_id != self._state.marker_id:
            return None
        self._reads += 1
        if self._refresh_xy is not None and self._reads > self._refresh_after:
            self._xy = self._refresh_xy
        return ArucoState(
            marker_id=self._state.marker_id,
            position=np.array([self._xy[0], self._xy[1], 0.0]),
            rotation_matrix=self._state.rotation_matrix,
            confidence=1.0,
            size_mm=22.5,
            age=10,
            updated_at=time.monotonic() + 0.2 * self._reads,
        )


class _GatedMarkerFeed:
    """Marker state that reads fresh only while the camera can see it.

    Models the live failure: while the marker is out of frame the fused
    position stays at the last (biased) estimate, so the computed bearing
    misses — and the moment a detection lands, the state snaps to the
    true position. ``visible_yaw`` is the heading from which the marker
    really is in view, deliberately off the stale estimate's bearing.
    """

    def __init__(
        self,
        cli,
        true_xy: tuple[float, float],
        stale_xy: tuple[float, float],
        visible_yaw: float,
        window_deg: float,
    ):
        self._cli = cli
        self._true = true_xy
        self._stale = stale_xy
        self._visible_yaw = visible_yaw
        self._window = math.radians(window_deg)
        self._frozen_ts = time.monotonic() - 10.0  # long stale
        z = np.array([1.0, 0.0, 0.0])
        y_up = np.array([0.0, 0.0, 1.0])
        self._rot = np.column_stack([np.cross(y_up, z), y_up, z])

    def get(self, marker_id: int):
        if marker_id != 0:
            return None
        yaw = self._cli.pose.rotation.angle_z.radians
        visible = abs(wrap_pi(yaw - self._visible_yaw)) <= self._window
        xy = self._true if visible else self._stale
        return ArucoState(
            marker_id=0,
            position=np.array([xy[0], xy[1], 0.0]),
            rotation_matrix=self._rot,
            confidence=1.0,
            size_mm=22.5,
            age=10,
            updated_at=time.monotonic() if visible else self._frozen_ts,
        )


class _ImprovingMarkerFeed:
    """Charger pose that improves only up close — and can keep improving.

    The live pattern: the pose that plans the point in front of the charger
    is measured from a distance, off axis, and only the first close,
    head-on look at the marker corrects it. Until the robot is within
    ``near_mm`` of ``far_xy`` this feed serves ``far_xy``; after that it
    serves ``near_xy`` plus ``step_mm`` per ``step_every`` reads since
    (capped at ``max_mm``). ``step_mm=0`` is a single improvement; a
    positive step is a pose that never settles.

    Reads stand in for elapsed time: every log line, settle poll and
    alignment check fetches the state, so a fast test still accumulates
    plenty of them.
    """

    def __init__(
        self,
        cli,
        far_xy: tuple[float, float] = (0.0, 0.0),
        near_xy: tuple[float, float] = (40.0, 0.0),
        near_mm: float = 300.0,
        step_mm: float = 0.0,
        step_every: int = 4,
        max_mm: float = 120.0,
    ):
        self._cli = cli
        self._far = far_xy
        self._near = near_xy
        self._near_mm = near_mm
        self._step = step_mm
        self._every = step_every
        self._max = max_mm
        self._reads_close = 0
        #: Where the feed currently says the charger is (mm, world x).
        self.x = far_xy[0]

    def get(self, marker_id: int):
        if marker_id != 0:
            return None
        rp = self._cli.pose
        close = math.hypot(rp.position.x - self._far[0], rp.position.y - self._far[1]) <= self._near_mm
        if close:
            self._reads_close += 1
            drift = self._step * float(self._reads_close // self._every)
            self.x = self._near[0] + min(drift, self._max)
        else:
            self.x = self._far[0]
        y = self._near[1] if close else self._far[1]
        z = np.array([1.0, 0.0, 0.0])
        y_up = np.array([0.0, 0.0, 1.0])
        return ArucoState(
            marker_id=0,
            position=np.array([self.x, y, 0.0]),
            rotation_matrix=np.column_stack([np.cross(y_up, z), y_up, z]),
            confidence=1.0,
            size_mm=22.5,
            age=10,
            updated_at=time.monotonic(),
        )


class _NoAxisMarkerFeed:
    """Charger marker whose facing axis was never resolved (zero normal)."""

    def get(self, marker_id: int):
        if marker_id != 0:
            return None
        return ArucoState(
            marker_id=0,
            position=np.array([0.0, 0.0, 0.0]),
            rotation_matrix=np.zeros((3, 3)),
            confidence=1.0,
            size_mm=22.5,
            age=10,
            updated_at=time.monotonic(),
        )


def _make(
    marker_known: bool = True,
    x: float = 500.0,
    marker_move: tuple[float, float] | None = None,
):
    ds = DockState()
    motion = MotionController()
    cli = _FakeCli(x=x)
    feed = _MarkerFeed(refresh_xy=marker_move)
    on_docked = MagicMock()
    docking = ArucoDocking(
        cli=cli,
        get_marker_state=feed.get if marker_known else (lambda mid: None),
        dock_state=ds,
        motion=motion,
        on_docked=on_docked,
    )
    # keep the tests fast: every dwell is a couple of poll ticks
    docking.STANDOFF_DWELL_S = 0.1
    docking.ALIGN_DWELL_S = 0.1
    docking.REVERSE_TURN_SETTLE_S = 0.05
    docking.REVERSE_AIM_SETTLE_S = 0.05
    return docking, cli, ds, motion, on_docked


class TestDockGuards:
    def test_unknown_marker_starts_nothing(self):
        docking, cli, _ds, motion, on_docked = _make(marker_known=False)
        assert docking.dock(0) is False
        assert motion.active_names == ()
        cli.go_to_pose.assert_not_called()
        on_docked.assert_not_called()

    def test_unresolved_charger_axis_refuses_the_dock(self):
        """No facing axis means no point in front of the charger — refuse.

        The sequence drives to that point or nowhere. A standoff picked any
        other way (the line of sight, say) is off the charger's centreline,
        and the open-loop reverse would miss the dock's back posts.
        """
        ds = DockState()
        motion = MotionController()
        cli = _FakeCli(x=500.0)
        on_docked = MagicMock()
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=_NoAxisMarkerFeed().get,
            dock_state=ds,
            motion=motion,
            on_docked=on_docked,
        )

        assert docking.dock(0) is True  # the marker exists, so the job starts
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        cli.go_to_pose.assert_not_called()
        cli.drive_wheels.assert_not_called()
        on_docked.assert_not_called()


class TestDockSequence:
    def test_head_is_leveled_before_the_approach(self):
        """The head goes level before anything is planned or driven.

        The standoff is planned from the charger's fused pose, and a head
        left tilted by whatever ran before (the demo looks down, a cube
        approach tilts down) leaves the marker at the frame's edge, where
        that pose is poor. Pin the order: head first, then the drive.
        """
        docking, cli, ds, motion, _on_docked = _make()
        ds.charging = True
        events: list[str] = []
        real_set_head = cli.set_head_angle.side_effect
        real_go = cli.go_to_pose.side_effect

        def set_head(*args, **kwargs):
            events.append("head")
            return real_set_head(*args, **kwargs)

        def go(*args, **kwargs):
            events.append("go")
            return real_go(*args, **kwargs)

        cli.set_head_angle.side_effect = set_head
        cli.go_to_pose.side_effect = go

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        assert events[0] == "head", "the head must be leveled before the approach"
        assert "go" in events
        assert cli.set_head_angle.call_args.args[0] == docking.LEVEL_HEAD_ANGLE_RAD

    def test_standoff_is_the_point_in_front_of_the_charger(self):
        """One drive: straight ahead to the point in front of the charger, then reverse.

        That point sits on the charger's facing axis, so the robot arrives
        on the charger's centreline. The robot starts on the axis (at
        (500, 0) with the marker's facing axis along +x), so the point is
        dead ahead at ``APPROACH_DIST_MM`` — on the axis, no lateral step.
        """
        docking, cli, ds, motion, on_docked = _make()
        ds.charging = True  # contacts detect immediately on reverse

        assert docking.dock(0) is True
        assert ArucoDocking.JOB_ARUCO_DOCK in motion.active_names
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        # The whole approach is one path segment: the robot starts at
        # (500, 0) with the marker at the origin (facing axis along +x), so
        # the point in front of the charger is dead ahead at 200 mm.
        assert cli.go_to_pose.call_count >= 1
        standoff = cli.go_to_pose.call_args_list[0].args[0]
        assert math.isclose(standoff.position.x, docking.APPROACH_DIST_MM, rel_tol=0.05)
        assert abs(standoff.position.y) < 1e-6, "the standoff must sit on the charger's axis, not off to the side"
        assert abs(wrap_pi(standoff.rotation.angle_z.radians - math.pi)) < 1e-9

        turn = [c for c in cli.drive_wheels.call_args_list if c.kwargs["lwheel_speed"] == -c.kwargs["rwheel_speed"]]
        assert turn
        assert on_docked.call_count == 1

    def test_the_improved_pose_replans_and_redrives_the_point_in_front(self):
        """The first standoff is the first good look: drive it again from the better pose.

        The feed serves the distant (0, 0) until the robot stands at the
        200 mm standoff, then corrects the charger to 40 mm — the live
        pattern. That moves the point in front of the charger 40 mm, past
        ``REPLAN_TOL_MM``, so the robot must drive to the *new* point
        rather than absorb the difference with a turn: it ends up on the
        point, not near it.
        """
        ds = DockState()
        motion = MotionController()
        cli = _FakeCli(x=500.0)
        feed = _ImprovingMarkerFeed(cli)
        on_docked = MagicMock()
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=feed.get,
            dock_state=ds,
            motion=motion,
            on_docked=on_docked,
        )
        docking.STANDOFF_DWELL_S = 0.1
        ds.charging = True

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        standoffs = [call.args[0] for call in cli.go_to_pose.call_args_list]
        assert len(standoffs) >= 2, "an improved pose must replan and re-drive"
        assert math.isclose(standoffs[0].position.x, docking.APPROACH_DIST_MM, rel_tol=0.05)
        assert math.isclose(standoffs[1].position.x, 40.0 + docking.APPROACH_DIST_MM, rel_tol=0.05)
        assert on_docked.call_count == 1

    def test_the_approach_replans_at_most_once(self):
        """A pose that keeps improving does not loop the approach forever.

        One replan is the fix (the first close look); a second pass would
        be chasing the fusion filter's own noise. The feed here improves
        again on every few reads once the robot is close, so an uncapped
        loop would drive to 280, 320, 360 … — exactly two drives means the
        cap held.
        """
        ds = DockState()
        motion = MotionController()
        cli = _FakeCli(x=500.0)
        feed = _ImprovingMarkerFeed(cli, step_mm=40.0)
        on_docked = MagicMock()
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=feed.get,
            dock_state=ds,
            motion=motion,
            on_docked=on_docked,
        )
        docking.STANDOFF_DWELL_S = 0.1
        ds.charging = True

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        standoffs = [call.args[0] for call in cli.go_to_pose.call_args_list]
        assert len(standoffs) == 2, "one replan, not a loop"
        assert feed.x >= 40.0, "the pose must actually have kept moving"
        assert on_docked.call_count == 1

    def test_off_axis_start_still_lands_on_the_charger_axis(self):
        """An off-axis start still lands on the charger's axis, not the line of sight.

        The point in front of the charger is the charger's own geometry, so
        regardless of where the robot starts it arrives on the charger's
        centreline — the reverse then runs straight back into the dock's
        back posts. Gating on the marker's normal-derived obliquity would
        take the marker's fused rotation, which is ill-conditioned (off by
        tens of degrees at a head-on start), so a guard would refuse the
        poses that dock best. The dock proceeds and the centring step does
        what it can.
        """
        docking, cli, ds, motion, on_docked = _make()
        ds.charging = True
        # Robot 45 deg off the marker's axis (marker facing axis along +x).
        cli.pose = _pose(350.0, 350.0, 0.0)

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        # The point in front of the charger is on its axis: 200 mm along
        # +x from the origin — NOT on the line of sight from (350, 350),
        # which would land at ~(177, 177).
        standoff = cli.go_to_pose.call_args_list[0].args[0]
        assert math.isclose(standoff.position.x, docking.APPROACH_DIST_MM, rel_tol=0.05)
        assert abs(standoff.position.y) < 1e-6, "the standoff must be on the charger's axis (+x), not the line of sight"
        # Heading is the bearing from the standoff back to the marker: pi.
        assert abs(wrap_pi(standoff.rotation.angle_z.radians - math.pi)) < 1e-9
        assert on_docked.call_count == 1

    def test_reverse_starts_from_rest_after_the_turn(self):
        """No reverse command until the turn has stopped and settled.

        The browser's reverse key always starts on a stationary robot and
        drives straight; the dock used to command the reverse while the
        180 deg turn was still coasting, one tread spun up first, and the
        chassis twisted ~5 deg off the line — enough to carry the
        open-loop reverse past the charger. Pin the ordering the fix
        relies on: turn -> stop -> settle -> re-aim stop -> settle ->
        reverse.
        """
        docking, cli, _ds, motion, _on_docked = _make()
        docking.REVERSE_TURN_SETTLE_S = 0.25
        docking.REVERSE_AIM_SETTLE_S = 0.15
        docking.REVERSE_TIMEOUT_S = 0.3  # a few reverse ticks, then give up

        events: list[tuple[float, str]] = []
        real_drive = cli.drive_wheels.side_effect

        def drive(*args, **kwargs):
            both = kwargs["lwheel_speed"] == kwargs["rwheel_speed"]
            kind = "reverse" if both and kwargs["lwheel_speed"] == docking.REVERSE_SPEED_MMPS else "turn"
            events.append((time.monotonic(), kind))
            return real_drive(*args, **kwargs)

        def stop(*args, **kwargs):
            del args, kwargs
            events.append((time.monotonic(), "stop"))

        cli.drive_wheels.side_effect = drive
        cli.stop_all_motors.side_effect = stop

        with motion.begin("test_reverse") as job:
            assert docking._reverse_into_charger(job, 0) is False  # timed out, never docked

        assert any(kind == "turn" for _t, kind in events), "the 180 deg turn must have run"
        stops = [t for t, kind in events if kind == "stop"]
        revs = [t for t, kind in events if kind == "reverse"]
        assert len(stops) >= 2, "the turn stops, then the re-aim stops again"
        # Turn -> full stop -> settle -> re-aim (measures the stopped
        # robot) -> stop -> settle -> only then the first reverse tick.
        assert stops[1] - stops[0] >= docking.REVERSE_TURN_SETTLE_S - 1e-3
        assert revs, "the robot must have tried to reverse"
        assert revs[0] - stops[1] >= docking.REVERSE_AIM_SETTLE_S - 1e-3

    def test_no_charging_does_not_reanchor_the_world(self):
        """Reaching the docked distance is not the same as docking.

        The geometric backstop fires on range alone, so a robot that bumps
        the charger off-centre gets there too. Anchoring the world origin
        on that would silently corrupt the map — so ``on_docked`` must not
        run without the contacts actually registering.
        """
        docking, cli, ds, motion, on_docked = _make()
        docking.REVERSE_TIMEOUT_S = 30.0  # must stop on geometry, not timeout

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        cli.stop_all_motors.assert_called()
        assert not ds.charging
        on_docked.assert_not_called()

    def test_reverse_stops_at_overshoot_backstop(self):
        """No charging flag → the robot still stops on the geometric backstop.

        The reverse must not rely on the charging flag alone (it can lag)
        but must also stop rather than drive into the charger forever.
        """
        docking, cli, ds, motion, _on_docked = _make()
        docking.REVERSE_TIMEOUT_S = 30.0

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        # Marker at (0,0) → backstop at 65 − 50 = 15 mm out.
        assert 5.0 <= math.hypot(cli.pose.position.x, cli.pose.position.y) <= 16.0
        assert not ds.charging

    def test_timeout_gives_up_without_docked(self):
        """No charge within the reverse timeout → no post-dock anchoring."""
        docking, cli, ds, _motion, on_docked = _make()
        docking.REVERSE_TIMEOUT_S = 0.2

        docking.dock(0)
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in docking.motion.active_names)

        reverse = [
            c
            for c in cli.drive_wheels.call_args_list
            if c.kwargs["lwheel_speed"] == c.kwargs["rwheel_speed"] == docking.REVERSE_SPEED_MMPS
        ]
        assert reverse  # the robot did try to back into the charger
        cli.stop_all_motors.assert_called()
        assert not ds.charging
        on_docked.assert_not_called()

    def test_cancel_stops_dock_without_docked(self):
        """Navigator.stop() cancels the job mid-approach — no docking."""
        docking, _cli, _ds, motion, on_docked = _make()
        docking.STANDOFF_DWELL_S = 5.0  # park the job in the standoff settle

        docking.dock(0)
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK in motion.active_names)
        motion.cancel_all()
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        on_docked.assert_not_called()

    def test_zero_length_path_never_sent(self):
        """Turns must never go through ``go_to_pose``.

        A zero-length ``AppendPathSegLine`` makes the firmware assert
        ``y != 0 || x != 0`` in ``trig_fast.cpp`` (atan2(0, 0)) and drop
        the connection — the bug behind the Z-key dock crash.
        """
        docking, cli, _ds, motion, _on_docked = _make()
        cli.pose = _pose(150.0, 0.0, 0.0)  # robot already at the target

        with motion.begin("test_turn") as job:
            assert docking._go(job, 150.0, 0.0, math.pi) is True

        cli.go_to_pose.assert_not_called()  # no path segment at all
        assert math.isclose(abs(cli.pose.rotation.angle_z.radians), math.pi, abs_tol=0.1)

    def test_supersede_new_request_cancels_the_old(self):
        docking, _cli, ds, motion, _on_docked = _make()
        ds.charging = True
        docking.STANDOFF_DWELL_S = 1.0

        docking.dock(0)
        first = next(t for t in motion._jobs if t.name == ArucoDocking.JOB_ARUCO_DOCK)
        docking.dock(0)

        assert first.cancelled
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)


class TestAlignToLine:
    """The reverse is open loop — a crooked approach docks crooked."""

    def test_off_line_robot_is_reaimed_at_the_marker(self):
        """Robot 40 mm off the marker's centreline → re-aimed onto it.

        The correction is a turn onto the bearing to the marker: aiming
        the robot at the marker is what zeroes the offset, and it holds the
        standoff distance exactly.
        """
        ds = DockState()
        motion = MotionController()
        # Standing 200 mm out and 40 mm to the side of the marker, facing
        # back along -x rather than at it — the classic "docked crooked"
        # geometry the reverse cannot recover from.
        cli = _FakeCli(x=200.0, y=-40.0, yaw=math.pi)
        feed = _MarkerFeed(x=0.0, y=0.0)
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=feed.get,
            dock_state=ds,
            motion=motion,
            on_docked=MagicMock(),
        )
        docking.ALIGN_DWELL_S = 0.1

        with motion.begin("test_align") as job:
            assert docking._align_to_line(job, 0) is True

        # Now pointing at the marker centre, from the same standoff.
        bearing = math.atan2(0.0 - cli.pose.position.y, 0.0 - cli.pose.position.x)
        assert math.isclose(cli.pose.rotation.angle_z.radians, bearing, abs_tol=0.05)
        # The turn is in place, so the robot has not moved: the standoff
        # from the origin is still 40 mm off the marker's centreline, but
        # the rear axis now points straight at it — which is what a reverse
        # needs. Distance to the marker is unchanged too.
        assert math.isclose(cli.pose.position.x, 200.0, abs_tol=1.0)
        assert math.isclose(cli.pose.position.y, -40.0, abs_tol=1.0)

    def test_aligned_robot_needs_no_correction(self):
        """Already aimed at the marker → nothing is sent."""
        docking, cli, _ds, motion, _on_docked = _make()
        cli.pose = _pose(200.0, 0.0, math.pi)  # marker dead ahead

        with motion.begin("test_align") as job:
            assert docking._align_to_line(job, 0) is True

        cli.go_to_pose.assert_not_called()
        cli.drive_wheels.assert_not_called()


class TestMarkerOutOfFrame:
    """The standoff settle must survive a marker the camera cannot see yet."""

    def test_sweep_turns_the_camera_onto_the_marker(self):
        """A stale marker position aims the camera ~10 deg off → the sweep finds it.

        The fused position is only as good as the last detection: while
        the marker is out of frame it stays biased, so the computed
        bearing can miss by more than the camera's useful view (live log
        after the standoff fix: "10 degrees more to the left and it will
        see it", then ``marker not re-detected during settle``). The
        settle must sweep the heading instead of aborting the dock — and
        land aimed at where the marker really is, since that is the
        heading the open-loop reverse needs.
        """
        ds = DockState()
        motion = MotionController()
        # At the standoff, facing the stale estimate at the origin.
        cli = _FakeCli(x=250.0, y=0.0, yaw=math.pi)
        visible_yaw = math.pi + math.radians(10.0)  # the marker really sits 10 deg to the left
        true_xy = (
            cli.pose.position.x + 250.0 * math.cos(visible_yaw),
            cli.pose.position.y + 250.0 * math.sin(visible_yaw),
        )
        feed = _GatedMarkerFeed(
            cli,
            true_xy=true_xy,
            stale_xy=(0.0, 0.0),
            visible_yaw=visible_yaw,
            window_deg=8.0,  # wider than the turn tolerance, narrower than the miss
        )
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=feed.get,
            dock_state=ds,
            motion=motion,
            on_docked=MagicMock(),
        )
        docking.ALIGN_DWELL_S = 0.1

        with motion.begin("test_align") as job:
            assert docking._align_to_line(job, 0) is True

        # Finished aimed at the marker's true position, within the turn
        # tolerance — the heading the reverse runs on.
        assert abs(wrap_pi(cli.pose.rotation.angle_z.radians - visible_yaw)) < 0.06

    def test_unfindable_marker_aborts(self):
        """Never in view within the sweep window → no blind reverse."""
        ds = DockState()
        motion = MotionController()
        cli = _FakeCli(x=250.0, y=0.0, yaw=math.pi)
        # Visible from nowhere the sweep reaches.
        feed = _GatedMarkerFeed(
            cli,
            true_xy=(250.0, -250.0),
            stale_xy=(0.0, 0.0),
            visible_yaw=math.pi + math.radians(40.0),
            window_deg=2.0,
        )
        docking = ArucoDocking(
            cli=cli,
            get_marker_state=feed.get,
            dock_state=ds,
            motion=motion,
            on_docked=MagicMock(),
        )
        docking.ALIGN_DWELL_S = 0.1
        docking.SEARCH_DWELL_S = 0.1

        with motion.begin("test_align") as job:
            assert docking._align_to_line(job, 0) is False


class TestZKeyWiring:
    """Regression: Z in the browser must dock on the ArUco marker (COZMO-66)."""

    def test_dock_action_routes_to_aruco_dock(self, monkeypatch):
        from pycozmo_ng.brain import Brain

        brain = Brain()
        try:
            spy = MagicMock(return_value=True)
            legacy = MagicMock()
            monkeypatch.setattr(brain.navigator, "dock_on_aruco", spy)
            monkeypatch.setattr(brain.navigator, "dock", legacy)
            brain.install_remote_control(MagicMock())

            brain.web_viewer.router.route('{"type": "dock"}')

            spy.assert_called_once_with()
            legacy.assert_not_called()
        finally:
            brain.stop_remote_control()
