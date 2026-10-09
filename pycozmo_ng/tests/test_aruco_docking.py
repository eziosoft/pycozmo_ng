"""Tests for ArucoDocking — vantage, normal, standoff, reverse into the charger.

``ArucoDocking`` is constructible from plain dependencies (mock client,
marker feed, dock state, motion controller) so the whole sequence runs
without a robot: the guard when the marker is unknown, the sidestep ->
measure -> standoff -> reverse flow, the ``on_docked`` callback that
anchors the world origin, the refusal to anchor after a failed dock, the
timeout that gives up without docking, and cancellation.

The sequence deliberately never re-derives the marker's normal while
driving straight at it: a fronto-parallel view is the ill-conditioned case
``marker_observation`` exists to avoid (see that module's docstring). The
tests below pin that structure down as well as the outcome.

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
from pycozmo_ng.navigation.marker_observation import (
    MIN_USABLE_OBLIQUITY_DEG,
    TARGET_OBLIQUITY_DEG,
    obliquity_deg,
)
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
    converge."""

    def __init__(self, x: float = 500.0, y: float = 0.0, yaw: float = 0.0):
        self.pose = _pose(x, y, yaw)
        self.go_to_pose = MagicMock(side_effect=self._go_to_pose)
        self.drive_wheels = MagicMock(side_effect=self._drive_wheels)
        self.stop_all_motors = MagicMock()

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
    docking.VANTAGE_DWELL_S = 0.1
    docking.SAMPLE_WINDOW_S = 0.05
    docking.STANDOFF_DWELL_S = 0.1
    docking.ALIGN_DWELL_S = 0.1
    return docking, cli, ds, motion, on_docked


class TestDockGuards:
    def test_unknown_marker_starts_nothing(self):
        docking, cli, _ds, motion, on_docked = _make(marker_known=False)
        assert docking.dock(0) is False
        assert motion.active_names == ()
        cli.go_to_pose.assert_not_called()
        on_docked.assert_not_called()


class TestDockSequence:
    def test_sidestep_then_standoff_then_reverse(self):
        """Sidestep to a vantage, read the normal, drive the standoff, reverse.

        Two path segments: the sideways move onto the vantage, then the
        drive to the standoff. The 180° turn must not send one.
        """
        docking, cli, ds, motion, on_docked = _make()
        ds.charging = True  # contacts detect immediately on reverse

        assert docking.dock(0) is True
        assert ArucoDocking.JOB_ARUCO_DOCK in motion.active_names
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        # First drive is the sidestep: it must NOT be the dead-ahead point
        # that makes the normal ill-conditioned.
        assert cli.go_to_pose.call_count >= 2
        vantage = cli.go_to_pose.call_args_list[0].args[0]
        # The lateral step is VANTAGE_DIST * sin(TARGET_OBLIQUITY), so it
        # follows the geometry rather than being a magic number.
        expected_step = docking.VANTAGE_DIST_MM * math.sin(math.radians(TARGET_OBLIQUITY_DEG))
        assert abs(vantage.position.y) > expected_step * 0.9, "first move must be a sidestep off the marker's normal"
        assert math.isclose(math.hypot(vantage.position.x, vantage.position.y), docking.VANTAGE_DIST_MM, rel_tol=0.05)

        # And that vantage must actually clear the threshold the dock
        # gates on — the point of the sidestep is the angle it buys.
        assert (
            obliquity_deg((0.0, 0.0), (1.0, 0.0), (vantage.position.x, vantage.position.y)) >= MIN_USABLE_OBLIQUITY_DEG
        )
        assert TARGET_OBLIQUITY_DEG >= MIN_USABLE_OBLIQUITY_DEG

        # Last drive lands on the standoff ~250 mm out, facing the marker
        # *from the standoff* — not from the vantage, whose bearing
        # differs by the obliquity and left the marker out of the
        # camera's view on arrival ("not re-detected during settle").
        standoff = cli.go_to_pose.call_args_list[-1].args[0]
        assert math.isclose(math.hypot(standoff.position.x, standoff.position.y), docking.APPROACH_DIST_MM, rel_tol=0.2)
        expected_heading = math.atan2(0.0 - standoff.position.y, 0.0 - standoff.position.x)
        assert abs(wrap_pi(standoff.rotation.angle_z.radians - expected_heading)) < 1e-9

        turn = [c for c in cli.drive_wheels.call_args_list if c.kwargs["lwheel_speed"] == -c.kwargs["rwheel_speed"]]
        assert turn
        assert on_docked.call_count == 1

    def test_head_on_view_is_refused(self):
        """A view that never gains obliquity must not aim a reverse.

        This is the whole point of the vantage: with a fronto-parallel
        marker the normal's bearing is meaningless, and docking on it aims
        the reverse metres off. When every attempt comes back too head-on
        the dock must decline and leave the world origin alone.
        """
        docking, cli, ds, motion, on_docked = _make(x=200.0)  # dead ahead
        docking.ARRIVE_TIMEOUT_S = 0.05  # the pinned rig never "arrives"
        # Pin the robot on the marker's normal line: it never actually
        # sidesteps, so no measurement can ever gain obliquity.
        cli.go_to_pose = MagicMock(side_effect=lambda pose, **kw: setattr(cli, "pose", _pose(200.0, 0.0, 0.0)))

        docking.dock(0)
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names, timeout=10.0)

        # Refused before the reverse, so the contacts never mattered and
        # the world origin is untouched.
        on_docked.assert_not_called()
        assert not ds.charging
        reverse = [
            c
            for c in cli.drive_wheels.call_args_list
            if c.kwargs["lwheel_speed"] == c.kwargs["rwheel_speed"] == docking.REVERSE_SPEED_MMPS
        ]
        assert not reverse, "must not reverse on a head-on view"

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
        docking.VANTAGE_DWELL_S = 5.0  # park the job in the vantage dwell

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
        docking.VANTAGE_DWELL_S = 1.0

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
