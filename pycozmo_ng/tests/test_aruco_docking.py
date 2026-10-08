"""Tests for ArucoDocking — approach, refresh pose, reverse into the charger.

``ArucoDocking`` is constructible from plain dependencies (mock client,
marker feed, dock state, motion controller) so the whole sequence runs
without a robot: the guard when the marker is unknown, the approach +
reverse flow, the ``on_docked`` callback that anchors the world origin,
the timeout that gives up without docking, and cancellation.

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
        self.pose = _pose(self.pose.position.x, self.pose.position.y, yaw)


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


def _make(marker_known: bool = True, x: float = 500.0, marker_move: tuple[float, float] | None = None):
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
    docking.SETTLE_S = 0.1  # keep the tests fast
    return docking, cli, ds, motion, on_docked


class TestDockGuards:
    def test_unknown_marker_starts_nothing(self):
        docking, cli, _ds, motion, on_docked = _make(marker_known=False)
        assert docking.dock(0) is False
        assert motion.active_names == ()
        cli.go_to_pose.assert_not_called()
        on_docked.assert_not_called()


class TestDockSequence:
    def test_approach_turn_reverse_then_docked(self):
        """Face the marker (20 cm out), turn 180°, reverse, dock."""
        docking, cli, ds, motion, on_docked = _make()
        ds.charging = True  # contacts detect immediately on reverse

        assert docking.dock(0) is True
        assert ArucoDocking.JOB_ARUCO_DOCK in motion.active_names
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        # One path for the approach — the 180° turn must not send one.
        assert cli.go_to_pose.call_count == 1
        approach = cli.go_to_pose.call_args_list[0].args[0]
        assert math.isclose(approach.position.x, 200.0)
        assert math.isclose(approach.position.y, 0.0)
        assert math.isclose(abs(approach.rotation.angle_z.radians), math.pi)

        # The turn ran on drive_wheels (closed loop), then the reverse.
        turn = [c for c in cli.drive_wheels.call_args_list if c.kwargs["lwheel_speed"] == -c.kwargs["rwheel_speed"]]
        assert turn
        assert on_docked.call_count == 1

    def test_correction_uses_refreshed_marker_pose(self):
        """Camera refreshes the marker pose → approach point is recomputed.

        Reproduces the live log: "marker pose updated — correcting to
        (80, 3)" — marker 0 fused to (-70, 3) after the camera saw it.
        """
        docking, cli, ds, _motion, on_docked = _make(marker_move=(-70.0, 3.0))
        ds.charging = True  # complete quickly

        assert docking.dock(0) is True
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in docking.motion.active_names)

        assert cli.go_to_pose.call_count == 2  # approach, then correction
        first, second = (c.args[0] for c in cli.go_to_pose.call_args_list)
        assert math.isclose(first.position.x, 200.0)  # from the stale pose
        assert math.isclose(second.position.x, 130.0)  # marker + 200 mm
        assert math.isclose(second.position.y, 3.0)
        assert on_docked.call_count == 1

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
        docking, cli, _ds, motion, on_docked = _make()
        docking.SETTLE_S = 5.0  # park the job in the settle phase

        docking.dock(0)
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK in motion.active_names)
        motion.cancel_all()
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        assert cli.drive_wheels.call_count == 0
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
        docking.SETTLE_S = 1.0

        docking.dock(0)
        first = next(t for t in motion._jobs if t.name == ArucoDocking.JOB_ARUCO_DOCK)
        docking.dock(0)

        assert first.cancelled
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)


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
