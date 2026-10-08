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
    """Minimal client: ``pose`` follows the last ``go_to_pose`` goal."""

    def __init__(self, x: float = 500.0, y: float = 0.0, yaw: float = 0.0):
        self.pose = _pose(x, y, yaw)
        self.go_to_pose = MagicMock(side_effect=self._go_to_pose)
        self.drive_wheels = MagicMock()
        self.stop_all_motors = MagicMock()

    def _go_to_pose(self, pose, **kwargs):
        del kwargs
        self.pose = _pose(pose.position.x, pose.position.y, pose.rotation.angle_z.radians)
        return True


class _MarkerFeed:
    """Serves the marker state; every read is a fresh detection."""

    def __init__(self, x: float = 0.0, y: float = 0.0, normal_yaw: float = 0.0):
        z = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
        y_up = np.array([0.0, 0.0, 1.0])
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
        return ArucoState(
            marker_id=self._state.marker_id,
            position=self._state.position,
            rotation_matrix=self._state.rotation_matrix,
            confidence=1.0,
            size_mm=22.5,
            age=10,
            updated_at=time.monotonic() + 0.2 * self._reads,
        )


def _make(marker_known: bool = True, x: float = 500.0):
    ds = DockState()
    motion = MotionController()
    cli = _FakeCli(x=x)
    feed = _MarkerFeed()
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
        """Face the marker (15 cm out), turn 180°, reverse, dock."""
        docking, cli, ds, motion, on_docked = _make()
        ds.charging = True  # contacts detect immediately on reverse

        assert docking.dock(0) is True
        assert ArucoDocking.JOB_ARUCO_DOCK in motion.active_names
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in motion.active_names)

        # Approach point: 150 mm in front of the marker along its normal,
        # facing it — then the 180° turn in place.
        assert cli.go_to_pose.call_count == 2
        approach = cli.go_to_pose.call_args_list[0].args[0]
        assert math.isclose(approach.position.x, 150.0)
        assert math.isclose(approach.position.y, 0.0)
        assert math.isclose(abs(approach.rotation.angle_z.radians), math.pi)

        assert on_docked.call_count == 1

    def test_timeout_gives_up_without_docked(self):
        """No charge within the reverse timeout → no post-dock anchoring."""
        docking, cli, ds, _motion, on_docked = _make()
        docking.REVERSE_TIMEOUT_S = 0.2

        docking.dock(0)
        assert _wait_until(lambda: ArucoDocking.JOB_ARUCO_DOCK not in docking.motion.active_names)

        speeds = [c.kwargs["lwheel_speed"] for c in cli.drive_wheels.call_args_list]
        assert speeds and all(s < 0 for s in speeds)  # only reverse
        cli.stop_all_motors.assert_called_once()
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
