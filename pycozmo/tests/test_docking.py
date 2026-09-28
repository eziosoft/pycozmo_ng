"""Tests for the Docking seam (extracted from Navigator.dock).

``Docking`` is now constructible from plain dependencies — dock state, the
shared motion controller, and an origin-reset callback — so the dock
sequence can be exercised without a ``Navigator`` or a robot: the guards
that decide whether a dock can start, registration of the shared ``dock``
job (the one ``Navigator.stop()`` cancels), contact/stall detection with
retries, and the origin reset that only happens once charging is seen.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.docking import Docking
from pycozmo.navigation.motion_controller import MotionController


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


def _make_docking(x: float = 0.0, y: float = 0.0, with_pose: bool = True):
    """Docking with a mock CLI (fixed pose) and a spy origin reset."""
    ds = DockState()
    if with_pose:
        ds.dock_pose = _pose(0.0, 0.0)
    motion = MotionController()
    reset_origin = MagicMock()
    docking = Docking(dock_state=ds, motion=motion, reset_origin=reset_origin)
    cli = MagicMock()
    type(cli).pose = property(lambda self: _pose(x, y))
    docking.cli = cli
    return docking, cli, ds, motion, reset_origin


class TestDockGuards:
    def test_no_client_starts_nothing(self):
        docking = Docking(
            dock_state=DockState(), motion=MotionController(), reset_origin=MagicMock()
        )
        docking.dock()
        assert docking.motion.active_names == ()

    def test_no_dock_pose_starts_nothing(self):
        docking, cli, _ds, motion, _reset = _make_docking(with_pose=False)
        docking.dock()
        assert motion.active_names == ()
        cli.go_to_pose.assert_not_called()


class TestDockJob:
    def test_dock_registers_shared_job(self):
        """The dock job runs on the injected controller — what stop() cancels."""
        docking, _cli, ds, motion, _reset = _make_docking()
        ds.charging = True  # complete quickly: this test is about the job name
        docking.dock()
        assert Docking.JOB_DOCK in motion.active_names
        assert _wait_until(lambda: Docking.JOB_DOCK not in motion.active_names)

    def test_charging_resets_origin(self):
        """Charge seen after the reverse attempt → motors stop, origin re-anchored."""
        docking, cli, ds, _motion, reset_origin = _make_docking()
        ds.charging = True
        docking.dock()
        assert _wait_until(lambda: Docking.JOB_DOCK not in docking.motion.active_names)

        cli.go_to_pose.assert_called_once()
        cli.stop_all_motors.assert_called_once()
        assert reset_origin.call_count == 1

    def test_stall_retries_then_gives_up_without_reset(self, monkeypatch):
        """Pose never approaches the aim point: reverse stalls, retries, no reset."""
        monkeypatch.setattr(Docking, "STALL_ITERS", 2)  # keep the test fast
        docking, cli, _ds, _motion, reset_origin = _make_docking(x=200.0, y=200.0)
        docking.dock()
        assert _wait_until(lambda: Docking.JOB_DOCK not in docking.motion.active_names)

        speeds = [c.kwargs["lwheel_speed"] for c in cli.drive_wheels.call_args_list]
        reverse = [s for s in speeds if s < 0]
        retry = [s for s in speeds if s > 0]
        # Two reverse ticks per attempt, then back off to a retry speed.
        assert len(reverse) == Docking.MAX_RETRIES * 2
        assert set(reverse) == {-Docking.REVERSE_SPEED_MMPS}
        assert len(retry) == Docking.MAX_RETRIES - 1
        assert set(retry) == {Docking.RETRY_SPEED_MMPS}

        reset_origin.assert_not_called()
        cli.stop_all_motors.assert_called_once()

    def test_cancel_stops_dock_without_reset(self):
        """Cancelling the shared controller (Navigator.stop) ends the job."""
        docking, cli, _ds, motion, reset_origin = _make_docking(x=200.0, y=200.0)
        docking.dock()
        motion.cancel_all()
        assert _wait_until(lambda: Docking.JOB_DOCK not in motion.active_names)

        reset_origin.assert_not_called()
        cli.stop_all_motors.assert_called_once()


class TestDockSupersede:
    def test_new_request_cancels_the_running_one(self):
        """A second dock() request supersedes the first: one job, old one cancelled."""
        docking, _cli, _ds, motion, _reset = _make_docking(x=200.0, y=200.0)
        docking.dock()
        first = next(t for t in motion._jobs if t.name == Docking.JOB_DOCK)

        docking.dock()

        assert first.cancelled
        assert _wait_until(lambda: Docking.JOB_DOCK not in motion.active_names)
