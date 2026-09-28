"""Tests for Navigator.stop() and the browser navigate-cancel contract.

``stop()`` is now one call into the motion controller: it cancels every
registered job (navigate, dock, approach, test mode, closed-loop drive),
clears the published path, and stops the motors. Jobs begun after the stop
belong to the new generation and start un-cancelled, so a stop never blocks
the next motion command. The navigate listener must still accept the
message router's single-``None`` cancel call (a signature mismatch here used
to be swallowed by the router's exception handler).
"""

import json
import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.navigator import Navigator
from pycozmo.web.msg_router import MsgRouter


def _make_nav(x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    """Navigator with a mock CLI (controllable pose) and mock world map."""
    nav: Any = Navigator(DockState(), world_map=MagicMock())
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=yaw)),
    )
    cli = MagicMock()
    type(cli).pose = property(lambda self: pose)
    nav.cli = cli
    return nav, cli


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll until predicate() is True (or the deadline passes)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestStop:
    def test_stop_cancels_running_job_and_stops_motors(self):
        nav, cli = _make_nav()

        with nav.motion.begin(Navigator.JOB_NAVIGATE) as job:
            nav.stop()
            assert job.cancelled

        cli.stop_all_motors.assert_called()
        nav.world_map.clear_astar_path.assert_called_once()

    def test_stop_without_world_map_or_cli_is_safe(self):
        nav = Navigator(DockState())  # cli=None, world_map=None
        nav.stop()  # must not raise

    def test_job_begun_after_stop_is_not_cancelled(self):
        """Re-arm semantics: stop() only cancels what was already running."""
        nav, _cli = _make_nav()
        nav.stop()

        with nav.motion.begin(Navigator.JOB_DRIVE) as job:
            assert not job.cancelled

    def test_stop_turns_off_test_mode(self):
        nav, _cli = _make_nav()
        nav.world_map.get_cubes.return_value = {}  # idle loop, no cubes
        nav.set_test_mode(True)
        assert Navigator.JOB_TEST_MODE in nav.motion.active_names

        nav.stop()

        assert _wait_until(
            lambda: Navigator.JOB_TEST_MODE not in nav.motion.active_names
        ), "test-mode job kept running after stop()"

    def test_stop_interrupts_active_closed_loop_drive(self):
        nav, cli = _make_nav()  # pose at origin, target 1000 mm away
        result: dict[str, bool] = {}

        def _drive():
            result["reached"] = nav.drive_to_point(1000.0, 0.0, timeout_s=10.0)

        thread = threading.Thread(target=_drive)
        thread.start()

        # Wait until the loop is actually commanding wheels, then stop it.
        assert _wait_until(lambda: cli.drive_wheels.called), "drive never started"

        nav.stop()
        thread.join(timeout=3.0)

        assert not thread.is_alive(), "drive did not abort after stop()"
        assert result["reached"] is False

    def test_drive_entry_rearms_after_stop(self):
        nav, cli = _make_nav()
        nav.stop()

        # A new drive call opens a fresh job, so the earlier stop cannot
        # block it (the pose never reaches the target → it times out).
        reached = nav.drive_to_point(1000.0, 0.0, timeout_s=0.1)

        assert reached is False
        assert cli.drive_wheels.called, "drive never ran after stop()"

    def test_stop_cancels_docking(self):
        """Docking used to run on: stop() never reached the dock thread."""
        nav, cli = _make_nav()
        nav.ds.dock_pose = cli.pose

        nav.dock()
        assert _wait_until(lambda: Navigator.JOB_DOCK in nav.motion.active_names), (
            "dock job never registered"
        )

        nav.stop()

        assert _wait_until(lambda: Navigator.JOB_DOCK not in nav.motion.active_names), (
            "dock job kept running after stop()"
        )


class TestNavigateCancel:
    def test_cancel_accepts_single_none_argument(self):
        """The message router calls the listener as ``listener(None)``."""
        nav, cli = _make_nav()

        with nav.motion.begin(Navigator.JOB_NAVIGATE) as job:
            nav.navigate_to(None)  # must not raise TypeError
            assert job.cancelled

        cli.stop_all_motors.assert_called()
        nav.world_map.clear_astar_path.assert_called_once()

    def test_cancel_leaves_other_jobs_running(self):
        """Navigate-cancel stops navigation, not the whole robot."""
        nav, cli = _make_nav()
        with nav.motion.begin(Navigator.JOB_TEST_MODE) as other:
            with nav.motion.begin(Navigator.JOB_NAVIGATE) as nav_job:
                nav.navigate_to(None)
                assert nav_job.cancelled
            assert not other.cancelled
        cli.stop_all_motors.assert_called()

    def test_navigate_without_robot_warns_and_returns(self):
        nav = Navigator(DockState())  # no cli
        nav.navigate_to(100.0, 50.0)  # must not raise

    def test_cancel_stops_the_running_navigate_job(self):
        nav, cli = _make_nav()
        cli.go_to_pose.side_effect = lambda *a, **k: time.sleep(0.5) or True

        nav.navigate_to(100.0, 0.0)
        assert _wait_until(lambda: Navigator.JOB_NAVIGATE in nav.motion.active_names), (
            "navigate job never registered"
        )

        nav.navigate_to(None)

        assert _wait_until(
            lambda: Navigator.JOB_NAVIGATE not in nav.motion.active_names
        ), "navigate job kept running after navigate-cancel"

    def test_router_cancel_reaches_navigator(self):
        """End-to-end: navigate_cancel message stops the motors.

        Before the cancel signature was fixed the TypeError was swallowed
        by the router and the motors kept running.
        """
        nav, cli = _make_nav()
        router = MsgRouter(world_map=MagicMock())
        router.set_actions(navigate=nav.navigate_to)

        router.route(json.dumps({"type": "navigate_cancel"}))

        cli.stop_all_motors.assert_called()

    def test_router_navigate_reaches_navigator(self):
        nav, cli = _make_nav()
        router = MsgRouter(world_map=MagicMock())
        router.set_actions(navigate=nav.navigate_to)

        # Robot at origin; scene (x=200, z=0) maps to world (0, 200, 0),
        # 200 mm to the robot's left -> spawns the nav thread, which issues
        # go_to_pose on the mock CLI.
        router.route(json.dumps({"type": "navigate_to", "x": 200.0, "z": 0.0}))
        assert _wait_until(lambda: cli.go_to_pose.called)
