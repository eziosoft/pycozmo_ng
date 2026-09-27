"""Tests for Navigator.stop() and the browser navigate-cancel contract.

``stop()`` composes the library's cancel primitives: it sets the drive
interrupt latch, cancels A* path following, exits test mode, clears the
published path, and stops the motors. The navigate listener must accept
the message router's single-``None`` cancel call (a signature mismatch
here used to be swallowed by the router's exception handler).
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


class TestStop:
    def test_stop_cancels_path_and_stops_motors(self):
        nav, cli = _make_nav()
        nav._path_following = True

        nav.stop()

        assert nav._stop_requested.is_set()
        assert nav._path_following is False
        cli.stop_all_motors.assert_called()
        nav.world_map.clear_astar_path.assert_called_once()

    def test_stop_without_world_map_or_cli_is_safe(self):
        nav = Navigator(DockState())  # cli=None, world_map=None
        nav.stop()
        assert nav._stop_requested.is_set()

    def test_stop_turns_off_test_mode(self):
        nav, _cli = _make_nav()
        nav._test_mode = True  # simulate an active test-mode loop

        nav.stop()

        assert nav._test_mode is False

    def test_stop_interrupts_active_closed_loop_drive(self):
        nav, cli = _make_nav()  # pose at origin, target 1000 mm away
        result: dict[str, bool] = {}

        def _drive():
            result["reached"] = nav.drive_to_point(1000.0, 0.0, timeout_s=10.0)

        thread = threading.Thread(target=_drive)
        thread.start()

        # Wait until the loop is actually commanding wheels, then stop it.
        deadline = time.monotonic() + 2.0
        while not cli.drive_wheels.called and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cli.drive_wheels.called, "drive loop never started"

        nav.stop()
        thread.join(timeout=3.0)

        assert not thread.is_alive(), "drive did not abort after stop()"
        assert result["reached"] is False

    def test_drive_entry_rearms_after_stop(self):
        nav, _cli = _make_nav()
        nav.stop()
        assert nav._stop_requested.is_set()

        # A new drive call clears the latch on entry (pose never reaches
        # the target, so it times out and returns False).
        reached = nav.drive_to_point(1000.0, 0.0, timeout_s=0.1)

        assert reached is False
        assert not nav._stop_requested.is_set()


class TestNavigateCancel:
    def test_cancel_accepts_single_none_argument(self):
        """The message router calls the listener as ``listener(None)``."""
        nav, cli = _make_nav()
        nav._path_following = True

        nav.navigate_to(None)  # must not raise TypeError

        cli.stop_all_motors.assert_called()
        assert nav._path_following is False
        nav.world_map.clear_astar_path.assert_called_once()

    def test_navigate_without_robot_warns_and_returns(self):
        nav = Navigator(DockState())  # no cli
        nav.navigate_to(100.0, 50.0)  # must not raise

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
        deadline = time.monotonic() + 2.0
        while not cli.go_to_pose.called and time.monotonic() < deadline:
            time.sleep(0.01)
        assert cli.go_to_pose.called
