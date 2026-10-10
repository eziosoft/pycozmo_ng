"""Tests for Navigator relative motion (COZMO-69): move_by, turn_by, stall.

The robot used through these tests is a kinematic double: its pose only
changes when ``drive_wheels`` is called, so a command the firmware would
ignore (``stuck=True`` — the picked-up / blocked case) leaves the pose
frozen and the loop has to notice on its own.
"""

import math
import time
from types import SimpleNamespace

from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.navigator import Navigator


class FakeRobot:
    """Minimal client double — unicycle kinematics driven by wheel commands."""

    def __init__(self, stuck: bool = False):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.stuck = stuck

    @property
    def pose(self):
        return SimpleNamespace(
            position=SimpleNamespace(x=self.x, y=self.y),
            rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=self.yaw)),
        )

    def drive_wheels(self, lwheel_speed: float = 0.0, rwheel_speed: float = 0.0, **kwargs) -> None:
        del kwargs
        if self.stuck:
            return
        dt = 0.03
        linear = (lwheel_speed + rwheel_speed) / 2.0
        angular = (rwheel_speed - lwheel_speed) / 90.0  # ~90 mm track
        self.yaw += angular * dt
        self.x += linear * math.cos(self.yaw) * dt
        self.y += linear * math.sin(self.yaw) * dt

    def stop_all_motors(self) -> None:
        pass


def _navigator(robot: FakeRobot | None = None) -> Navigator:
    nav = Navigator(DockState())
    if robot is not None:
        nav.cli = robot
    return nav


class TestTurnBy:
    def test_without_robot_returns_false(self):
        assert _navigator().turn_by(math.radians(90.0)) is False

    def test_turns_the_requested_angle(self):
        robot = FakeRobot()
        assert _navigator(robot).turn_by(math.radians(30.0)) is True
        assert abs(math.degrees(robot.yaw) - 30.0) <= math.degrees(0.06)

    def test_stalls_out_fast_when_wheels_are_ignored(self):
        """A held or blocked robot never moves — fail in ~stall_s, not at timeout."""
        robot = FakeRobot(stuck=True)
        started = time.monotonic()
        aligned = _navigator(robot).turn_by(math.radians(90.0), timeout_s=30.0, stall_s=0.2)
        elapsed = time.monotonic() - started
        assert aligned is False
        assert elapsed < 3.0


class TestMoveBy:
    def test_without_robot_returns_false(self):
        assert _navigator().move_by(100.0) is False

    def test_drives_the_requested_distance(self):
        robot = FakeRobot()
        assert _navigator(robot).move_by(100.0, speed_mmps=60.0) is True
        assert math.hypot(robot.x, robot.y) >= 85.0  # within the arrival radius

    def test_reverse_moves_backwards(self):
        robot = FakeRobot()
        assert _navigator(robot).move_by(-100.0, speed_mmps=60.0) is True
        assert robot.x <= -85.0

    def test_stalls_out_fast_when_wheels_are_ignored(self):
        robot = FakeRobot(stuck=True)
        started = time.monotonic()
        arrived = _navigator(robot).move_by(500.0, timeout_s=30.0, stall_s=0.2)
        elapsed = time.monotonic() - started
        assert arrived is False
        assert elapsed < 3.0
