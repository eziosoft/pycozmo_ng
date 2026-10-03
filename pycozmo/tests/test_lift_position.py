"""Tests for lift feedback conversion — the raw field is an arm angle.

``RobotState.lift_height_mm`` is misnamed: the firmware reports the lift
arm angle in radians (live: ≈ -0.2 forks down, ≈ 0.8 forks up — exactly
``MIN_LIFT_ANGLE``/``MAX_LIFT_ANGLE``). The client converts it to
millimetres through the lift geometry, which is what every consumer
(``cozmo.sensors.lift_height``, telemetry, the web HUD's LIFT row, the
carry logs) reads.
"""

import math

import pytest

from pycozmo.client import Client
from pycozmo.protocol import protocol_encoder
from pycozmo.util import robot


def test_readings_at_the_travel_limits_map_to_them():
    # The values actually observed on the robot, rounded to 1 decimal.
    down = robot.lift_position_from_raw(-0.2)
    up = robot.lift_position_from_raw(0.8)
    assert down.height.mm == pytest.approx(32.0, abs=0.5)
    assert up.height.mm == pytest.approx(92.0, abs=0.5)


def test_exact_limit_angles_map_exactly():
    down = robot.lift_position_from_raw(robot.MIN_LIFT_ANGLE.radians)
    up = robot.lift_position_from_raw(robot.MAX_LIFT_ANGLE.radians)
    assert down.height.mm == pytest.approx(32.0, abs=1e-9)
    assert up.height.mm == pytest.approx(92.0, abs=1e-9)


def test_mid_range_follows_the_lift_geometry():
    raw = 0.3
    expected = math.sin(raw) * robot.LIFT_ARM_LENGTH.mm + robot.LIFT_PIVOT_HEIGHT.mm
    assert robot.lift_position_from_raw(raw).height.mm == pytest.approx(expected)


def test_out_of_range_readings_are_clamped_to_travel():
    # Never a negative or over-tall height, whatever the firmware sends.
    assert robot.lift_position_from_raw(-50.0).height.mm == pytest.approx(
        32.0, abs=1e-9
    )
    assert robot.lift_position_from_raw(50.0).height.mm == pytest.approx(92.0, abs=1e-9)


def test_ratio_and_angle_stay_consistent():
    pos = robot.lift_position_from_raw(0.3)
    assert 0.0 < pos.ratio < 1.0
    assert pos.angle.radians == pytest.approx(0.3)
    assert robot.lift_position_from_raw(-50.0).ratio == pytest.approx(0.0)
    assert robot.lift_position_from_raw(50.0).ratio == pytest.approx(1.0)


def test_client_converts_the_robot_state_field():
    """The wiring: `_on_robot_state` must not store the raw radians."""
    cli = Client()
    cli._on_robot_state(
        None,
        protocol_encoder.RobotState(lift_height_mm=0.8, cliff_data_raw=(0, 0, 0, 0)),
    )
    assert cli.lift_position.height.mm == pytest.approx(92.0, abs=0.5)
    cli._on_robot_state(
        None,
        protocol_encoder.RobotState(lift_height_mm=-0.2, cliff_data_raw=(0, 0, 0, 0)),
    )
    assert cli.lift_position.height.mm == pytest.approx(32.0, abs=0.5)
