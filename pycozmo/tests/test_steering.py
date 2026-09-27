"""Tests for the closed-loop steering controller (pycozmo/navigation/steering.py).

The controller is validated against a kinematic unicycle model: wheel speeds
produce forward speed ``(vl + vr) / 2`` and yaw rate ``(vr - vl) / track``,
integrated at the same 20 Hz tick used by ``Navigator.drive_to_point``.
"""

import math

import pytest

from pycozmo.navigation.steering import steer_velocities, wrap_pi

# Simulation-only model constants (not used by the controller).
TRACK_W_MM = 60.0
DT_S = 0.05
STOP_MM = 20.0
TIMEOUT_S = 15.0
ALIGN_TOL = 0.06


def _advance(x, y, yaw, vl, vr):
    """Integrate one tick of the unicycle model."""
    forward = (vl + vr) / 2.0
    yaw_rate = (vr - vl) / TRACK_W_MM  # + => CCW (left)
    yaw += yaw_rate * DT_S
    x += forward * math.cos(yaw) * DT_S
    y += forward * math.sin(yaw) * DT_S
    return x, y, yaw


def _simulate(
    x0, y0, yaw0, tx, ty, *, align_to=None, stop_mm=STOP_MM, timeout_s=TIMEOUT_S
):
    """Mirror ``Navigator.drive_to_point``: drive closed-loop to the target,
    then turn in place to ``align_to`` (if given). Returns
    ``(arrived, t_s, end_pose, min_dist)``."""
    x, y, yaw = x0, y0, yaw0
    t = 0.0
    min_d = math.inf
    reached = False
    while t <= timeout_s:
        d = math.hypot(tx - x, ty - y)
        min_d = min(min_d, d)
        if d <= stop_mm:
            reached = True
            break
        vl, vr = steer_velocities(x, y, yaw, tx, ty)
        x, y, yaw = _advance(x, y, yaw, vl, vr)
        t += DT_S

    if reached and align_to is not None:
        while t <= timeout_s:
            err = wrap_pi(align_to - yaw)
            if abs(err) < ALIGN_TOL:
                break
            omega = max(-1.0, min(1.0, err * 2.2)) * 45.0
            x, y, yaw = _advance(x, y, yaw, -omega, omega)
            t += DT_S

    return reached, t, (x, y, yaw), min_d


class TestWrapPi:
    def test_wraps_to_pi(self):
        assert wrap_pi(0.0) == pytest.approx(0.0)
        assert wrap_pi(math.pi) == pytest.approx(-math.pi, abs=1e-9)
        assert wrap_pi(-math.pi) == pytest.approx(-math.pi, abs=1e-9)
        assert wrap_pi(3.0 * math.pi / 2.0) == pytest.approx(-math.pi / 2.0)
        assert wrap_pi(-3.0 * math.pi / 2.0) == pytest.approx(math.pi / 2.0)


class TestSteerVelocitiesSign:
    def test_target_ahead_drives_straight(self):
        vl, vr = steer_velocities(0, 0, 0.0, 100.0, 0.0)
        assert vl == pytest.approx(vr)
        assert vl > 0

    def test_target_left_turns_left(self):
        """Target on the +Y (left) side must make the right wheel faster."""
        vl, vr = steer_velocities(0, 0, 0.0, 0.0, 100.0)
        assert vr > vl  # vr > vl => CCW (left) in the unicycle model

    def test_target_right_turns_right(self):
        vl, vr = steer_velocities(0, 0, 0.0, 0.0, -100.0)
        assert vl > vr

    def test_target_behind_reverses(self):
        vl, vr = steer_velocities(0, 0, 0.0, -100.0, 0.0)
        assert vl < 0
        assert vr < 0

    def test_slowdown_near_target(self):
        slow = steer_velocities(0, 0, 0.0, 10.0, 0.0)
        fast = steer_velocities(0, 0, 0.0, 1000.0, 0.0)
        assert abs(slow[0]) < abs(fast[0])
        assert slow[0] > 0  # still driving forward, not stalled


class TestSteeringConvergence:
    """Closed-loop drives must reach the target from tricky start poses."""

    @pytest.mark.parametrize(
        ("x0", "y0", "yaw0", "tx", "ty"),
        [
            (0.0, 0.0, 0.0, 100.0, 0.0),  # straight ahead
            (0.0, 0.0, 0.0, 60.0, 40.0),  # ahead + right (previously broke)
            (0.0, 0.0, 0.0, 50.0, 80.0),  # off the initial line-of-sight
            (0.0, 0.0, 0.0, -60.0, 0.0),  # directly behind (reverse)
            (0.0, 0.0, math.pi, 40.0, -30.0),  # facing away (reverse)
            (0.0, 0.0, 0.0, 0.0, 70.0),  # to the side
            (20.0, 5.0, 0.3, 90.0, 80.0),  # far diagonal
            (0.0, 0.0, 0.0, -10.0, 40.0),  # close behind-left
        ],
    )
    def test_reaches_target(self, x0, y0, yaw0, tx, ty):
        arrived, t, _, _ = _simulate(x0, y0, yaw0, tx, ty)
        assert arrived, f"did not reach target from ({x0},{y0}) yaw={yaw0}"
        assert t < TIMEOUT_S

    def test_arrives_aligned_when_requested(self):
        # Target NE, final heading = face back toward the origin.
        align_to = math.atan2(-40, -60)
        arrived, _, end, _ = _simulate(0, 0, 0.0, 60.0, 40.0, align_to=align_to)
        assert arrived
        _, _, yaw = end
        assert abs(wrap_pi(align_to - yaw)) < ALIGN_TOL

    def test_monotonic_approach_no_overshoot_runaway(self):
        """Distance to target must shrink to the stop radius, not blow past."""
        _, _, _, min_d = _simulate(0, 0, 0.0, 50.0, 80.0)
        assert min_d < STOP_MM + 1.0
