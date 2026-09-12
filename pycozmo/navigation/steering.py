"""Point-to-point differential-drive steering.

Pure functions with no robot I/O — the :class:`Navigator` calls them every
control tick (≈20 Hz) with the current robot pose and the desired target. The
closed loop lives here so it can be unit-tested against a kinematic unicycle
model (``pycozmo/tests/test_steering.py``).

Conventions (world frame, Anki): X forward, Y left, yaw about +Z (CCW
positive). Wheel speeds are body-relative mm/s, forward positive. A positive
angular component ``omega`` turns the robot CCW (left); it is applied as
``left = v - omega, right = v + omega`` so ``right > left`` veers left.
"""

import math

__all__ = [
    "steer_velocities",
    "wrap_pi",
]


def wrap_pi(rad: float) -> float:
    """Wrap an angle into [-pi, pi)."""
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def steer_velocities(
    x: float,
    y: float,
    yaw: float,
    tx: float,
    ty: float,
    speed_mmps: float = 60.0,
    heading_gain: float = 2.2,
    turn_max_mmps: float = 80.0,
    slow_dist_mm: float = 60.0,
    min_speed_frac: float = 0.15,
    reverse_angle_rad: float = math.pi / 2.0,
) -> tuple[float, float]:
    """Compute wheel speeds driving the robot toward ``(tx, ty)``.

    Re-aims at the *current* robot pose every call: ``yaw`` is the current
    heading and the target bearing is recomputed from ``(x, y)``, so the
    robot converges onto the target instead of chasing a stale line-of-sight.

    Args:
        x, y, yaw: Current robot pose (mm, world frame; yaw in radians).
        tx, ty: Target position (mm, world frame).
        speed_mmps: Cruise wheel speed (mm/s) — ramps down near the target.
        heading_gain: Proportional gain mapping heading error (rad) to a
            [-1, 1] turn signal.
        turn_max_mmps: Maximum differential wheel speed (mm/s), so the turn
            signal is ``[-turn_max, turn_max]``.
        slow_dist_mm: Distance (mm) below which linear speed ramps down.
        min_speed_frac: Linear speed floor as a fraction of ``speed_mmps``.
        reverse_angle_rad: Below this heading error the robot drives
            forward; beyond it, it backs up while curving toward the target.

    Returns:
        ``(left_speed, right_speed)`` in mm/s.
    """
    ex = tx - x
    ey = ty - y
    d = math.hypot(ex, ey)
    if d < 1e-6:
        bearing = yaw
    else:
        bearing = math.atan2(ey, ex)

    err = wrap_pi(bearing - yaw)

    speed = speed_mmps * min(1.0, max(min_speed_frac, d / slow_dist_mm))
    reverse = abs(err) > reverse_angle_rad
    if reverse:
        # Steer so the robot's rear points at the target.
        err = wrap_pi(err - math.pi)
        speed = -speed

    omega = max(-1.0, min(1.0, err * heading_gain)) * turn_max_mmps
    return speed - omega, speed + omega
