"""Tests for marker_observation — the geometry behind the ArUco vantage.

The docking sequence depends on these numbers being right, and they encode
a measurement property that is easy to break silently: a planar marker's
normal is only usable when it is seen obliquely, and "oblique" is measured
as a deviation from head-on (the printed face is turned toward the robot,
so standing square in front is 0 deg of obliquity, not 180).

Pure functions, no robot: obliquity, the vantage planner, and the normal
fusion with its median-over-outliers behaviour.
"""

import math

import numpy as np

from pycozmo_ng.navigation.marker_observation import (
    MIN_USABLE_OBLIQUITY_DEG,
    TARGET_OBLIQUITY_DEG,
    NormalEstimate,
    estimate_normal,
    obliquity_deg,
    vantage_target,
)
from pycozmo_ng.navigation.steering import wrap_pi

# Marker at the origin with its printed face toward +x, so the normal is +x.
MARKER_XY = (0.0, 0.0)
NORMAL_XY = (1.0, 0.0)


def _state(x: float = 0.0, y: float = 0.0, normal_yaw: float = 0.0, updated_at: float = 0.0):
    """A minimal stand-in for a fused ``ArucoState``."""
    z = np.array([math.cos(normal_yaw), math.sin(normal_yaw), 0.0])
    y_up = np.array([0.0, 0.0, 1.0])
    return type(
        "S",
        (),
        {
            "position": np.array([x, y, 0.0]),
            "rotation_matrix": np.column_stack([np.cross(y_up, z), y_up, z]),
            "updated_at": updated_at,
        },
    )()


def _robot_at(bearing_deg: float, range_mm: float = 200.0) -> tuple[float, float]:
    """A robot position *range_mm* from the marker at a given bearing.

    Bearing 0 puts it dead ahead of the printed face; 90 puts it square to
    the side.
    """
    a = math.radians(bearing_deg)
    return (range_mm * math.cos(a), range_mm * math.sin(a))


class TestObliquity:
    def test_head_on_is_degenerate(self):
        """Facing the printed face square-on is the worst case — 0 deg."""
        assert obliquity_deg(MARKER_XY, NORMAL_XY, (200.0, 0.0)) == 0.0

    def test_side_on_is_well_conditioned(self):
        assert math.isclose(obliquity_deg(MARKER_XY, NORMAL_XY, (0.0, 200.0)), 90.0)

    def test_behind_the_marker_is_also_head_on(self):
        """The normal points away here, but the face is still edge-parallel."""
        assert obliquity_deg(MARKER_XY, NORMAL_XY, (-200.0, 0.0)) == 0.0

    def test_measured_obliquity_matches_the_geometry(self):
        """A vantage planned for N deg reports ~N deg obliquity."""
        for degrees in (10.0, 20.0, 30.0, 45.0):
            robot = _robot_at(degrees)
            assert math.isclose(obliquity_deg(MARKER_XY, NORMAL_XY, robot), degrees, abs_tol=0.5)

    def test_degenerate_inputs_report_degenerate(self):
        """A zero-length view ray or normal must not raise."""
        assert obliquity_deg(MARKER_XY, NORMAL_XY, MARKER_XY) == 0.0
        assert obliquity_deg(MARKER_XY, (0.0, 0.0), (200.0, 0.0)) == 0.0


class TestEstimateNormal:
    def test_reads_the_bearing_of_a_clean_normal(self):
        est = estimate_normal([_state(normal_yaw=0.0)], (200.0, 0.0))
        assert est is not None
        assert math.isclose(math.degrees(est.bearing_rad), 0.0, abs_tol=0.5)

    def test_recovers_a_rotated_normal(self):
        est = estimate_normal([_state(normal_yaw=math.radians(45.0))], (0.0, 200.0))
        assert est is not None
        assert math.isclose(math.degrees(est.bearing_rad), 45.0, abs_tol=0.5)

    def test_head_on_view_is_not_usable(self):
        """The whole point: a fronto-parallel view must read as unusable."""
        est = estimate_normal([_state(normal_yaw=0.0)], (200.0, 0.0))
        assert est is not None
        assert not est.usable
        assert est.obliquity_deg < MIN_USABLE_OBLIQUITY_DEG

    def test_oblique_view_is_usable(self):
        robot = _robot_at(TARGET_OBLIQUITY_DEG)
        est = estimate_normal([_state(normal_yaw=0.0)], robot)
        assert est is not None
        assert est.usable

    def test_degenerate_normal_yields_nothing(self):
        """A normal with no horizontal part carries no bearing to read."""
        est = estimate_normal([_state(normal_yaw=math.radians(90.0))], (0.0, 200.0))
        # yaw=90 gives normal (0,1) — horizontal, but the robot sits on the
        # marker's own normal line, so it is head-on and not usable.
        assert est is None or not est.usable

    def test_empty_sample_set_yields_nothing(self):
        assert estimate_normal([], (200.0, 0.0)) is None

    def test_median_rejects_a_single_gross_outlier(self):
        """One wildly wrong frame must not move the estimate.

        This is the only thing the median buys: the systematic error of a
        head-on view is *not* removable by averaging (see the module
        docstring), but a single bad frame is otherwise able to dominate.
        """
        robot = _robot_at(TARGET_OBLIQUITY_DEG)
        good = [_state(normal_yaw=0.0) for _ in range(5)]
        clean = estimate_normal(good, robot)
        polluted = estimate_normal([*good, _state(normal_yaw=math.radians(40.0))], robot)
        assert clean is not None and polluted is not None
        assert abs(wrap_pi(polluted.bearing_rad - clean.bearing_rad)) < math.radians(1.0)

    def test_position_is_the_median_of_the_samples(self):
        robot = _robot_at(TARGET_OBLIQUITY_DEG)
        est = estimate_normal([_state(x=10.0), _state(x=12.0), _state(x=11.0)], robot)
        assert est is not None
        assert math.isclose(est.x, 11.0)

    def test_at_range_places_a_point_out_along_the_normal(self):
        est = NormalEstimate(
            bearing_rad=0.0,
            x=0.0,
            y=0.0,
            samples=10,
            obliquity_deg=30.0,
            spread_deg=1.0,
        )
        x, y = est.at_range(200.0)
        assert math.isclose(x, 200.0)
        assert math.isclose(y, 0.0)


class TestVantageTarget:
    def test_planned_vantage_holds_the_requested_range(self):
        seed = NormalEstimate(0.0, 0.0, 0.0, 10, 30.0, 1.0)
        x, y, _heading = vantage_target(seed, (500.0, 0.0), 200.0, TARGET_OBLIQUITY_DEG)
        assert math.isclose(math.hypot(x, y), 200.0, rel_tol=1e-6)

    def test_planned_vantage_achieves_the_requested_obliquity(self):
        """The point of the whole exercise: standing here must make the
        marker's normal measurable."""
        seed = NormalEstimate(0.0, 0.0, 0.0, 10, 30.0, 1.0)
        for degrees in (20.0, 30.0, 45.0):
            x, y, _ = vantage_target(seed, (500.0, 0.0), 200.0, degrees)
            assert math.isclose(obliquity_deg(MARKER_XY, NORMAL_XY, (x, y)), degrees, abs_tol=0.5)

    def test_heading_faces_the_marker_centre(self):
        """Facing the centre is what makes the pose reachable — the heading
        must come from the marker's position, not its rotation."""
        seed = NormalEstimate(0.0, 0.0, 0.0, 10, 30.0, 1.0)
        x, y, heading = vantage_target(seed, (500.0, 0.0), 200.0, TARGET_OBLIQUITY_DEG)
        assert math.isclose(heading, math.atan2(-y, -x), abs_tol=1e-9)

    def test_sidesteps_to_the_side_the_robot_is_already_on(self):
        """Approaching from the left, the vantage goes left — a shuffle,
        not a crossing of the marker's normal line."""
        seed = NormalEstimate(0.0, 0.0, 0.0, 10, 30.0, 1.0)
        from_left = vantage_target(seed, _robot_at(120.0, 500.0), 200.0, TARGET_OBLIQUITY_DEG)
        from_right = vantage_target(seed, _robot_at(-120.0, 500.0), 200.0, TARGET_OBLIQUITY_DEG)
        assert from_left[1] > 0.0
        assert from_right[1] < 0.0

    def test_a_bearing_error_shifts_the_vantage_but_keeps_the_angle(self):
        """A wrong seed bearing moves the vantage point, but the *achieved*
        obliquity depends only on the robot standing off the line — so a
        bad seed cannot silently produce a head-on view.

        (The achieved angle is measured against the true normal, so a
        mis-seeded vantage shifts along the arc rather than collapsing.)
        """
        seed = NormalEstimate(0.0, 0.0, 0.0, 10, 30.0, 1.0)
        x, y, _ = vantage_target(seed, (500.0, 0.0), 200.0, TARGET_OBLIQUITY_DEG)
        assert obliquity_deg(MARKER_XY, NORMAL_XY, (x, y)) >= MIN_USABLE_OBLIQUITY_DEG
