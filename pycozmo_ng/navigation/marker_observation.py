"""MarkerObservation — measuring a planar marker's normal without the degeneracy.

A planar marker's normal is the worst-conditioned degree of freedom in its
six-DoF pose. Viewed *fronto-parallel* — the robot facing the marker, the
case docking creates deliberately — the marker's face is parallel to the
image plane, the perspective cues that pin its tilt collapse, and the
horizontal component of the recovered normal shrinks to ``sin(obliquity)``:
about 0.03 at 2 deg of obliquity. Docking code that wants a *bearing* from
that component normalises it by exactly this vanishing number, so a
sub-degree tilt error becomes tens of degrees of bearing error, which the
standoff distance then multiplies into tens of millimetres of lateral error
at the charger.

Measured on a 22.5 mm marker at a 200 mm standoff (QVGA, ~230 px focal,
0.25 px corner noise), median error in the normal's horizontal bearing:

    ==========  ==================  ==================
    obliquity   horizontal norm    bearing error
    ==========  ==================  ==================
      2 deg          0.035             43.5 deg
     10 deg          0.174             17.3 deg
     15 deg          0.259              7.8 deg
     20 deg          0.342              4.4 deg
     30 deg          0.500              2.0 deg
     40 deg          0.643              1.0 deg
    ==========  ==================  ==================

``TARGET_OBLIQUITY_DEG`` sits at 15 deg — lower than the accuracy column
would like — because at 30 deg the marker can leave the camera frame on a
real robot, and an unseen marker fails harder than a bearing error. The
choice is a visibility/accuracy trade, not a free parameter; see the
constant's docstring.

Two properties of that table drive the design:

* **Obliquity is the only real lever.** The residual error is systematic —
  the bias of an ill-conditioned solve — so averaging frames or reading from
  two symmetric vantages both converge to the same offset and gain nothing.
  Only the viewing angle moves the number.
* **Samples must still be aggregated.** Median (not mean) over the dwell,
  purely to drop the occasional gross outlier; it is not an accuracy fix.

So this module does two things and nothing else: it reports how oblique a
view is (:func:`obliquity_deg`), and it plans the vantage that makes a view
oblique (:func:`vantage_target`). Everything downstream stays in plain
world-frame geometry.

The standoff point and the reverse heading deliberately do **not** use the
normal at all — see :mod:`pycozmo_ng.navigation.aruco_docking`. Only the
*aim* needs the normal, and it is measured once, obliquely.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pycozmo_ng.navigation.steering import wrap_pi

__all__ = [
    "MAX_SPREAD_DEG",
    "MIN_USABLE_OBLIQUITY_DEG",
    "TARGET_OBLIQUITY_DEG",
    "NormalEstimate",
    "estimate_normal",
    "obliquity_deg",
    "vantage_target",
]

#: Obliquity the vantage is planned for. 15 deg, not the 30 deg the
#: accuracy table would prefer: at that angle the marker can leave the
#: camera frame entirely on a real robot, and an unseen marker is a far
#: worse outcome than a bearing error. 15 deg still buys most of the
#: conditioning (bearing error ~7.8 deg vs ~43 deg head-on) while
#: keeping the marker comfortably in view.
#:
#: If the robot's camera turns out to have a wider field of view than
#: assumed, raise this — it is a straight accuracy/visibility trade.
TARGET_OBLIQUITY_DEG: float = 15.0

#: Below this, the bearing is not trustworthy enough to aim a reverse from.
#: 12 deg is ~9 deg of bearing error. Under this the caller should
#: re-vantage or accept a degraded aim, not dock blind. Set just below
#: ``TARGET_OBLIQUITY_DEG`` so a vantage that lands on target passes, with
#: a little margin for arrival slop.
MIN_USABLE_OBLIQUITY_DEG: float = 12.0

#: Sanity bound on sample scatter. The per-view bias is systematic, so a wide
#: spread means the marker was poorly tracked (motion blur, partial view) —
#: a different failure from the bias, and worth surfacing.
MAX_SPREAD_DEG: float = 12.0


@dataclass(frozen=True)
class NormalEstimate:
    """A marker's normal and position, measured across one dwell.

    Attributes:
        bearing_rad: Horizontal bearing of the marker's normal, pointing at
            the robot's side of the marker (toward the vantage).
        x: Median marker x over the sampled frames (mm, world frame).
        y: Median marker y over the sampled frames (mm, world frame).
        samples: How many marker states were aggregated.
        obliquity_deg: Angle between the view ray and the marker's normal at
            the sampled pose — the conditioning of this measurement.
        spread_deg: Median absolute deviation of the per-sample bearings.
            Outlier screen, not an accuracy figure (see the module docstring).
    """

    bearing_rad: float
    x: float
    y: float
    samples: int
    obliquity_deg: float
    spread_deg: float

    @property
    def usable(self) -> bool:
        """True when this view was oblique enough to aim a reverse from."""
        return self.obliquity_deg >= MIN_USABLE_OBLIQUITY_DEG

    def unit(self) -> tuple[float, float]:
        """The normal as a unit XY vector."""
        return math.cos(self.bearing_rad), math.sin(self.bearing_rad)

    def at_range(self, range_mm: float) -> tuple[float, float]:
        """A point *range_mm* out along the normal — the standoff point."""
        ux, uy = self.unit()
        return self.x + ux * range_mm, self.y + uy * range_mm

    def describe(self) -> str:
        """One-line summary for the dock log."""
        return (
            f"normal {math.degrees(self.bearing_rad):.1f} deg, "
            f"{self.samples} samples, obliquity {self.obliquity_deg:.1f} deg, "
            f"spread {self.spread_deg:.1f} deg"
        )


def obliquity_deg(
    marker_xy: tuple[float, float],
    normal_xy: tuple[float, float],
    robot_xy: tuple[float, float],
) -> float:
    """How far the robot's view of the marker is from head-on, in degrees.

    The marker's printed face is turned toward the robot, so the normal
    points *back* at it: standing square in front, the angle between the
    view ray and the normal is 180 deg, and that — not 0 deg — is the
    degenerate case. This folds that angle, so the result is the deviation
    from head-on: 0 deg is degenerate (fronto-parallel), 90 deg is a fully
    grazing view and the best-conditioned.

    This is the conditioning of the measurement, and it is what
    :attr:`NormalEstimate.usable` is judged on.

    Args:
        marker_xy: Marker position (mm, world frame).
        normal_xy: Marker normal as an XY vector (scale irrelevant).
        robot_xy: Robot position (mm, world frame).

    Returns:
        The folded angle in degrees, always in [0, 90].
    """
    rx = marker_xy[0] - robot_xy[0]
    ry = marker_xy[1] - robot_xy[1]
    rlen = math.hypot(rx, ry)
    nlen = math.hypot(normal_xy[0], normal_xy[1])
    if rlen < 1e-6 or nlen < 1e-9:
        return 0.0
    cosang = (rx * normal_xy[0] + ry * normal_xy[1]) / (rlen * nlen)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    return min(angle, 180.0 - angle)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _median_bearing(angles: Sequence[float]) -> float:
    """Median of angles on the circle, unwrapped about the first sample."""
    ref = angles[0]
    return wrap_pi(_median([ref + wrap_pi(a - ref) for a in angles]))


def _normal_xy(state) -> tuple[float, float] | None:
    """XY part of a fused marker's normal, or None when it is degenerate.

    A near-zero horizontal component is exactly the fronto-parallel case
    this module exists to detect, so it is reported as "no usable normal"
    rather than normalised into noise.
    """
    rot = state.rotation_matrix
    nx = float(rot[0, 2])
    ny = float(rot[1, 2])
    length = math.hypot(nx, ny)
    if length < 1e-6:
        return None
    return nx, ny


def estimate_normal(states: Sequence, robot_xy: tuple[float, float]) -> NormalEstimate | None:
    """Fuse a dwell's marker states into one normal + position estimate.

    Aggregates with a median rather than a mean purely to reject gross
    outliers. This does **not** reduce the systematic bearing error of a
    poorly-conditioned view — only obliquity does that — so ``obliquity_deg``
    on the result is the number to branch on.

    Args:
        states: Fused marker states sampled across the dwell. Each needs
            ``position`` and ``rotation_matrix``.
        robot_xy: Robot position at the dwell (mm, world frame).

    Returns:
        The :class:`NormalEstimate`, or None when no sample carried a usable
        normal (fronto-parallel throughout, or the marker was never seen).
    """
    bearings: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    normals: list[tuple[float, float]] = []
    for state in states:
        normal = _normal_xy(state)
        if normal is None:
            continue
        # The printed face is turned toward the robot, so the normal points
        # from the marker back at the vantage: negate the marker-to-robot
        # bearing to land on the marker's outward face.
        to_robot = math.atan2(robot_xy[1] - float(state.position[1]), robot_xy[0] - float(state.position[0]))
        nx, ny = normal
        # Normal is recovered mod 180 deg for a square marker; orient it to
        # agree with the geometric guess (marker -> robot) before averaging.
        if math.cos(math.atan2(ny, nx) - to_robot) < 0.0:
            nx, ny = -nx, -ny
        bearings.append(math.atan2(ny, nx))
        normals.append((nx, ny))
        xs.append(float(state.position[0]))
        ys.append(float(state.position[1]))
    if not bearings:
        return None

    med_bearing = _median_bearing(bearings)
    spread = _median([abs(wrap_pi(b - med_bearing)) for b in bearings])
    # Condition on the median sample so a single bad frame cannot claim a
    # well-conditioned view.
    mx, my = _median(xs), _median(ys)
    obliq = obliquity_deg((mx, my), (math.cos(med_bearing), math.sin(med_bearing)), robot_xy)
    return NormalEstimate(
        bearing_rad=med_bearing,
        x=mx,
        y=my,
        samples=len(bearings),
        obliquity_deg=obliq,
        spread_deg=math.degrees(spread),
    )


def vantage_target(
    estimate: NormalEstimate,
    robot_xy: tuple[float, float],
    range_mm: float,
    obliquity_deg: float = TARGET_OBLIQUITY_DEG,
) -> tuple[float, float, float]:
    """Plan a pose that views the marker at *obliquity_deg*.

    The robot moves to a point *range_mm* from the marker, offset sideways
    from the normal by the requested obliquity, facing the marker centre.
    Facing the *centre* (rather than the normal) is what makes the resulting
    pose reachable: the heading is then a well-conditioned function of the
    marker's position, which is measured ~20x better than its rotation.

    The offset is taken on whichever side the robot already sits, so the move
    is a lateral shuffle rather than a crossing of the marker's normal line.

    Args:
        estimate: The current (possibly ill-conditioned) normal estimate —
            used only to decide *which way to sidestep*. Its bearing error
            moves the vantage point but does not change the obliquity
            achieved, since the marker is still seen from the side.
        robot_xy: Robot position now (mm, world frame).
        range_mm: Distance to hold from the marker at the vantage.
        obliquity_deg: Requested viewing angle.

    Returns:
        ``(x, y, heading)`` for the vantage pose.
    """
    to_robot = math.atan2(
        robot_xy[1] - estimate.y,
        robot_xy[0] - estimate.x,
    )
    side = 1.0 if wrap_pi(to_robot - estimate.bearing_rad) >= 0.0 else -1.0
    offset = wrap_pi(estimate.bearing_rad + side * math.radians(obliquity_deg))
    x = estimate.x + math.cos(offset) * range_mm
    y = estimate.y + math.sin(offset) * range_mm
    heading = math.atan2(estimate.y - y, estimate.x - x)
    return x, y, heading
