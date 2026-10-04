"""Tests for cube approach face selection (Navigator._pick_face_normal).

Regression for the "approaches from 45° / aligned to the corner" bug: the
cube rests on one face, so exactly one of its body axes is vertical and the
four wall normals are the ± of the *other two* (horizontal) axes. Narrowly
normalizing a near-vertical axis's horizontal noise produced arbitrary ~45°
directions.
"""

import math

import numpy as np
import pytest

from pycozmo_ng.navigation.navigator import Navigator

CUBE = np.array([100.0, 200.0, 22.0])


def _on_face(up_axis: int, yaw_deg: float) -> np.ndarray:
    """Build a proper rotation matrix (cube in world frame) with the body
    axis ``up_axis`` (0/1/2) vertical and the cube yawed by ``yaw_deg``.

    Returns an orthonormal, right-handed matrix R (det = +1) whose columns
    are the cube body axes in world frame.
    """
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    horizontal = [np.array([c, s, 0.0]), np.array([-s, c, 0.0])]
    vertical = np.array([0.0, 0.0, 1.0])

    if up_axis == 0:
        return np.column_stack([vertical, horizontal[0], horizontal[1]])
    if up_axis == 1:
        return np.column_stack([horizontal[0], vertical, horizontal[1]])
    return np.column_stack([horizontal[0], horizontal[1], vertical])


def _true_wall_normals(rot: np.ndarray) -> list[np.ndarray]:
    """The 4 wall-face outward normals in the XY plane, from the two
    horizontal (largest-|XY|) cube axes."""
    plan = []
    for col in range(3):
        v = rot[:2, col]
        length = np.linalg.norm(v)
        if length > 0.5:
            plan.append(v / length)
    return [plan[0], -plan[0], plan[1], -plan[1]]


class TestPickFaceNormal:
    @pytest.mark.parametrize("up_axis", [0, 1, 2])
    @pytest.mark.parametrize("yaw_deg", [0, 23, 77, 130])
    def test_returns_a_true_wall_normal_for_any_resting_face(self, up_axis, yaw_deg):
        rot = _on_face(up_axis, yaw_deg)
        face = Navigator._pick_face_normal(
            CUBE, rot, (float(CUBE[0]) + 200.0, float(CUBE[1]) + 60.0), 0.5
        )
        assert face is not None
        nx, ny, _ = face
        distances = [
            np.linalg.norm([nx - w[0], ny - w[1]]) for w in _true_wall_normals(rot)
        ]
        assert min(distances) < 1e-6

    def test_flat_cube_robot_due_east_approaches_x_wall(self):
        """Cube resting flat (body Y up), robot due +X: the wall facing the
        robot is the +X wall — never a 45° direction."""
        rot = _on_face(1, 0.0)  # up_axis=1 => Y vertical
        face = Navigator._pick_face_normal(
            CUBE, rot, (float(CUBE[0]) + 200.0, float(CUBE[1])), 0.0
        )
        assert face is not None
        nx, ny, _ = face
        assert (nx, ny) == pytest.approx((1.0, 0.0), abs=1e-6)

    def test_vertical_axis_noise_does_not_win(self):
        """Regression: old code normalized the near-vertical column's small
        XY noise to unit length, producing a plausible ~45° stand-in that
        could beat both true walls. The vertical axis must be excluded."""
        # Cube with body Y near-vertical but with XY noise (0.02, 0.03).
        rot = _on_face(1, 0.0).copy()
        rot[0, 1] = 0.02
        rot[1, 1] = 0.03
        rot[2, 1] = math.sqrt(max(0.0, 1.0 - 0.02**2 - 0.03**2))
        # Robot at 45° NE of the cube: both true walls score ~0.707, so the
        # old code's noisy normal (0.55, 0.83) could win.
        face = Navigator._pick_face_normal(
            CUBE, rot, (float(CUBE[0]) + 200.0, float(CUBE[1]) + 200.0), 0.0
        )
        assert face is not None
        nx, ny, _ = face
        true_walls = _true_wall_normals(rot)
        distances = [np.linalg.norm([nx - w[0], ny - w[1]]) for w in true_walls]
        assert min(distances) < 1e-6

    def test_degenerate_rotation_returns_none(self):
        rot = np.zeros((3, 3))
        face = Navigator._pick_face_normal(
            CUBE, rot, (float(CUBE[0]) + 100.0, float(CUBE[1])), 0.0
        )
        assert face is None

    def test_cube_at_robot_returns_none(self):
        face = Navigator._pick_face_normal(
            CUBE, np.eye(3), (float(CUBE[0]), float(CUBE[1])), 0.0
        )
        assert face is None
