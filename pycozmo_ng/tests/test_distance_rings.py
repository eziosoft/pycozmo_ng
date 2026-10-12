"""Tests for DistanceRings — the translucent ground-distance zones.

The projection is asserted against the camera model worked by hand (K known,
head level), and the drawing is asserted on pixels — no robot, no Brain.
"""

import numpy as np
import pytest
from PIL import Image

from pycozmo_ng.vision.distance_rings import (
    DEFAULT_ZONE_ALPHAS,
    DEFAULT_ZONE_COLORS,
    DEFAULT_ZONE_RADII_MM,
    DistanceRings,
)

WIDTH, HEIGHT = 320, 240
FX = FY = 300.0
CX, CY = 160.0, 120.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])

# Tint rendered over black: base * (1 - a) + color * a.
NEON_TINT = (255, 130, 0)


def make_rings(**kwargs) -> DistanceRings:
    return DistanceRings(K, **kwargs)


def solid_image(color=(0, 0, 0)) -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), color)


def tinted_pixels(image: Image.Image, base) -> np.ndarray:
    """Boolean (H, W) mask of pixels that differ from the *base* colour."""
    arr = np.asarray(image)
    return np.any(arr != np.array(base, dtype=np.uint8), axis=-1)


class TestProjection:
    def test_straight_ahead_point_projects_to_principal_column(self):
        """A point on body Y = 0 has camera X = 0 → u = cx."""
        rings = make_rings()
        uv, valid = rings.project_ground_points(np.array([[250.0, 0.0]]), head_angle_rad=0.0)

        assert valid[0]
        assert uv[0, 0] == pytest.approx(CX)

    def test_ground_point_projects_below_the_optical_axis(self):
        """Head level, camera 38 mm up: v = cy + fy·38/(r-30)."""
        rings = make_rings()
        uv, valid = rings.project_ground_points(np.array([[250.0, 0.0]]), head_angle_rad=0.0)

        assert valid[0]
        assert uv[0, 1] == pytest.approx(CY + FY * 38.0 / 220.0)
        assert uv[0, 1] > CY  # ground is always below the optical axis

    def test_farther_boundary_climbs_toward_the_horizon(self):
        """v shrinks with radius and approaches cy — never crosses it."""
        rings = make_rings()
        vs = []
        for r in DEFAULT_ZONE_RADII_MM:
            pt_uv, pt_valid = rings.project_ground_points(np.array([[float(r), 0.0]]), 0.0)
            assert pt_valid[0]
            vs.append(pt_uv[0, 1])

        assert vs == sorted(vs, reverse=True)
        assert all(CY < v < HEIGHT for v in vs)

    def test_points_behind_the_camera_are_invalid(self):
        rings = make_rings()
        # Body x = -500 is well behind the camera at any head angle.
        uv, valid = rings.project_ground_points(np.array([[-500.0, 0.0]]), head_angle_rad=0.0)

        assert not valid[0]
        assert np.isnan(uv[0]).all()

    def test_head_up_pushes_the_ground_down_in_frame(self):
        rings = make_rings()
        uv_level, _ = rings.project_ground_points(np.array([[250.0, 0.0]]), head_angle_rad=0.0)
        uv_up, valid_up = rings.project_ground_points(np.array([[250.0, 0.0]]), head_angle_rad=0.3)

        assert valid_up[0]
        assert uv_up[0, 1] > uv_level[0, 1]  # tilted-up camera drops the floor lower


class TestGroundDistanceMap:
    def test_bottom_of_frame_is_close_ground(self):
        """The bottom-centre pixel looks at floor ~130 mm from the robot."""
        rings = make_rings()
        dist = rings._ground_distance_map((WIDTH, HEIGHT), head_angle_rad=0.0)

        assert dist[HEIGHT - 1, int(CX)] < 250.0

    def test_distance_grows_up_the_frame(self):
        rings = make_rings()
        dist = rings._ground_distance_map((WIDTH, HEIGHT), head_angle_rad=0.0)

        column = dist[130:HEIGHT, int(CX)]
        assert np.all(np.diff(column) < 0)  # closer to horizon = farther
        assert column[-1] < column[0]

    def test_above_the_horizon_is_nan(self):
        """Rays at or above the horizon never hit the ground."""
        rings = make_rings()
        dist = rings._ground_distance_map((WIDTH, HEIGHT), head_angle_rad=0.0)

        assert np.isnan(dist[100, int(CX)])
        assert np.isnan(dist[: int(CY) - 10, :]).all()

    def test_head_tilt_moves_the_distances(self):
        rings = make_rings()
        level = rings._ground_distance_map((WIDTH, HEIGHT), head_angle_rad=0.0)
        tilted = rings._ground_distance_map((WIDTH, HEIGHT), head_angle_rad=0.3)

        assert tilted[HEIGHT - 1, int(CX)] != level[HEIGHT - 1, int(CX)]


class TestDrawing:
    def test_default_configuration_is_two_orange_zones(self):
        assert DEFAULT_ZONE_RADII_MM == (250, 500)
        assert DEFAULT_ZONE_COLORS == ((255, 130, 0), (255, 130, 0))
        assert DEFAULT_ZONE_ALPHAS == (120, 55)
        rings = make_rings()
        assert rings.radii_mm == (250.0, 500.0)

    def test_draw_tints_the_floor(self):
        image = solid_image()
        make_rings().draw(image, head_angle_rad=0.0)

        assert tinted_pixels(image, (0, 0, 0)).sum() > 1000

    def test_tint_is_translucent_not_opaque(self):
        """Both zones mix with the original pixels — never replace them."""
        base = (200, 200, 200)
        image = solid_image(base)
        make_rings().draw(image, head_angle_rad=0.0)
        arr = np.asarray(image)

        near = arr[HEIGHT - 5, int(CX)]  # ~130 mm — inside the near zone
        mid = arr[155, int(CX)]  # between the 250 and 500 mm boundaries

        for pixel in (near, mid):
            assert tuple(pixel) != base, "zone painted over the original colour"
            assert tuple(pixel) != NEON_TINT, "zone is fully opaque"
            # Orange direction: red lifted most, blue lifted least.
            assert pixel[0] > pixel[1] > pixel[2]

    def test_near_zone_is_more_saturated_than_the_far_zone(self):
        base = (200, 200, 200)
        image = solid_image(base)
        make_rings().draw(image, head_angle_rad=0.0)
        arr = np.asarray(image)

        near_excess = int(arr[HEIGHT - 5, int(CX)][0]) - int(arr[HEIGHT - 5, int(CX)][2])
        mid_excess = int(arr[155, int(CX)][0]) - int(arr[155, int(CX)][2])

        assert near_excess > mid_excess > 0

    def test_above_the_horizon_stays_untinted(self):
        base = (200, 200, 200)
        image = solid_image(base)
        make_rings().draw(image, head_angle_rad=0.0)

        assert tuple(np.asarray(image)[100, int(CX)]) == base

    def test_labels_sit_on_the_left_edge(self):
        """Zone labels carry a black backing box — all of them hug the left edge."""
        image = solid_image((200, 200, 200))
        make_rings().draw(image, head_angle_rad=0.0)
        arr = np.asarray(image)

        backing = np.all(arr < 40, axis=-1)  # black boxes on the light floor
        assert backing.sum() > 50, "no label backing boxes rendered"
        assert backing[:, 80:].sum() == 0, "a label leaked right of the left column"

    def test_empty_radii_draw_nothing(self):
        rings = make_rings(radii_mm=())
        image = solid_image()

        rings.draw(image, head_angle_rad=0.0)

        assert tinted_pixels(image, (0, 0, 0)).sum() == 0

    def test_boundary_beyond_the_visible_floor_draws_nothing(self):
        """A 1 mm ring lies behind the camera — no tint, no label, no raise."""
        rings = make_rings(radii_mm=(1,))
        image = solid_image()

        rings.draw(image, head_angle_rad=0.0)

        assert tinted_pixels(image, (0, 0, 0)).sum() == 0

    def test_non_positive_radii_are_dropped(self):
        rings = DistanceRings(K, radii_mm=(0, -5, 250))

        assert rings.radii_mm == (250.0,)

    def test_alpha_out_of_range_is_clamped(self):
        rings = DistanceRings(K, alphas=(-10, 999))

        assert rings._alphas == (0, 255)
