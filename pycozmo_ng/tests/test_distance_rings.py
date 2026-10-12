"""Tests for DistanceRings — the ground range-band overlay.

The projection is asserted against the camera model worked by hand (K known,
head level), and the drawing is asserted on pixels — no robot, no Brain.
"""

import numpy as np
import pytest
from PIL import Image

from pycozmo_ng.vision.distance_rings import (
    DEFAULT_BAND_HALF_WIDTH_MM,
    DEFAULT_OUTLINE_WIDTH_PX,
    DEFAULT_RING_COLORS,
    DEFAULT_RING_RADII_MM,
    DistanceRings,
)

WIDTH, HEIGHT = 320, 240
FX = FY = 300.0
CX, CY = 160.0, 120.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def make_rings(**kwargs) -> DistanceRings:
    return DistanceRings(K, **kwargs)


def black_image() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))


def white_pixels(image: Image.Image) -> int:
    """Count of non-black pixels."""
    return int(np.any(np.asarray(image), axis=-1).sum())


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

    def test_farther_ring_climbs_toward_the_horizon(self):
        """v shrinks with radius and approaches cy — never crosses it."""
        rings = make_rings()
        vs = []
        for r in DEFAULT_RING_RADII_MM:
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


class TestDrawing:
    def test_draw_marks_pixels(self):
        image = black_image()
        make_rings().draw(image, head_angle_rad=0.0)

        assert white_pixels(image) > 100

    def test_ring_apex_label_sits_below_the_apex(self):
        """The '250 mm' label is drawn under the ring's forward point."""
        rings = make_rings(radii_mm=(250,))
        image = black_image()
        rings.draw(image, head_angle_rad=0.0)

        apex_v = CY + FY * 38.0 / 220.0  # ≈ 172
        band = image.crop((CX - 40, int(apex_v) + 6, CX + 40, int(apex_v) + 24))
        assert white_pixels(band) > 10, "no label glyphs below the apex"

    def test_empty_radii_draw_nothing(self):
        rings = make_rings(radii_mm=())
        image = black_image()

        rings.draw(image, head_angle_rad=0.0)

        assert white_pixels(image) == 0

    def test_ring_behind_the_camera_draws_nothing(self):
        """Every sample invalid (bands hug the camera's rear) — no stray pixels."""
        rings = DistanceRings(K, radii_mm=(1,), samples=8)
        image = black_image()

        rings.draw(image, head_angle_rad=0.0)  # must not raise

        assert white_pixels(image) == 0

    def test_default_radii_are_the_three_band_set(self):
        assert DEFAULT_RING_RADII_MM == (250, 500, 1000)
        assert make_rings().radii_mm == (250.0, 500.0, 1000.0)

    def test_default_colors_are_distinct(self):
        assert DEFAULT_RING_COLORS == ((0, 220, 0), (255, 220, 0), (255, 80, 0))

    def test_band_width_and_outline_defaults(self):
        assert DEFAULT_BAND_HALF_WIDTH_MM == 20.0
        assert DEFAULT_OUTLINE_WIDTH_PX == 2

    def test_non_positive_radii_are_dropped(self):
        rings = DistanceRings(K, radii_mm=(0, -5, 250))

        assert rings.radii_mm == (250.0,)

    def test_colored_bands_produce_distinct_colors(self):
        """Green, yellow, and red pixels all appear in the rendered frame."""
        rings = make_rings()
        image = black_image()
        rings.draw(image, head_angle_rad=0.0)

        arr = np.asarray(image)
        green = (arr[..., 0] < 60) & (arr[..., 1] > 180) & (arr[..., 2] < 60)
        yellow = (arr[..., 0] > 200) & (arr[..., 1] > 150) & (arr[..., 2] < 60)
        red = (arr[..., 0] > 200) & (arr[..., 1] < 120) & (arr[..., 2] < 60)

        assert green.any(), "no green band (250 mm) pixels"
        assert yellow.any(), "no yellow band (500 mm) pixels"
        assert red.any(), "no red band (1000 mm) pixels"

    def test_filled_bands_are_thicker_than_outline_only(self):
        """With bands filled, the pixel count is far higher than a 1-px outline."""
        img = black_image()
        make_rings().draw(img, head_angle_rad=0.0)
        band_count = white_pixels(img)

        # A pure 1-px outline of 3 rings would be a few hundred pixels at most;
        # filled bands cover many more.
        assert band_count > 500, f"bands too sparse ({band_count} px) — not filled?"


class TestContiguousRuns:
    def test_no_valid_points_returns_empty(self):
        from pycozmo_ng.vision.distance_rings import _contiguous_runs

        assert _contiguous_runs(np.array([False, False, False])) == []

    def test_full_run_returns_single_list(self):
        from pycozmo_ng.vision.distance_rings import _contiguous_runs

        runs = _contiguous_runs(np.array([True, True, True]))
        assert runs == [[0, 1, 2]]

    def test_wraparound_run_is_merged(self):
        from pycozmo_ng.vision.distance_rings import _contiguous_runs

        mask = np.array([True, True, False, False, True, True, True])
        runs = _contiguous_runs(mask)
        # First and last True edges merge into one wraparound run.
        assert len(runs) == 1
        assert runs[0] == [4, 5, 6, 0, 1]

    def test_multiple_runs_split(self):
        from pycozmo_ng.vision.distance_rings import _contiguous_runs

        mask = np.array([False, True, True, False, True, True, False])
        runs = _contiguous_runs(mask)
        assert runs == [[1, 2], [4, 5]]


class TestSegmentClip:
    def test_segment_inside_frame_passes_through(self):
        from pycozmo_ng.vision.distance_rings import _clip_segment

        clipped = _clip_segment((10.0, 10.0), (100.0, 100.0), WIDTH, HEIGHT)

        assert clipped == ((10.0, 10.0), (100.0, 100.0))

    def test_crossing_segment_is_cut_at_the_edge(self):
        from pycozmo_ng.vision.distance_rings import _clip_segment

        clipped = _clip_segment((-500.0, 50.0), (500.0, 50.0), WIDTH, HEIGHT)

        assert clipped is not None
        (x0, y0), (x1, y1) = clipped
        assert 0 <= x0 <= WIDTH - 1
        assert 0 <= x1 <= WIDTH - 1
        assert y0 == y1 == pytest.approx(50.0)

    def test_miss_returns_none(self):
        from pycozmo_ng.vision.distance_rings import _clip_segment

        assert _clip_segment((-500.0, -50.0), (-10.0, -10.0), WIDTH, HEIGHT) is None
        assert _clip_segment((100.0, -50.0), (100.0, -10.0), WIDTH, HEIGHT) is None
