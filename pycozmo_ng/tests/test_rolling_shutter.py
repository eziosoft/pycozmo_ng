"""Tests for RollingShutterCorrector — the engine rolling-shutter port.

Every engine behavior is asserted against the ported algorithm:
per-sample stamping (``frame_ts - (rows - line) * 65/rows``), straggler
extrapolation from the preceding stamped entry, per-band rate integration
into cumulative shifts, and the band-wise horizontal warp. No robot.
"""

from itertools import pairwise

import numpy as np
import pytest

from pycozmo_ng.vision.rolling_shutter import (
    FRAME_READOUT_MS,
    MAX_HISTORY_SAMPLES,
    RollingShutterCorrector,
)

IMAGE_ID = 17
FRAME_TS = 10000.0


def make_image(rows=240, columns=320):
    """A test image with a unique value per column (shifts are visible)."""
    row = np.arange(columns, dtype=np.uint16).astype(np.uint8)
    return np.repeat(row.reshape(1, columns), rows, axis=0)


class TestAddSample:
    def test_keeps_most_recent_samples(self):
        corrector = RollingShutterCorrector()
        for i in range(MAX_HISTORY_SAMPLES + 5):
            corrector.add_sample(i, 0.1, 0.2, 0.3, i)
        samples = corrector._samples
        assert len(samples) == MAX_HISTORY_SAMPLES
        assert samples[0].image_id == 5  # the five oldest dropped
        assert samples[-1].image_id == MAX_HISTORY_SAMPLES + 4

    def test_packet_handler_adds_sample(self):
        corrector = RollingShutterCorrector()
        pkt = type("Pkt", (), {})()
        pkt.image_id = IMAGE_ID
        pkt.rate_x = 1.0
        pkt.rate_y = 2.0
        pkt.rate_z = 3.0
        pkt.line_2_number = 100
        corrector.on_image_imu_data(None, pkt)
        sample = corrector._samples[0]
        assert (sample.image_id, sample.rate_x, sample.rate_y, sample.rate_z, sample.line_number) == (
            IMAGE_ID,
            1.0,
            2.0,
            3.0,
            100,
        )
        assert sample.stamp == 0.0  # unstamped until an image arrives


class TestStamping:
    def test_same_image_samples_follow_readout_formula(self):
        """stamp = frame_ts - (rows - line) * 65/rows (engine formula)."""
        corrector = RollingShutterCorrector()
        corrector.add_sample(IMAGE_ID, 0.0, 0.0, 0.0, 0)  # first row read out
        corrector.add_sample(IMAGE_ID, 0.0, 0.0, 0.0, 238)  # last band row
        corrector._stamp_samples(IMAGE_ID, int(FRAME_TS), 240)
        first, last = corrector._samples
        rows = 240
        per_row = FRAME_READOUT_MS / rows
        assert first.stamp == pytest.approx(FRAME_TS - (rows - 0) * per_row)
        assert last.stamp == pytest.approx(FRAME_TS - (rows - 238) * per_row)
        # The full read-out spans the whole frame: first stamp is 65 ms back.
        assert FRAME_TS - first.stamp == pytest.approx(FRAME_READOUT_MS)

    def test_straggler_extrapolates_from_preceding_stamped_entry(self):
        """Previous-image stragglers stamp from the entry right before them."""
        corrector = RollingShutterCorrector()
        prev_image = IMAGE_ID - 1
        corrector.add_sample(prev_image, 0.0, 0.0, 0.0, 200)
        corrector.add_sample(IMAGE_ID, 0.0, 0.0, 0.0, 10)
        corrector.add_sample(prev_image, 0.0, 0.0, 0.0, 202)  # straggler
        corrector._stamp_samples(IMAGE_ID, int(FRAME_TS), 240)
        straggler = corrector._samples[2]
        per_row = FRAME_READOUT_MS / 240
        stamped_reference = corrector._samples[1]
        assert straggler.stamp == pytest.approx(stamped_reference.stamp + per_row * (202 - 10))

    def test_straggler_without_stamped_predecessor_stays_unstamped(self):
        corrector = RollingShutterCorrector()
        prev_image = IMAGE_ID - 1
        corrector.add_sample(prev_image, 0.0, 0.0, 0.0, 200)  # also unstamped
        corrector.add_sample(prev_image, 0.0, 0.0, 0.0, 202)
        corrector._stamp_samples(IMAGE_ID, int(FRAME_TS), 240)
        assert corrector._samples[1].stamp == 0.0


class TestComputePixelShifts:
    def test_constant_yaw_gives_linear_cumulative_shifts(self):
        """Constant rate_z integrates into a ramp of exact per-band steps."""
        corrector = RollingShutterCorrector()
        rate = 0.5
        # 120 samples, one per band, all reporting the same rate.
        for band_row in range(0, 240, 2):
            corrector.add_sample(IMAGE_ID, 0.0, 0.0, rate, band_row)
        shifts = corrector.compute_pixel_shifts(int(FRAME_TS), 240)
        assert len(shifts) == 120
        per_band_shift = 22.0 * rate / 120
        # Cumulative: shift[k] = (k+1) * per_band_shift (exact rate lookups).
        for k, shift in enumerate(shifts):
            assert shift == pytest.approx((k + 1) * per_band_shift, rel=1e-6)

    def test_shifts_are_cumulative_and_monotonic(self):
        corrector = RollingShutterCorrector()
        for k in range(120):
            corrector.add_sample(IMAGE_ID, 0.0, 0.0, 0.01 * (k + 1), 2 * k)
        shifts = corrector.compute_pixel_shifts(int(FRAME_TS), 240)
        assert all(b > a for a, b in pairwise(shifts))

    def test_empty_history_passes_frame_through(self):
        corrector = RollingShutterCorrector()
        image = make_image()
        out = corrector.correct(image, int(FRAME_TS))
        assert out is image
        assert corrector._warned_no_imu  # warned once, quietly


class TestWarpImage:
    def test_zero_shifts_reproduce_the_input(self):
        corrector = RollingShutterCorrector()
        image = make_image()
        out = corrector.warp_image(image, np.zeros(120))
        assert np.array_equal(out, image)

    def test_band_is_shifted_horizontally_and_aliased_to_zero(self):
        """out[y][x] = in[y][x - shift]; reads falling off the left edge are 0."""
        corrector = RollingShutterCorrector()
        image = make_image()
        shifts = np.zeros(120)
        shift = 3.7  # truncates to 3, engine float→int round
        shifts[40:50] = shift
        out = corrector.warp_image(image, shifts)

        # Bands 41..50 of 120 cover rows 197..216 (two-row bands from bottom).
        # row 197..198 = band 41? Engine band g indexes rows via float math;
        # compute it the same way here to pin the mapping.
        height, width = image.shape
        h1 = height - 1
        bands = 120
        for band in range(41, 51):
            y_start = int(h1 - (h1 / bands) * band)
            y_end = int(h1 - (h1 / bands) * (band - 1))
            for y in range(y_start, y_end):
                for x in range(width):
                    source = x - 3
                    expected = image[y, source] if 0 <= source < width else 0
                    assert out[y, x] == expected

        # Unshifted bands are untouched.
        assert np.array_equal(out[:10], image[:10])

    def test_negative_shift_reads_from_the_right(self):
        corrector = RollingShutterCorrector()
        image = make_image()
        shifts = np.full(120, -2.0)
        out = corrector.warp_image(image, shifts)
        height, width = image.shape
        # Row height-1 is never visited by the engine's truncation math and
        # keeps the original content — excluded here.
        for y in range(height - 1):
            for x in range(width - 2, width):
                assert out[y, x] == 0  # falls off the right edge
            # out[y][x] = in[y][x + 2]
            assert np.array_equal(out[y, : width - 2], image[y, 2:])


class TestCorrectEndToEnd:
    def test_correct_warps_using_imu_history(self):
        corrector = RollingShutterCorrector()
        rate = 0.5
        for band_row in range(0, 240, 2):
            corrector.add_sample(IMAGE_ID, 0.0, 0.0, rate, band_row)
        image = make_image()
        out = corrector.correct(image, int(FRAME_TS), IMAGE_ID)
        shifts = corrector.shifts
        assert len(shifts) == 120
        # Bands near the bottom barely move; bands near the top shift most.
        h1, bands = image.shape[0] - 1, 120
        y_bottom = int(h1 - (h1 / bands) * 1)
        y_top_start = int(h1 - (h1 / bands) * bands)
        assert int(shifts[0]) < int(shifts[119])
        if int(shifts[119]) > 0:
            # The top band row must differ from the input (it was warped).
            assert not np.array_equal(out[y_top_start], image[y_top_start])
            assert np.array_equal(out[y_bottom], image[y_bottom])

    def test_correct_with_none_timestamp_passes_through(self):
        corrector = RollingShutterCorrector()
        corrector.add_sample(IMAGE_ID, 0.0, 0.0, 1.0, 100)
        image = make_image()
        assert corrector.correct(image, None, IMAGE_ID) is image
        assert corrector.correct(image, int(FRAME_TS), IMAGE_ID) is not image
