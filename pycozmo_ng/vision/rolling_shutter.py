"""Rolling shutter correction — port of Cozmo's engine ``RollingShutterCorrector``.

Cozmo's camera rolls its shutter: the sensor read-out takes a whole frame
(~65 ms per the engine), so each 2-row band of an image is exposed at a
slightly different time, and a rotating robot skews the image horizontally.
While a frame is being read out the robot therefore streams ``imageGyro``
(0xF4) samples — each tagged with the frame's ``image_id`` and the
``line_2_number`` (the sensor row of the 2-row band) the sample was taken
on. The engine's ``RollingShutterCorrector`` turns those samples into a
per-band horizontal pixel shift and resamples the image band-by-band.

Engine reference (libcozmoEngine.so, Ghidra):

- ``ImuDataHistory::AddImuData`` (0x00cca8f4) keeps the most recent 20
  ``ImuData`` entries (pop-front on overflow, push-back);
  ``ImuData`` = ``{image_id, rate_x, rate_y, rate_z, line_number, stamp}``.
- ``VisionComponent::SetNextImage`` (0x00e55550) stamps the history on
  image arrival: ``stamp = frame_ts - (line_count - line) * 65.0/line_count``
  — the full-frame read-out is a hardcoded 65.0 ms. Entries of the
  *previous* image that arrived late are extrapolated from an earlier
  stamped neighbour: ``stamp = ref.stamp + 65/line_count * (line - ref.line)``
  (``ImuDataHistory::CalculateTimestampForImageIMU``, 0x00ccaa30).
- ``VisionSystem::Update`` (0x00e05ca8) computes the shifts after CLAHE
  (flag-gated): 120 iterations, one per 2-row band, each querying the
  history at ``frame_ts - i * 65/120`` and evaluating the interpolated
  rate at row ``(h/120) * (119 - i)`` (``ComputePixelShifts`` 0x00ccb0d8,
  ``ComputePixelShiftsWithImageIMU`` 0x00ccb2f8). Rates are linearly
  interpolated between the bracketing history samples; when consecutive
  samples straddle an image boundary the line numbers are shifted by a
  whole frame height to keep the merged timeline monotonic.
  Each band contributes ``rate * 22.0 / 120`` pixels and the
  contributions accumulate: ``shift[k]`` is the running integral over the
  k+1 most recently read-out bands.
- ``WarpImage`` (0x00ccb768) resamples: ``out[y][x] = in[y][x - shift]``
  per 2-row band (shift truncated toward zero, out-of-range reads 0).

The engine also accumulated a vertical shift term from ``rate_y`` (pitch),
gated by a corrector flag — but its ``WarpImage`` reads only the horizontal
component, so that term is dead code in this build and is not ported here.

Here the corrector owns the history and is fed ``ImageImuData`` packets
directly; ``FrameStage`` calls :meth:`correct` on each frame before
detection (no CLAHE step exists in the pycozmo_ng pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pycozmo_ng.logger import logger_cubes

__all__ = ["FRAME_READOUT_MS", "MAX_HISTORY_SAMPLES", "ImuSample", "RollingShutterCorrector"]

#: Full-frame sensor read-out in milliseconds (engine hardcoded value).
FRAME_READOUT_MS = 65.0

#: ``ImuDataHistory`` keeps at most this many samples (engine ``AddImuData``).
MAX_HISTORY_SAMPLES = 20

#: Pixels per (rate unit band) — the hardcoded 22.0 in
#: ``ComputePixelShiftsWithImageIMU``, scaled per band as ``22.0/bands``.
PIXEL_SCALE = 22.0

#: A sensor row is covered over the frame height; rows come in 2-row bands.
_ROWS_PER_BAND = 2


@dataclass
class ImuSample:
    """One ``imageGyro`` sample: angular rates plus the band row it was read out on.

    ``line_number`` is the raw ``line_2_number`` — the sensor row of the
    2-row band a sample was taken on, counting within that sample's image.
    ``stamp`` is the exposure time in milliseconds; it is computed per
    image by :meth:`RollingShutterCorrector._stamp_samples` until then it
    is 0 (unstamped).
    """

    image_id: int
    rate_x: float
    rate_y: float
    rate_z: float
    line_number: int
    stamp: float = 0.0


class RollingShutterCorrector:
    """Corrects rolling-shutter skew in camera frames from in-frame IMU samples.

    Fed by ``ImageImuData`` (0xF4) packets (:meth:`on_image_imu_data`); each
    frame is corrected with :meth:`correct(image, frame_timestamp)` before
    detection. When no ``imageGyro`` data has arrived, ``correct`` passes
    frames through unchanged — the engine warns per frame; here it warns
    once so a robot without IMU streaming does not flood the log.
    """

    def __init__(self) -> None:
        self._samples: list[ImuSample] = []
        self._warned_no_imu = False
        # Per-band pixel shifts of the most recently corrected frame
        # (index 0 = latest-read-out band) — see ``compute_pixel_shifts``.
        self._shifts: np.ndarray = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def on_image_imu_data(self, cli, pkt) -> None:
        """Client packet handler for ``ImageImuData`` (0xF4)."""
        del cli
        self.add_sample(pkt.image_id, pkt.rate_x, pkt.rate_y, pkt.rate_z, pkt.line_2_number)

    def add_sample(self, image_id: int, rate_x: float, rate_y: float, rate_z: float, line_number: int) -> None:
        """Record one IMU sample (engine ``ImuDataHistory::AddImuData``).

        Keeps the most recent :data:`MAX_HISTORY_SAMPLES`; the oldest is
        dropped first.
        """
        if len(self._samples) >= MAX_HISTORY_SAMPLES:
            self._samples.pop(0)
        self._samples.append(
            ImuSample(
                image_id=image_id & 0xFFFFFFFF,
                rate_x=float(rate_x),
                rate_y=float(rate_y),
                rate_z=float(rate_z),
                line_number=int(line_number) & 0xFF,
            )
        )

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------

    def correct(self, image: np.ndarray, frame_timestamp: int | None, frame_image_id: int | None = None) -> np.ndarray:
        """Return ``image`` with rolling-shutter skew removed.

        Stamps the samples for ``frame_image_id``, integrates the angular
        rate over the frame's 2-row bands into the shift buffer, then
        warps each band horizontally. With no samples (or no timestamp)
        the image passes through unchanged and a one-shot warning is
        logged — no robot IMU streaming means nothing to correct.
        """
        if frame_timestamp is None or not self._samples:
            self._warn_no_imu()
            return image

        self._stamp_samples(
            frame_image_id & 0xFFFFFFFF if frame_image_id is not None else None,
            frame_timestamp & 0xFFFFFFFF,
            image.shape[0],
        )
        self._shifts = np.asarray(self.compute_pixel_shifts(frame_timestamp & 0xFFFFFFFF, image.shape[0]))
        return self.warp_image(image, self.shifts)

    @property
    def shifts(self) -> np.ndarray:
        """Per-band horizontal pixel shifts of the last corrected frame.

        Index 0 is the most recently read-out band (bottom of the image, a
        single-sample shift); each following index accumulates one more
        band's worth of integrated rate — the shift buffer the engine
        feeds to ``WarpImage``.
        """
        return self._shifts

    def compute_pixel_shifts(self, frame_timestamp: int, num_rows: int) -> list[float]:
        """Engine ``ComputePixelShifts`` — per-band cumulative horizontal shifts.

        One query per 2-row band at ``frame_ts - i * readout/bands``, with
        the interpolated rate evaluated at row ``(num_rows/bands) * (bands - i)``
        (the even row of the band), accumulated in read-out order so
        ``shifts[k]`` is the integral over the ``k + 1`` latest bands.
        """
        bands = self._bands(num_rows)
        per_band_ms = FRAME_READOUT_MS / bands
        rows_per_band = _ROWS_PER_BAND

        shifts: list[float] = []
        cumulative = 0.0
        failed = False
        for i in range(1, bands + 1):
            target_stamp = frame_timestamp - i * per_band_ms
            target_row = rows_per_band * (bands - i)
            rate = self._sample_rate_at(target_stamp, target_row, num_rows)
            if rate is None:
                failed = True
                rate = (0.0, 0.0)
            cumulative += PIXEL_SCALE * rate[1] / bands
            shifts.append(cumulative)

        if failed:
            self._warn_lookup(frame_timestamp)
        return shifts

    def warp_image(self, image: np.ndarray, shifts: np.ndarray) -> np.ndarray:
        """Engine ``WarpImage`` — resample each 2-row band horizontally.

        ``out[y][x] = in[y][x - shift]`` with the shift truncated toward
        zero like the engine's float→int cast; samples outside the row are
        0. Band boundaries replicate the engine's float arithmetic
        (``(height-1) - (height-1)/bands * band`` truncated), so the
        output is bit-identical to the engine's intent for a 240-row frame.
        """
        num_rows = image.shape[0]
        bands = self._bands(num_rows)
        # Copy of the input: the engine allocates a fresh image, and its
        # truncating band math leaves the very last row unwritten (the
        # engine's buffer there is stale/undefined). Keeping the original
        # content is the deterministic equivalent for a live pipeline.
        out = image.copy()
        if len(shifts) < bands:
            return out

        height_minus_one = num_rows - 1
        for band in range(1, bands + 1):
            y_start = int(height_minus_one - (height_minus_one / bands) * band)
            y_end = int(height_minus_one - (height_minus_one / bands) * (band - 1))
            if y_start >= y_end:
                continue
            sx = int(self._wrap_shift(shifts[band - 1]))
            if sx == 0:
                out[y_start:y_end] = image[y_start:y_end]
                continue
            columns = np.arange(image.shape[1]) - sx
            valid = (columns >= 0) & (columns < image.shape[1])
            out_band = out[y_start:y_end]
            out_band[:, valid] = image[y_start:y_end, :][:, columns[valid]]
            out_band[:, ~valid] = 0
        return out

    # ------------------------------------------------------------------
    # Stamping (engine ImuDataHistory::CalculateTimestampForImageIMU)
    # ------------------------------------------------------------------

    def _stamp_samples(self, frame_image_id: int | None, frame_timestamp: int, num_rows: int) -> None:
        """Turn raw line numbers into exposure times for this and the previous image.

        Matches the deque scan of ``CalculateTimestampForImageIMU``: samples
        belonging to this image are stamped from the formula; stragglers of
        the previous image (still unstamped) are extrapolated from the
        entry immediately before them in arrival order, and only when that
        entry carries a stamp. ``frame_image_id`` None leaves everything
        unstamped (timestamps stay unknown — samples are simply skipped).
        """
        if frame_image_id is None:
            return
        ts_per_row = FRAME_READOUT_MS / num_rows
        previous_id = (frame_image_id - 1) & 0xFFFFFFFF
        for index, sample in enumerate(self._samples):
            if sample.image_id == frame_image_id:
                sample.stamp = frame_timestamp - (num_rows - sample.line_number) * ts_per_row
            elif (
                sample.image_id == previous_id
                and sample.stamp == 0.0
                and index > 0
                and self._samples[index - 1].stamp != 0.0
            ):
                previous = self._samples[index - 1]
                sample.stamp = previous.stamp + ts_per_row * (sample.line_number - previous.line_number)

    # ------------------------------------------------------------------
    # Rate lookup (engine ComputePixelShiftsWithImageIMU)
    # ------------------------------------------------------------------

    def _sample_rate_at(self, target_stamp: float, target_row: float, num_rows: int):
        """Interpolate ``(rate_y, rate_z)`` at ``target_row``/``target_stamp``.

        Returns ``None`` when the bracketing sample has no predecessor
        (engine "not found"); on history exhaustion the engine falls back
        to the last sample's rates, mirrored here.
        """
        samples = self._samples
        # First sample stamped at/after the target (forward scan).
        for upper_idx, sample in enumerate(samples):
            if sample.stamp >= target_stamp:
                if upper_idx == 0:
                    return None
                lower = samples[upper_idx - 1]
                return self._interpolate(lower, sample, target_row, num_rows)
        # No sample at/after the target: engine uses the last sample as-is.
        return (samples[-1].rate_y, samples[-1].rate_z)

    def _interpolate(self, lower: ImuSample, upper: ImuSample, target_row: float, num_rows: int):
        """Linear interpolation between two bracketing samples (engine's line math)."""
        lower_line = lower.line_number
        upper_line = upper.line_number
        if upper.stamp == 0.0:
            upper_line += num_rows
        elif lower.image_id < upper.image_id:
            lower_line -= num_rows

        span = upper_line - lower_line
        if span == 0:
            return (upper.rate_y, upper.rate_z)
        frac = (target_row - lower_line) / span
        rate_y = lower.rate_y + (upper.rate_y - lower.rate_y) * frac
        rate_z = lower.rate_z + (upper.rate_z - lower.rate_z) * frac
        return (rate_y, rate_z)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bands(self, num_rows: int) -> int:
        """Number of 2-row bands — 120 for the engine's 240-row frames."""
        return num_rows // _ROWS_PER_BAND

    @staticmethod
    def _wrap_shift(shift: float) -> float:
        """Truncate toward zero the way the engine's float→int cast does."""
        return np.trunc(shift)

    def _warn_no_imu(self) -> None:
        """Engine ``RollingShutterCorrector.ComputePixelShifts.EmptyHistory`` — once."""
        if self._warned_no_imu:
            return
        self._warned_no_imu = True
        logger_cubes.warning("Rolling shutter correction: no image IMU data received")

    def _warn_lookup(self, frame_timestamp: int) -> None:
        """Engine ``RollingShutterCorrector.ComputePixelShifts.NoImageIMUData``."""
        first, last = None, None
        traces = [sample.stamp for sample in self._samples]
        if traces:
            first, last = int(traces[0]), int(traces[-1])
        logger_cubes.warning(
            "Rolling shutter: no image IMU data at timestamp %s (have stamps %s:%s)",
            frame_timestamp,
            first,
            last,
        )
