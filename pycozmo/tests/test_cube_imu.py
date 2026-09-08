"""Tests for CubeMoveDetector (pycozmo/cube_imu.py).

Covers the three failure modes that used to drop a *stationary* cube from the
world map once the cube lights were enabled:
  1. garbage accel right after each connect (light worker reconnects ~2 s),
  2. an absolute (biased) resting magnitude reading,
  3. single-sample noise spikes.
"""

from pycozmo.cube_imu import CubeMoveDetector

REST = 9.8  # m/s² — gravity magnitude while sitting on the floor


def make_detector(connected_at: float = 0.0, **kwargs) -> CubeMoveDetector:
    det = CubeMoveDetector(**kwargs)
    det.on_connection(connected_at)
    return det


def rest_samples(det, t0: float, n: int = 5, mag: float = REST, step: float = 0.03):
    """Feed ``n`` at-rest samples starting at ``t0`` (11 = after grace)."""
    return [det.update(t0 + i * step, mag) for i in range(n)]


class TestGracePeriod:
    def test_samples_during_grace_are_ignored(self):
        det = make_detector()
        # Wild samples within the 1.5 s grace window — all ignored.
        assert det.update(0.1, 100.0) is False
        assert det.update(0.5, 0.0) is False
        assert det.update(1.4, 250.0) is False

    def test_connected_at_none_is_ignored(self):
        det = CubeMoveDetector()  # never connected
        assert det.update(10.0, 100.0) is False


class TestStillCube:
    def test_resting_never_triggers(self):
        det = make_detector()
        rest_samples(det, 2.0, n=50)
        assert det.update(3.5, REST) is False

    def test_single_noise_spike_does_not_trigger(self):
        det = make_detector()
        rest_samples(det, 2.0, n=5)
        # One spike, then back to rest — must not accumulate a trigger.
        assert det.update(2.2, 30.0) is False
        rest_samples(det, 2.23, n=5)
        assert det.update(2.4, 30.0) is False  # fresh spike resets again
        rest_samples(det, 2.43, n=5)
        assert det._over_count == 0

    def test_biased_resting_magnitude_never_triggers(self):
        # If the sensor is biased (resting mag = 20, not 9.8), the old
        # absolute threshold would fire constantly. Baseline-relative must not.
        det = make_detector()
        rest_samples(det, 2.0, n=5, mag=20.0)  # learn the biased baseline
        assert all(v is False for v in rest_samples(det, 2.3, n=20, mag=20.0))


class TestRealMove:
    def test_sustained_move_triggers_once_then_rebaselines(self):
        det = make_detector()
        rest_samples(det, 2.0, n=10)
        # Cube picked up: sustained deviation well above 6.0.
        results = [det.update(2.5 + i * 0.03, REST + 12.0) for i in range(6)]
        # Deviation is huge, so >debounce consecutive samples must fire ONCE.
        assert results.count(True) == 1
        assert results.index(True) < len(results) - 1
        # After the reset, the cube now rests at its new spot without re-firing.
        assert all(v is False for v in rest_samples(det, 3.0, n=10, mag=REST + 12.0))

    def test_sustained_move_below_debounce_does_not_trigger(self):
        det = make_detector(debounce=5)
        rest_samples(det, 2.0, n=5)
        results = [det.update(2.5 + i * 0.03, REST + 10.0) for i in range(4)]
        assert results == [False, False, False, False]

    def test_after_connect_resets_pending_trigger(self):
        det = make_detector()
        rest_samples(det, 2.0, n=5)
        det.update(2.5, REST + 20.0)
        det.update(2.53, REST + 20.0)  # two of three needed
        det.on_connection(10.0)  # cube blips / reconnects
        assert det.update(10.1, 100.0) is False  # in grace again
        assert det.update(12.0, REST + 20.0) is False  # fresh baseline, not a trigger


class TestParams:
    def test_defaults_are_documented(self):
        det = CubeMoveDetector()
        assert det.grace_s == 1.5
        assert det.deviation == 6.0
        assert det.debounce == 3
        assert det.alpha == 0.5

    def test_debounce_floor_is_one(self):
        det = CubeMoveDetector(debounce=0)
        assert det.debounce == 1
