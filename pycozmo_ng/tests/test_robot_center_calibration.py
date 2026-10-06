"""Tests for RobotCenterCalibration — the single owner of the session offset.

The contract under test: readings come from frames while a run is armed (one
reading per frame, never counted twice), and only ``freeze`` writes the
session value.
"""

import numpy as np

from pycozmo_ng.vision.calibration.robot_center import RobotCenterResult
from pycozmo_ng.vision.calibration.robot_center_calibration import RobotCenterCalibration


class FakeDetector:
    """Detector stand-in: hands back whatever result the test queued up."""

    def __init__(self) -> None:
        self.head_angle_rad = 0.0
        self.result: RobotCenterResult | None = None
        self.detect_calls = 0

    def detect(self, img_bgr: np.ndarray) -> RobotCenterResult | None:
        self.detect_calls += 1
        return self.result


def make_image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def make_calibration(result: RobotCenterResult | None = None) -> tuple[RobotCenterCalibration, FakeDetector]:
    detector = FakeDetector()
    detector.result = result
    return RobotCenterCalibration(detector), detector


class TestArming:
    def test_no_work_before_a_run_is_armed(self):
        calib, detector = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=1.0))
        calib.observe(make_image(), -0.4)
        assert detector.detect_calls == 0
        assert calib.readings == ()
        assert calib.measured is False
        assert calib.offset_mm == 0.0

    def test_arming_starts_fresh_and_keeps_previous_value_in_force(self):
        calib, _ = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=4.0))
        calib.arm()
        calib.observe(make_image(), -0.4)
        calib.freeze(sum(calib.readings) / len(calib.readings))

        calib.arm()
        assert calib.readings == ()
        assert calib.measured is True
        assert calib.offset_mm == 4.0  # the old value stays in force during the re-run

    def test_missing_detector_is_harmless(self):
        calib = RobotCenterCalibration(None)
        calib.arm()
        calib.observe(make_image(), -0.4)
        assert calib.readings == ()


class TestObserve:
    def test_one_fresh_reading_per_frame(self):
        calib, detector = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=1.5))
        calib.arm()
        for _ in range(3):
            calib.observe(make_image(), -0.4)
        assert calib.readings == (1.5, 1.5, 1.5)
        assert detector.detect_calls == 3

    def test_missed_frame_records_no_reading_but_is_still_seen(self):
        calib, _ = make_calibration(RobotCenterResult(detected=False))
        calib.arm()
        calib.observe(make_image(), -0.4)
        assert calib.readings == ()
        assert calib.last_result is not None
        assert calib.last_result.detected is False

    def test_head_angle_reaches_the_detector(self):
        calib, detector = make_calibration(RobotCenterResult(detected=False))
        calib.arm()
        calib.observe(make_image(), -0.436)
        assert detector.head_angle_rad == -0.436

    def test_no_observation_after_disarm(self):
        calib, detector = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=2.0))
        calib.arm()
        calib.disarm()
        calib.observe(make_image(), -0.4)
        assert detector.detect_calls == 0
        assert calib.readings == ()

    def test_disarm_keeps_the_readings_it_collected(self):
        calib, _ = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=2.0))
        calib.arm()
        calib.observe(make_image(), -0.4)
        calib.disarm()
        assert calib.readings == (2.0,)


class TestFreeze:
    def test_freeze_publishes_and_ends_the_run(self):
        calib, detector = make_calibration(RobotCenterResult(detected=True, lateral_offset_mm=3.0))
        calib.arm()
        calib.observe(make_image(), -0.4)
        calib.freeze(3.25)

        assert calib.measured is True
        assert calib.offset_mm == 3.25

        calib.observe(make_image(), -0.4)  # run is over — no more collection
        assert detector.detect_calls == 1
        assert calib.offset_mm == 3.25  # observing never rewrites the value

    def test_average_of_the_frames_collected_by_the_run(self):
        calib, detector = make_calibration()
        calib.arm()
        for offset in (1.0, 2.0, 6.0):
            detector.result = RobotCenterResult(detected=True, lateral_offset_mm=offset)
            calib.observe(make_image(), -0.4)
        readings = calib.readings
        calib.freeze(sum(readings) / len(readings))
        assert calib.offset_mm == 3.0
