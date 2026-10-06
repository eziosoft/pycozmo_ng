"""Single owner of the robot-center measurement (CONTEXT: Robot center calibration).

The camera's lateral offset from the robot centre is measured once — head at
max-down, the two ring markers visible — and then frozen for the session.
Nobody writes that value directly: everyone goes through this object.

* The detection pipeline calls :meth:`observe` once per processed frame while
  a run is armed. Each frame that sees the rings contributes exactly one
  reading, so a reading can never be counted twice.
* The calibration run (``Cozmo.calibrate_robot_center``) arms the run, waits
  for :attr:`readings`, then publishes the average with :meth:`freeze` — or
  disarms without publishing when the rings never showed up.

Between runs the detector does no work, which is what makes the value
immutable outside an explicit run: an arbitrary head-down frame cannot latch
the session offset, and only :meth:`freeze` ever writes it.
"""

from __future__ import annotations

import threading

import numpy as np

from pycozmo_ng.vision.calibration.robot_center import RobotCenterResult
from pycozmo_ng.vision.interfaces import RobotCenterDetector

__all__ = ["RobotCenterCalibration"]


class RobotCenterCalibration:
    """Observation, freshness, and freezing of the robot-center measurement."""

    def __init__(self, detector: RobotCenterDetector | None = None) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self._collecting = False
        self._measured = False
        self._offset_mm = 0.0
        self._readings: list[float] = []
        self._last_result: RobotCenterResult | None = None

    @property
    def measured(self) -> bool:
        """True once :meth:`freeze` has published a session value."""
        with self._lock:
            return self._measured

    @property
    def offset_mm(self) -> float:
        """Lateral offset of the camera from the robot centre (mm).

        Frozen for the session; 0.0 until the first run publishes a value.
        """
        with self._lock:
            return self._offset_mm

    @property
    def last_result(self) -> RobotCenterResult | None:
        """Latest detector output, for annotation. Stale between runs."""
        with self._lock:
            return self._last_result

    @property
    def readings(self) -> tuple[float, ...]:
        """Offsets collected since :meth:`arm` — one per frame that saw the rings."""
        with self._lock:
            return tuple(self._readings)

    def arm(self) -> None:
        """Start a calibration run: frames now feed :attr:`readings`.

        The published session value stays in force until :meth:`freeze`
        replaces it, so a re-run never un-measures the robot.
        """
        with self._lock:
            self._collecting = True
            self._readings = []

    def disarm(self) -> None:
        """End the run without publishing. Collected readings are kept."""
        with self._lock:
            self._collecting = False

    def observe(self, img_bgr: np.ndarray, head_angle_rad: float) -> None:
        """Run the detector on one frame of an armed run.

        Each frame that detects the rings appends exactly one fresh reading.
        No-op when no run is armed, so the detector only works while somebody
        is collecting.
        """
        with self._lock:
            if not self._collecting or self._detector is None:
                return
            self._detector.head_angle_rad = head_angle_rad
            result = self._detector.detect(img_bgr)
            self._last_result = result
            if result is not None and result.detected:
                self._readings.append(result.lateral_offset_mm)

    def freeze(self, offset_mm: float) -> None:
        """Publish ``offset_mm`` as the session value and end the run."""
        with self._lock:
            self._offset_mm = offset_mm
            self._measured = True
            self._collecting = False
