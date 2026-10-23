"""Cube motion gate — refuse new cube detections while the robot turns fast enough to blur."""


class CubeMotionGate:
    """True while turning fast enough that blurred blobs could false-match cube templates.

    The firmware's own gate (``VisionComponent::WasRotatingTooFast``)
    discards marker observations above 30 deg/s of body yaw or 10 deg/s of
    head motion; the same thresholds are used here. Body yaw comes from the
    wheel speeds (``TRACK_WIDTH`` apart, so differential speed / track width
    is the yaw rate) — the gyro axes' units are not verified, while wheel
    speeds are millimetres per second by protocol.
    """

    def __init__(self, body_rate_gate_radps: float = 0.5236, head_rate_gate_radps: float = 0.1745) -> None:
        self.body_rate_gate_radps = body_rate_gate_radps
        self.head_rate_gate_radps = head_rate_gate_radps
        self._last_head_angle: float | None = None
        self._last_head_time: float | None = None
        self._head_rate_radps = 0.0

    def observe_head(self, angle_rad: float | None, now: float) -> None:
        """Measure the head's angular rate from successive frames.

        Must run before the gate is consulted: it compares against the previous
        frame's angle and then stores the current one. A missing angle reads as
        no motion (rate 0) and does not disturb the stored sample.
        """
        rate = 0.0
        if angle_rad is not None:
            if self._last_head_angle is not None and self._last_head_time is not None:
                elapsed = now - self._last_head_time
                # Ignore gaps that are not a frame apart (stalls, reconnects).
                if 0.0 < elapsed <= 0.25:
                    rate = abs(angle_rad - self._last_head_angle) / elapsed
            self._head_rate_radps = rate
            self._last_head_angle = angle_rad
            self._last_head_time = now
        else:
            self._head_rate_radps = 0.0

    def is_blurred(self, body_yaw_rate_radps: float) -> bool:
        """True while the robot turns fast enough that detection should be gated."""
        if body_yaw_rate_radps > self.body_rate_gate_radps:
            return True
        return self._head_rate_radps > self.head_rate_gate_radps
