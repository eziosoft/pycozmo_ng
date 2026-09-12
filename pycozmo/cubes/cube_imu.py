"""
Cube-accelerometer movement detection.

The cube-light worker in ``main.py`` enables the cube's accel stream
(``StreamObjectAccel``) so that physically moving a cube drops it from the
world map and forces a clean re-detection at its new world position (COZMO-27).
That stream is only active while the cube lights are running — which is why a
stationary cube started "vanishing" on the web viewer once the lights were
turned on (COZMO-32): the raw accel samples right after each ``ObjectConnect``
(sent by the light worker every cycle) are unreliable and routinely exceed the
absolute 15 m/s² threshold the old handler used.

``CubeMoveDetector`` makes the trigger robust:

* a **grace period** after connect — the cube's first samples after waking up
  are garbage and are ignored entirely;
* a **resting baseline** (slow exponential follower) — a biased or drifting
  resting reading never trips the threshold on its own;
* a **debounce count** — movement is only declared after several consecutive
  samples deviate from the baseline, so a single noise spike is discarded.
"""

from __future__ import annotations

__all__ = ["CubeMoveDetector"]


class CubeMoveDetector:
    """Decide whether accel samples mean the cube was physically moved.

    Feed one magnitude per incoming ``ObjectAccel`` packet (which arrives
    roughly every 30 ms, see docs/functions.md); call :meth:`on_connection`
    whenever the cube (re)connects. :meth:`update` returns ``True`` exactly
    once per detected move, after which the detector re-baselines at the
    cube's current reading so it is ready for the next move.
    """

    def __init__(
        self,
        grace_s: float = 1.5,
        deviation: float = 6.0,
        debounce: int = 3,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            grace_s: Seconds after a connect during which samples are ignored.
            deviation: Deviation (m/s²) from the resting baseline that counts
                as movement. Default 6 is comfortably above noise (~0.5) but
                well below a real pick-up/carry (>10).
            debounce: Consecutive deviating samples required before a move is
                declared. At 30 ms/sample, 3 samples is ~90 ms.
            alpha: Exponential-smoothing weight for the resting-baseline
                follower (0 = never move, 1 = follow every sample).
        """
        self.grace_s = grace_s
        self.deviation = deviation
        self.debounce = max(1, debounce)
        self.alpha = alpha
        self._connected_at: float | None = None
        self._baseline: float | None = None
        self._over_count: int = 0

    def on_connection(self, now: float) -> None:
        """Call when the cube connects (or reconnects).

        Starts the grace period and discards the old baseline so the new
        resting position is learned from scratch.
        """
        self._connected_at = now
        self._baseline = None
        self._over_count = 0

    def update(self, now: float, accel_magnitude: float) -> bool:
        """Feed one accel magnitude (m/s²).

        Returns:
            True when the cube is considered to have moved (and the detector
            has already reset itself for the next resting position).
        """
        if self._connected_at is None:
            return False
        if now - self._connected_at < self.grace_s:
            return False

        if self._baseline is None:
            self._baseline = accel_magnitude
            return False

        deviation = accel_magnitude - self._baseline
        if abs(deviation) < self.deviation:
            # At rest (or near it): blend the baseline toward the reading and
            # cancel any pending trigger.
            self._baseline = (
                self.alpha * accel_magnitude + (1.0 - self.alpha) * self._baseline
            )
            self._over_count = 0
            return False

        self._over_count += 1
        if self._over_count < self.debounce:
            return False

        # Declared moved: reset so the next samples re-baseline at the cube's
        # new resting position (and we don't fire again while it is carried).
        self._baseline = None
        self._over_count = 0
        return True
