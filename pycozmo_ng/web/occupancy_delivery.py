"""Occupancy PNG delivery — the single "does it need sending?" decision.

The occupancy grid changes a few cells per frame, so encoding and re-sending
the full PNG that often is wasted bandwidth — while a browser that never
receives the current grid keeps drawing a stale one. This object owns that
whole trip:

* :meth:`push` records a fresh grid, throttled to at most one encode per
  ``send_interval_s``;
* :meth:`take` answers the one question the state message asks — "does the
  browser need the occupancy PNG in *this* message?" — and marks it sent in
  the same call, so the changed/send/mark-sent decision exists exactly once
  (it used to be evaluated separately by ``state_builder`` and
  ``WebViewerServer``, which is how the two copies drift apart).
"""

from __future__ import annotations

import base64
import time

import cv2

__all__ = ["OCCUPANCY_SEND_INTERVAL_S", "OccupancyDelivery"]

#: Minimum seconds between two encodes of the grid (it changes incrementally).
OCCUPANCY_SEND_INTERVAL_S: float = 0.2

#: What a state message carries: ``(base64 PNG, [origin_x, origin_y])``.
OccupancyPayload = tuple[str, list[float]]


class OccupancyDelivery:
    """Throttled encoding, change detection, and mark-sent for the grid PNG."""

    def __init__(self, send_interval_s: float = OCCUPANCY_SEND_INTERVAL_S) -> None:
        self._send_interval_s = send_interval_s
        self._png: str | None = None
        self._origin: list[float] = [0.0, 0.0]
        self._seq = 0
        self._seq_sent = -1
        self._last_encode_time = 0.0

    def push(self, grid_image, origin_x: float = 0.0, origin_y: float = 0.0) -> None:
        """Record a fresh grid; encodes at most once per send interval."""
        now = time.time()
        if now - self._last_encode_time < self._send_interval_s:
            return
        self._last_encode_time = now

        _, buf = cv2.imencode(".png", grid_image)
        self._png = base64.b64encode(buf.tobytes()).decode("ascii")
        self._origin = [float(origin_x), float(origin_y)]
        self._seq += 1

    def take(self, force: bool = False) -> OccupancyPayload | None:
        """The payload for this state message, or None when nothing new exists.

        Handing out a payload marks it sent — the next call returns None
        until a newer grid arrives. ``force`` overrides the change check, for
        a full snapshot when a browser connects.
        """
        if self._png is None:
            return None
        if not force and self._seq == self._seq_sent:
            return None
        self._seq_sent = self._seq
        return self._png, self._origin
