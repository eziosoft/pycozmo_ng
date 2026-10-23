"""Frame tracker — raw-frame stamping and FPS bookkeeping."""

import time

from pycozmo_ng.vision.frame_sink import FrameSink
from pycozmo_ng.vision.frame_state import FrameState


class FrameTracker:
    """Records raw camera frames and keeps the frame-rate counter.

    The raw MJPEG stream mirrors the camera exactly and runs for every
    frame, independent of the vision gates; this module owns both the
    payload bookkeeping and the counters behind them.
    """

    def __init__(self, frame_state: FrameState, sink: FrameSink) -> None:
        self.frame_state = frame_state
        self.sink = sink

    def stamp(self, image) -> None:
        """Record the raw frame (runs for every frame, even skipped ones)."""
        self.frame_state.latest_raw_frame = image
        self.frame_state.frame_count += 1
        self.frame_state.last_frame_time = time.time()
        # The raw MJPEG stream mirrors the camera exactly — it runs for
        # every frame, independent of the vision gates below.
        self.sink.push_raw_frame(image)

    def update_fps(self) -> None:
        """Update the FPS counter in frame state (once per raw frame)."""
        elapsed = time.time() - self.frame_state.start_time
        if elapsed > 1.0:
            self.frame_state.fps = self.frame_state.frame_count / elapsed
            self.frame_state.frame_count = 0
            self.frame_state.start_time = time.time()
