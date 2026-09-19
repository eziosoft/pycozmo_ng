"""Frame state — typed container for vision frame and detection state."""

import threading
import time


class FrameState:
    """Mutable container for frame lifecycle, detections, and camera state.

    Owned by FrameHandler; read by Brain.run() for staleness checks and
    by WebBridge for streaming.
    """

    def __init__(self) -> None:
        # Frame counters
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0

        # Camera lifecycle (COZMO-22)
        self.camera_active = False
        self.camera_enabled_at: float | None = None
        self.camera_resolution = None

        # Detections lock (COZMO-3)
        self.detections_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections: list = []
        self.latest_floor_image = None
        self.latest_occupancy_grid = None
        self.latest_occupancy_map_image = None

        # Staleness detection (COZMO-22)
        self.last_frame_time = time.time()
