"""Frame state — typed container for vision frame and detection state."""

import threading
import time


class FrameState:
    """Mutable container for frame lifecycle, detections, and camera state.

    Owned by FrameHandler; read by Brain.run() for staleness checks and
    by WebBridge for streaming.

    All mutable scalar state is protected by :attr:`frame_lock`.
    :attr:`detections_lock` protects :attr:`latest_detections` only.
    """

    def __init__(self) -> None:
        # Frame counters — protected by frame_lock
        self.frame_lock = threading.Lock()
        self.running = True
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        self.last_frame_time = time.time()

        # Camera lifecycle (COZMO-22) — protected by frame_lock
        self.camera_active = False
        self.camera_enabled_at: float | None = None
        self.camera_resolution = None

        # Detections lock (COZMO-3) — protects latest_detections only
        self.detections_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections: list = []
        self.latest_floor_image = None
        self.latest_occupancy_grid = None
        self.latest_occupancy_map_image = None

    @property
    def running(self) -> bool:
        with self.frame_lock:
            return self._running

    @running.setter
    def running(self, value: bool) -> None:
        with self.frame_lock:
            self._running = value

    @property
    def frame_count(self) -> int:
        with self.frame_lock:
            return self._frame_count

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        with self.frame_lock:
            self._frame_count = value

    @property
    def start_time(self) -> float:
        with self.frame_lock:
            return self._start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        with self.frame_lock:
            self._start_time = value

    @property
    def fps(self) -> float:
        with self.frame_lock:
            return self._fps

    @fps.setter
    def fps(self, value: float) -> None:
        with self.frame_lock:
            self._fps = value

    @property
    def last_frame_time(self) -> float:
        with self.frame_lock:
            return self._last_frame_time

    @last_frame_time.setter
    def last_frame_time(self, value: float) -> None:
        with self.frame_lock:
            self._last_frame_time = value

    @property
    def camera_active(self) -> bool:
        with self.frame_lock:
            return self._camera_active

    @camera_active.setter
    def camera_active(self, value: bool) -> None:
        with self.frame_lock:
            self._camera_active = value

    @property
    def camera_enabled_at(self) -> float | None:
        with self.frame_lock:
            return self._camera_enabled_at

    @camera_enabled_at.setter
    def camera_enabled_at(self, value: float | None) -> None:
        with self.frame_lock:
            self._camera_enabled_at = value
