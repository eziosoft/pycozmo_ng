"""Frame state — typed container for vision frame and detection state."""

import threading
import time


class FrameState:
    """Mutable container for frame lifecycle and detection state.

    Written by :class:`~pycozmo_ng.vision.frame_stage.FrameStage` (frames,
    detections); read by ``Brain.run()`` for staleness checks and by the
    ``Cozmo`` facade for frame access. Camera lifecycle state lives in
    :class:`~pycozmo_ng.camera.service.CameraService`.

    :attr:`frame_lock` protects the scalar lifecycle state;
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

        # Detections lock (COZMO-3) — protects latest_detections only
        self.detections_lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        self.latest_detections: list = []

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
