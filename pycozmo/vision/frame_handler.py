"""
Frame handler — processes camera frames and updates FrameState.

Handles raw camera images, annotated images, and floor-projected images.
No web streaming logic — that's delegated to WebStreamer.
"""

import time

from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.vision.detection_pipeline import DetectionPipeline
from pycozmo.vision.floor_pipeline import FloorPipeline
from pycozmo.vision.frame_state import FrameState


class FrameHandler:
    """Processes camera frames and updates FrameState.

    Receives frames from event handlers, runs detection pipeline,
    and stores results in FrameState. Web streaming is handled by
    WebStreamer (separation of concerns).
    """

    def __init__(
        self,
        frame_state: FrameState,
        cube_tracking: CubeTrackingState,
        detection_pipeline: DetectionPipeline | None = None,
    ):
        self.fs = frame_state
        self.ct = cube_tracking
        self.detection_pipeline = detection_pipeline
        # Cross-domain refs set by Brain after construction
        self.floor_pipeline: FloorPipeline | None = None

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images (EvtNewRawCameraImage)."""
        self.fs.latest_raw_frame = image
        self.fs.frame_count += 1
        self.fs.last_frame_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images (EvtAnnotatedCameraImage)."""
        self.fs.latest_annotated_frame = annotated_image
        with self.fs.detections_lock:
            self.fs.latest_detections = detections
        self._stamp_detections_seen()

    def _stamp_detections_seen(self):
        """Record the current frame index for every detected cube."""
        frame_count = self.fs.frame_count
        with self.fs.detections_lock:
            detected_cube_ids = {det.cube_id for det in self.fs.latest_detections}
        for cube_id in detected_cube_ids:
            self.ct.last_seen_frame[cube_id] = frame_count

    def on_floor_projected_image(self, cli, floor_image, occupancy_grid):
        """Handler for floor-projected images (EvtFloorProjectedImage)."""
        self.fs.latest_floor_image = floor_image
        self.fs.latest_occupancy_grid = occupancy_grid

    def save_frames(self):
        """Save current frames to disk (debug helper)."""
        if self.fs.latest_raw_frame is None:
            print("No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        raw_filename = f"frame_raw_{timestamp}.png"
        self.fs.latest_raw_frame.save(raw_filename)
        print(f"Saved raw frame: {raw_filename}")

        if self.fs.latest_annotated_frame is not None:
            annotated_filename = f"frame_annotated_{timestamp}.png"
            self.fs.latest_annotated_frame.save(annotated_filename)
            print(f"Saved annotated frame: {annotated_filename}")

        with self.fs.detections_lock:
            n_det = len(self.fs.latest_detections)
        print(f"   Detections: {n_det}")