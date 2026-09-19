"""Vision display — frame handlers, web viewer streaming, save."""

import time

from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.vision.detection_pipeline import DetectionPipeline
from pycozmo.vision.floor_pipeline import FloorPipeline
from pycozmo.vision.frame_state import FrameState


class VisionDisplay:
    """Handles frames, detection state, and web viewer streaming.

    Uses :class:`FrameState` for frame/detection data,
    :class:`CubeTrackingState` for cube last-seen timestamps,
    :class:`FloorPipeline` for floor projection results, and
    :class:`DetectionPipeline` for occupancy map image.
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
        # Set by Brain after construction (web_viewer, web, floor_pipeline are cross-domain)
        self.web_viewer = None
        self.web = None
        self.floor_pipeline: FloorPipeline | None = None

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.fs.latest_raw_frame = image
        self.fs.frame_count += 1
        self.fs.last_frame_time = time.time()

        if image is not None:
            self.web_viewer.push_frame(image)

        elapsed = time.time() - self.fs.start_time
        if elapsed > 1.0:
            self.fs.fps = self.fs.frame_count / elapsed
            self.fs.frame_count = 0
            self.fs.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.fs.latest_annotated_frame = annotated_image
        with self.fs.detections_lock:
            self.fs.latest_detections = detections
        self._stamp_detections_seen()
        if annotated_image is not None:
            self.web_viewer.push_annotated_frame(annotated_image)
        self.web.update_web_viewer()

    def _stamp_detections_seen(self):
        """Record the current frame index for every detected cube."""
        with self.fs.detections_lock:
            detected_cube_ids = {det.cube_id for det in self.fs.latest_detections}
        for cube_id in detected_cube_ids:
            self.ct.last_seen_frame[cube_id] = self.fs.frame_count

    def on_floor_projected_image(self, cli, floor_image, occupancy_grid):
        """Handler for floor-projected (bird's-eye) images."""
        self.fs.latest_floor_image = floor_image
        self.fs.latest_occupancy_grid = occupancy_grid
        self.web_viewer.push_floor_frame(floor_image)
        # Push occupancy map image if available
        if (
            self.detection_pipeline is not None
            and self.floor_pipeline is not None
            and self.floor_pipeline.occupancy_enabled
            and self.detection_pipeline.latest_occupancy_map_image is not None
        ):
            occ = self.floor_pipeline.occupancy_map
            self.web_viewer.push_occupancy_grid(
                self.detection_pipeline.latest_occupancy_map_image,
                occ.origin[0],
                occ.origin[1],
            )
        # Sync floor grid dimensions to browser on first frame
        if self.floor_pipeline is not None and self.floor_pipeline.floor_projector:
            fp = self.floor_pipeline.floor_projector
            self.web_viewer.set_floor_size(fp.floor_size[0], fp.floor_size[1])
            self.web_viewer.set_floor_center(fp.floor_center[0], fp.floor_center[1])

    def save_frames(self):
        """Save current frames to disk."""
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
