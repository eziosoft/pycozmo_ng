"""
Web streamer — MJPEG frame streaming + WebSocket broadcasting for the web viewer.

This module handles all web-facing frame publishing (camera, annotated, floor)
and WebSocket state broadcasting. It's a thin adapter over WebViewerServer.
"""

import time

from pycozmo.vision.frame_state import FrameState
from pycozmo.web.viewer_server import WebViewerServer


class WebStreamer:
    """Handles web streaming: MJPEG frames + WebSocket state broadcast.

    This replaces the web streaming logic that was in VisionDisplay.
    """

    def __init__(
        self,
        web_viewer: WebViewerServer,
        frame_state: FrameState,
    ):
        self.web_viewer = web_viewer
        self.frame_state = frame_state

    # --- Frame push methods (called from frame handlers) --------------------

    def push_raw_frame(self, image) -> None:
        """Publish raw camera frame for MJPEG streaming."""
        if image is not None:
            self.web_viewer.push_frame(image)
        self._update_fps()

    def push_annotated_frame(self, annotated_image, detections) -> None:
        """Publish annotated camera frame and update detections."""
        self.frame_state.latest_annotated_frame = annotated_image
        with self.frame_state.detections_lock:
            self.frame_state.latest_detections = detections
        if annotated_image is not None:
            self.web_viewer.push_annotated_frame(annotated_image)

    def push_floor_frame(
        self, floor_image, occupancy_grid, floor_pipeline=None
    ) -> None:
        """Publish floor-projected frame and occupancy grid."""
        self.frame_state.latest_floor_image = floor_image
        self.frame_state.latest_occupancy_grid = occupancy_grid
        self.web_viewer.push_floor_frame(floor_image)

        # Push occupancy map image if available
        if (
            self.frame_state.latest_occupancy_map_image is not None
            and floor_pipeline is not None
            and floor_pipeline.occupancy_enabled
        ):
            occ = floor_pipeline.occupancy_map
            self.web_viewer.push_occupancy_grid(
                self.frame_state.latest_occupancy_map_image,
                occ.origin[0],
                occ.origin[1],
            )

        # Sync floor grid dimensions to browser on first frame
        if floor_pipeline is not None and floor_pipeline.floor_projector:
            fp = floor_pipeline.floor_projector
            self.web_viewer.set_floor_size(fp.floor_size[0], fp.floor_size[1])
            self.web_viewer.set_floor_center(fp.floor_center[0], fp.floor_center[1])

    # --- Internal helpers ----------------------------------------------------

    def _update_fps(self) -> None:
        """Update FPS counter in frame state."""
        elapsed = time.time() - self.frame_state.start_time
        if elapsed > 1.0:
            self.frame_state.fps = self.frame_state.frame_count / elapsed
            self.frame_state.frame_count = 0
            self.frame_state.start_time = time.time()

    # --- WebSocket broadcast -------------------------------------------------

    def broadcast(self) -> None:
        """Trigger WebSocket state broadcast."""
        self.web_viewer.broadcast()

    # --- Lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Start the web viewer server."""
        self.web_viewer.start()

    def stop(self) -> None:
        """Stop the web viewer server."""
        self.web_viewer.close()

    def save_frames(self) -> None:
        """Save current frames to disk (debug helper)."""
        if self.frame_state.latest_raw_frame is None:
            print("No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        raw_filename = f"frame_raw_{timestamp}.png"
        self.frame_state.latest_raw_frame.save(raw_filename)
        print(f"Saved raw frame: {raw_filename}")

        if self.frame_state.latest_annotated_frame is not None:
            annotated_filename = f"frame_annotated_{timestamp}.png"
            self.frame_state.latest_annotated_frame.save(annotated_filename)
            print(f"Saved annotated frame: {annotated_filename}")

        with self.frame_state.detections_lock:
            n_det = len(self.frame_state.latest_detections)
        print(f"   Detections: {n_det}")
