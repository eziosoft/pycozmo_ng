"""Frame publisher — store the annotated frame and detections, push to the sink."""

from pycozmo_ng.vision.frame_sink import FrameSink
from pycozmo_ng.vision.frame_state import FrameState


class FramePublisher:
    """Store the annotated frame + detections and publish to the browser."""

    def __init__(self, frame_state: FrameState, sink: FrameSink) -> None:
        self.frame_state = frame_state
        self.sink = sink

    def publish(self, annotated_image, cube_detections) -> None:
        """Store the annotated frame + detections and push it to the browser."""
        self.frame_state.latest_annotated_frame = annotated_image
        with self.frame_state.detections_lock:
            self.frame_state.latest_detections = cube_detections

        if annotated_image is not None:
            self.sink.push_annotated_frame(annotated_image)
