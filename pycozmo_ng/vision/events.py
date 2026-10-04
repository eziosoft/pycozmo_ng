"""Detection event dispatcher — emits cube/face events from vision pipeline.

Separates event emission logic from the detection pipeline for better
testability and locality.
"""

from pycozmo_ng import event
from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.vision.cube_detector.detection_pipeline import CubeDetection
from pycozmo_ng.vision.face_detector import FaceDetection
from pycozmo_ng.vision.face_identity import FaceFrame


class DetectionEventDispatcher:
    """Dispatches cube and face detection events to the client.

    Cube transitions are tracked here; face presence transitions are
    owned by ``FaceIdentity`` and arrive with each frame as a
    :class:`FaceFrame` — this class only turns them into events.
    Event payloads use public IDs: cube IDs 1-3, face presence IDs 1+.
    """

    def __init__(self, obs_throttle_frames: int = 5):
        """
        Args:
            obs_throttle_frames: Emit observation event every N frames with detections.
        """
        self._last_cube_indices: set[int] = set()
        self._obs_frame_counter: int = 0
        self._face_obs_counter: int = 0
        self._obs_throttle = obs_throttle_frames

    def dispatch_cube_events(self, cli, detections: list[CubeDetection]) -> None:
        """
        Dispatch cube detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: List of current cube detections
        """
        current = {det.cube_index for det in detections if det.cube_index is not None}

        if current != self._last_cube_indices:
            new_indices = current - self._last_cube_indices
            if new_indices:
                new_detections = [d for d in detections if d.cube_index in new_indices]
                cli.dispatch(event.EvtCubeDetected, cli, new_detections)

            for cube_index in self._last_cube_indices - current:
                cli.dispatch(event.EvtCubeLost, cli, to_cube_id(cube_index))

            self._last_cube_indices = current

        if not detections:
            self._obs_frame_counter = 0
            return

        self._obs_frame_counter += 1
        if self._obs_frame_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtCubeObserved, cli, detections)

    def dispatch_face_events(self, cli, detections: list[FaceDetection], face_frame: FaceFrame) -> None:
        """Dispatch face detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: Faces in this frame (each with ``face_id`` assigned)
            face_frame: Presence transitions computed by ``FaceIdentity``
                this frame — nothing to do if it reports no transition.
        """
        if face_frame.started is not None:
            cli.dispatch(event.EvtFaceDetected, cli, detections)
        if face_frame.ended is not None:
            cli.dispatch(event.EvtFaceLost, cli, face_frame.ended)

        if not detections:
            self._face_obs_counter = 0
            return

        self._face_obs_counter += 1
        if self._face_obs_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtFaceObserved, cli, detections)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._last_cube_indices.clear()
        self._obs_frame_counter = 0
        self._face_obs_counter = 0
