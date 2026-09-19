"""Detection event dispatcher — emits cube/face events from vision pipeline.

Separates event emission logic from the detection pipeline for better
testability and locality.
"""

from pycozmo import event
from pycozmo.vision.detection_pipeline import CubeDetection
from pycozmo.vision.face_detector import FaceDetection


class DetectionEventDispatcher:
    """Dispatches cube and face detection events to the client.

    Tracks detection state across frames to emit EvtCubeDetected/EvtCubeLost/
    EvtCubeObserved (and face equivalents) only on state changes.
    """

    def __init__(self, obs_throttle_frames: int = 5):
        """
        Args:
            obs_throttle_frames: Emit observation event every N frames with detections.
        """
        self._last_cube_ids: set[int] = set()
        self._obs_frame_counter: int = 0
        self._last_face_ids: set[int] = set()
        self._face_obs_counter: int = 0
        self._obs_throttle = obs_throttle_frames

    def dispatch_cube_events(self, cli, detections: list[CubeDetection]) -> None:
        """
        Dispatch cube detection events (detected, observed, lost).

        Args:
            cli: Client instance
            detections: List of current cube detections
        """
        current_ids = {det.cube_id for det in detections if det.cube_id is not None}

        if current_ids != self._last_cube_ids:
            new_ids = current_ids - self._last_cube_ids
            if new_ids:
                new_detections = [d for d in detections if d.cube_id in new_ids]
                cli.dispatch(event.EvtCubeDetected, cli, new_detections)

            lost_ids = self._last_cube_ids - current_ids
            for cube_id in lost_ids:
                cli.dispatch(event.EvtCubeLost, cli, cube_id)

            self._last_cube_ids = current_ids

        if not detections:
            self._obs_frame_counter = 0
            return

        self._obs_frame_counter += 1
        if self._obs_frame_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtCubeObserved, cli, detections)

    def dispatch_face_events(self, cli, detections: list[FaceDetection]) -> None:
        """Dispatch face detection events (detected, observed, lost)."""
        current_ids = {getattr(d, "_face_id", None) for d in detections}
        current_ids.discard(None)

        if current_ids != self._last_face_ids:
            new_ids = current_ids - self._last_face_ids
            if new_ids:
                new_dets = [
                    d for d in detections if getattr(d, "_face_id", None) in new_ids
                ]
                cli.dispatch(event.EvtFaceDetected, cli, new_dets)

            lost_ids = self._last_face_ids - current_ids
            for fid in lost_ids:
                cli.dispatch(event.EvtFaceLost, cli, fid)

            self._last_face_ids = current_ids

        if not detections:
            self._face_obs_counter = 0
            return

        self._face_obs_counter += 1
        if self._face_obs_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtFaceObserved, cli, detections)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._last_cube_ids.clear()
        self._obs_frame_counter = 0
        self._last_face_ids.clear()
        self._face_obs_counter = 0
