"""Detection event dispatcher — emits cube/face/charger events from vision pipeline.

Separates event emission logic from the detection pipeline for better
testability and locality.
"""

from pycozmo_ng import event
from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.vision.aruco_detector.aruco_detection import ArucoDetection
from pycozmo_ng.vision.charger_detector.charger_detection import ChargerDetection
from pycozmo_ng.vision.cube_detector.detection_pipeline import CubeDetection
from pycozmo_ng.vision.face_detector import FaceDetection
from pycozmo_ng.vision.face_identity import FaceFrame


class DetectionEventDispatcher:
    """Dispatches cube, face, charger and ArUco detection events to the client.

    Cube transitions are tracked here; face presence transitions are
    owned by ``FaceIdentity`` and arrive with each frame as a
    :class:`FaceFrame` — this class only turns them into events.
    The charger is a single static object, so only presence transitions
    are tracked (one visibility flag). ArUco markers are multi-object:
    the set of visible marker IDs is diffed per frame. Event payloads
    use public IDs: cube IDs 1-3, face presence IDs 1+.
    """

    def __init__(self, obs_throttle_frames: int = 5):
        """
        Args:
            obs_throttle_frames: Emit observation event every N frames with detections.
        """
        self._last_cube_indices: set[int] = set()
        self._obs_frame_counter: int = 0
        self._face_obs_counter: int = 0
        self._charger_visible: bool = False
        self._charger_obs_counter: int = 0
        self._last_aruco_ids: set[int] = set()
        self._aruco_obs_counter: int = 0
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

    def dispatch_charger_events(self, cli, detections: list[ChargerDetection]) -> None:
        """Dispatch charger marker events (detected, observed, lost).

        The charger is one static object, so tracking is a single visibility
        flag — there is no per-object ID to diff (COZMO-65).

        Args:
            cli: Client instance
            detections: Current frame's charger detections (0 or 1)
        """
        visible = bool(detections)

        if visible and not self._charger_visible:
            cli.dispatch(event.EvtChargerDetected, cli, detections)
        elif not visible and self._charger_visible:
            cli.dispatch(event.EvtChargerLost, cli)
        self._charger_visible = visible

        if not visible:
            self._charger_obs_counter = 0
            return

        self._charger_obs_counter += 1
        if self._charger_obs_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtChargerObserved, cli, detections)

    def dispatch_aruco_events(self, cli, detections: list[ArucoDetection]) -> None:
        """Dispatch ArUco marker events (detected, observed, lost).

        Multi-object tracking: the set of visible marker IDs is diffed
        against the previous frame — a marker appearing triggers
        EvtArucoDetected with just the new ones, one leaving triggers
        EvtArucoLost with its ID.

        Args:
            cli: Client instance
            detections: Current frame's ArUco detections (0..n)
        """
        current = {det.marker_id for det in detections}

        if current != self._last_aruco_ids:
            new_ids = current - self._last_aruco_ids
            if new_ids:
                cli.dispatch(event.EvtArucoDetected, cli, [d for d in detections if d.marker_id in new_ids])

            lost_ids = self._last_aruco_ids - current
            if lost_ids:
                cli.dispatch(event.EvtArucoLost, cli, sorted(lost_ids))

            self._last_aruco_ids = current

        if not current:
            self._aruco_obs_counter = 0
            return

        self._aruco_obs_counter += 1
        if self._aruco_obs_counter % self._obs_throttle == 0:
            cli.dispatch(event.EvtArucoObserved, cli, detections)

    def reset(self) -> None:
        """Reset all tracking state."""
        self._last_cube_indices.clear()
        self._obs_frame_counter = 0
        self._face_obs_counter = 0
        self._charger_visible = False
        self._charger_obs_counter = 0
        self._last_aruco_ids.clear()
        self._aruco_obs_counter = 0
