"""World map feeder — frame measurements into the world map, then wake the browser."""

from typing import cast

from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.vision.aruco_detector.aruco_detection import ArucoDetection
from pycozmo_ng.vision.charger_detector.charger_detection import ChargerDetection
from pycozmo_ng.vision.cube_detector.detection_pipeline import DetectionPipeline
from pycozmo_ng.vision.face_detector import FaceDetection
from pycozmo_ng.vision.face_identity import FaceFrame
from pycozmo_ng.vision.frame_sink import FrameSink


class WorldMapFeeder:
    """Feeds per-frame measurements into the world map and wakes the browser.

    Face identities that ended this frame are dropped from the map. The
    charger and ArUco markers are stationary — their poses are fused but
    never forgotten.
    """

    def __init__(
        self,
        detection_pipeline: DetectionPipeline | None,
        world_map: WorldMap | None,
        sink: FrameSink,
    ) -> None:
        self.detection_pipeline = detection_pipeline
        self.world_map = world_map
        self.sink = sink

    def feed(
        self,
        cli,
        face_detections: list[FaceDetection],
        face_frame: FaceFrame,
        charger_detections: list[ChargerDetection] | None = None,
        aruco_detections: list[ArucoDetection] | None = None,
    ) -> None:
        """Feed cube, face, charger, and ArUco measurements into the world map."""
        if not self.detection_pipeline or self.world_map is None:
            return

        head_angle = cli.head_angle.radians if cli is not None else 0.0

        # Read the session camera lateral offset (frozen by robot-center calibration)
        robot_center = self.detection_pipeline.robot_center_calibration
        lateral_offset = robot_center.offset_mm if robot_center is not None else 0.0

        for m in self.detection_pipeline.get_cube_measurements():
            self.world_map.update_cube_measurement(
                m["cube_index"],
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
                m.get("num_faces"),
                camera_lateral_offset_mm=lateral_offset,
                rotation_source=m.get("rotation_source"),
            )

        # Feed face measurements (identity assigned earlier this frame),
        # then drop presence epochs that ended.
        for face_det in face_detections:
            self.world_map.update_face_measurement(
                cast(int, face_det.face_id),
                face_det.position_mm,
                face_det.confidence,
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
                label=f"Face {face_det.face_id}",
            )
        if face_frame.ended is not None:
            self.world_map.forget_face(face_frame.ended)

        # Feed charger marker measurements (single static landmark — no
        # forgetting on visibility loss, COZMO-65; pose carries the dock
        # facing for alignment, COZMO-67).
        for charger_det in charger_detections or []:
            self.world_map.update_charger_measurement(
                charger_det.position_mm,
                charger_det.rotation_matrix,
                charger_det.confidence,
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
            )

        # Feed ArUco marker measurements (static printed landmarks keyed
        # by ID — no forgetting on visibility loss, like the charger).
        for aruco_det in aruco_detections or []:
            self.world_map.update_aruco_measurement(
                aruco_det.marker_id,
                aruco_det.position_mm,
                aruco_det.rotation_matrix,
                aruco_det.confidence,
                head_angle,
                aruco_det.size_mm,
                camera_lateral_offset_mm=lateral_offset,
            )

        # Push procedural face image to the browser
        anim_ctrl = getattr(cli, "anim_controller", None)
        if anim_ctrl is not None:
            face_im = getattr(anim_ctrl, "latest_face_image", None)
            if face_im is not None:
                self.sink.push_face_image(face_im)

        self.sink.broadcast()
