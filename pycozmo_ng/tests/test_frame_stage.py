"""Tests for the FrameStage — the deep per-frame vision module.

FrameStage's interface is ``configure()`` + ``on_camera_image()``; every
test crosses that seam with fakes for the injected dependencies (world
map, frame sink, detectors) — no live robot, no Brain. What the display
receives is asserted on the ``FrameSink`` interface itself, through the
recording adapter below.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from PIL import Image

from pycozmo_ng import event, util
from pycozmo_ng.display.expressions import Happiness
from pycozmo_ng.vision.face_identity import FaceIdentity
from pycozmo_ng.vision.frame_sink import FrameSink
from pycozmo_ng.vision.frame_stage import FrameStage
from pycozmo_ng.vision.frame_state import FrameState


class RecordingFrameSink:
    """Recording adapter for the ``FrameSink`` seam — the test surface.

    Every call lands in ``order`` (shared, for sequence assertions) and in
    the per-call list it belongs to.
    """

    def __init__(self, order: list | None = None) -> None:
        self.order: list[str] = order if order is not None else []
        self.annotated: list = []
        self.floor: list = []
        self.faces: list = []
        self.occupancy: list = []
        self.grids: list = []
        self.broadcasts = 0

    def push_annotated_frame(self, image) -> None:
        self.order.append("publish")
        self.annotated.append(image)

    def push_floor_frame(self, image) -> None:
        self.order.append("floor_frame")
        self.floor.append(image)

    def push_face_image(self, pil_image) -> None:
        self.order.append("face")
        self.faces.append(pil_image)

    def push_occupancy_frame(self, image, origin_x_mm: float, origin_y_mm: float) -> None:
        self.order.append("occupancy")
        self.occupancy.append((image, origin_x_mm, origin_y_mm))

    def set_floor_grid(self, size_mm: tuple[float, float], center_mm: tuple[float, float]) -> None:
        self.order.append("grid")
        self.grids.append((size_mm, center_mm))

    def broadcast(self) -> None:
        self.order.append("broadcast")
        self.broadcasts += 1


def log_call(order: list[str], name: str, result=None):
    """Mock side effect that records its own call in a shared ordering list."""

    def side(*args, **kwargs):
        order.append(name)
        return result

    return side


def make_image() -> Image.Image:
    return Image.new("RGB", (8, 6), (255, 0, 0))


def make_face(face_id=None) -> SimpleNamespace:
    """Minimal face detection stand-in (identity is assigned by FaceIdentity)."""
    return SimpleNamespace(
        x=1,
        y=1,
        width=4,
        height=4,
        confidence=0.9,
        position_mm=(100.0, 0.0, 500.0),
        face_id=face_id,
    )


def make_cli(**overrides):
    """Minimal client stand-in: head angle, pose, anim controller, dispatch."""
    cli = SimpleNamespace(
        head_angle=SimpleNamespace(radians=0.5),
        pose=object(),
        anim_controller=SimpleNamespace(
            set_face_expression=Mock(),
            clear_face_expression=Mock(),
            latest_face_image=None,
        ),
        dispatch=Mock(),
    )
    for key, value in overrides.items():
        setattr(cli, key, value)
    return cli


def make_stage(**configure_overrides) -> SimpleNamespace:
    """FrameStage with fakes for every injected dependency."""
    fs = FrameState()
    world_map = Mock()
    sink: FrameSink = RecordingFrameSink()

    detection_pipeline = Mock()
    detection_pipeline.robot_center_calibration = SimpleNamespace(offset_mm=0.0, last_result=None)
    detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=0)]
    detection_pipeline.detect_charger.return_value = []
    detection_pipeline.detect_aruco.return_value = []
    detection_pipeline.detect_faces.return_value = []
    detection_pipeline.get_cube_measurements.return_value = [
        {
            "cube_index": 0,
            "position": (100.0, 0.0, 15.0),
            "rotation_matrix": [[1.0, 0.0, 0.0]] * 3,
            "confidence": 0.9,
            "num_faces": 2,
            "rotation_source": "pnp",
        }
    ]

    annotation_engine = Mock()
    annotation_engine.detect_and_annotate.return_value = make_image()

    event_dispatcher = Mock()
    face_identity = FaceIdentity()

    floor_pipeline = Mock()
    floor_img = object()
    occ_grid = object()
    occ_map_image = object()
    floor_pipeline.process.return_value = (floor_img, occ_grid, occ_map_image)
    floor_pipeline.occupancy_enabled = True
    floor_pipeline.occupancy_map.origin = (10, 20)
    floor_pipeline.floor_projector.floor_size = (600.0, 600.0)
    floor_pipeline.floor_projector.floor_center = (200.0, 0.0)

    stage = FrameStage(fs, world_map, sink)
    stage.configure(
        detection_pipeline=detection_pipeline,
        annotation_engine=annotation_engine,
        event_dispatcher=event_dispatcher,
        face_identity=face_identity,
        floor_pipeline=floor_pipeline,
        **configure_overrides,
    )
    return SimpleNamespace(
        stage=stage,
        fs=fs,
        world_map=world_map,
        sink=sink,
        detection_pipeline=detection_pipeline,
        annotation_engine=annotation_engine,
        event_dispatcher=event_dispatcher,
        face_identity=face_identity,
        floor_pipeline=floor_pipeline,
        floor_img=floor_img,
        occ_grid=occ_grid,
        occ_map_image=occ_map_image,
    )


def test_full_sequence_updates_every_sink():
    h = make_stage()
    cli = make_cli()
    image = make_image()

    h.stage.on_camera_image(cli, image)

    # Raw frame stamped
    assert h.fs.latest_raw_frame is image
    assert h.fs.frame_count == 1

    # Detection ran with the robot's head angle
    h.detection_pipeline.detect_cubes.assert_called_once()
    assert h.detection_pipeline.detect_cubes.call_args.args[1] == 0.5

    # Annotated frame + detections stored
    assert h.fs.latest_annotated_frame is not None
    assert h.fs.latest_detections == h.detection_pipeline.detect_cubes.return_value

    # Published through the frame sink (the display-side seam)
    assert len(h.sink.annotated) == 1

    # Events dispatched (face dispatch carries the identity's FaceFrame)
    h.event_dispatcher.dispatch_cube_events.assert_called_once()
    h.event_dispatcher.dispatch_face_events.assert_called_once()
    assert len(h.event_dispatcher.dispatch_face_events.call_args.args) == 3

    # World map fed with the cube measurement (head angle passed through)
    h.world_map.update_cube_measurement.assert_called_once()
    assert h.world_map.update_cube_measurement.call_args.args[4] == 0.5
    assert h.sink.broadcasts == 1

    # Floor: event dispatched, image + occupancy + grid dims published
    cli.dispatch.assert_called_once()
    assert cli.dispatch.call_args.args[0] is event.EvtFloorProjectedImage
    assert h.sink.floor == [h.floor_img]
    assert h.sink.occupancy == [(h.occ_map_image, 10, 20)]
    assert h.sink.grids == [((600.0, 600.0), (200.0, 0.0))]


def test_steps_run_in_order():
    """annotate → publish → dispatch → feed world map → floor."""
    h = make_stage()
    order = h.sink.order
    h.annotation_engine.detect_and_annotate.side_effect = log_call(order, "annotate", make_image())
    h.event_dispatcher.dispatch_cube_events.side_effect = log_call(order, "dispatch")
    h.world_map.update_cube_measurement.side_effect = log_call(order, "feed")
    h.floor_pipeline.process.side_effect = log_call(order, "floor", (h.floor_img, h.occ_grid, h.occ_map_image))

    h.stage.on_camera_image(make_cli(), make_image())

    assert order == [
        "annotate",
        "publish",
        "dispatch",
        "feed",
        "broadcast",
        "floor",
        "floor_frame",
        "occupancy",
        "grid",
    ]


def test_unconfigured_stage_is_inert():
    """Before configure() a frame is stamped but nothing downstream runs."""
    sink = RecordingFrameSink()
    stage = FrameStage(FrameState(), Mock(), sink)
    cli = make_cli()
    image = make_image()

    stage.on_camera_image(cli, image)

    assert stage.fs.latest_raw_frame is image
    assert stage.fs.frame_count == 1
    assert sink.annotated == []
    assert sink.broadcasts == 0


def test_frame_gate_skips_processing_but_still_stamps():
    h = make_stage(process_every_n_frames=2)
    cli = make_cli()

    h.stage.on_camera_image(cli, make_image())  # frame 1 — skipped
    h.detection_pipeline.detect_cubes.assert_not_called()
    assert h.fs.latest_raw_frame is not None  # raw frame still recorded
    assert h.fs.frame_count == 1

    h.stage.on_camera_image(cli, make_image())  # frame 2 — processed
    h.detection_pipeline.detect_cubes.assert_called_once()


def test_face_gate_runs_every_nth_frame():
    h = make_stage(face_detect_every_n_frames=2)
    cli = make_cli()

    h.stage.on_camera_image(cli, make_image())
    h.detection_pipeline.detect_faces.assert_not_called()
    # Not a due frame — expression must not be touched at all
    cli.anim_controller.clear_face_expression.assert_not_called()

    h.stage.on_camera_image(cli, make_image())
    h.detection_pipeline.detect_faces.assert_called_once()


def test_face_detection_sets_happiness_and_clears_when_gone():
    h = make_stage()
    cli = make_cli()

    h.detection_pipeline.detect_faces.return_value = [make_face()]
    h.stage.on_camera_image(cli, make_image())
    (expr,) = cli.anim_controller.set_face_expression.call_args.args
    assert isinstance(expr, Happiness)
    cli.anim_controller.clear_face_expression.assert_not_called()

    h.detection_pipeline.detect_faces.return_value = []
    h.stage.on_camera_image(cli, make_image())
    cli.anim_controller.clear_face_expression.assert_called_once()


def test_face_identity_feeds_world_map_and_forgets_on_loss():
    h = make_stage()
    cli = make_cli()

    h.detection_pipeline.detect_faces.return_value = [make_face()]
    h.stage.on_camera_image(cli, make_image())

    # The world map gets the identity-assigned epoch ID, not a hardcoded 0
    args = h.world_map.update_face_measurement.call_args
    assert args.args[0] == 1
    assert args.kwargs["label"] == "Face 1"

    h.detection_pipeline.detect_faces.return_value = []
    h.stage.on_camera_image(cli, make_image())

    h.world_map.forget_face.assert_called_once_with(1)
    assert h.face_identity.current_id is None


def test_face_identity_survives_skipped_face_frames():
    """A frame skip is 'not measured' — presence must not flap."""
    h = make_stage(face_detect_every_n_frames=2)
    cli = make_cli()
    face = make_face()
    h.detection_pipeline.detect_faces.return_value = [face]

    h.stage.on_camera_image(cli, make_image())  # frame 1 — detection skipped
    assert h.face_identity.current_id is None

    h.stage.on_camera_image(cli, make_image())  # frame 2 — face seen
    first_id = h.face_identity.current_id
    assert first_id == 1
    assert face.face_id == first_id

    h.stage.on_camera_image(cli, make_image())  # frame 3 — skipped: hold
    assert h.face_identity.current_id == first_id
    h.world_map.forget_face.assert_not_called()

    h.stage.on_camera_image(cli, make_image())  # frame 4 — same epoch continues
    assert h.face_identity.current_id == first_id


def test_fps_updates_after_one_second():
    h = make_stage()
    h.fs.start_time = time.time() - 2.0
    h.fs.frame_count = 9  # _stamp_raw_frame adds one → 10 over 2 s

    h.stage.on_camera_image(make_cli(), make_image())

    assert h.fs.fps > 0
    assert h.fs.frame_count == 0
    assert time.time() - h.fs.start_time < 1.0


def test_distance_rings_drawn_on_the_frame_before_annotation():
    """Rings land under the detection overlays — same image, earlier in the sequence."""
    h = make_stage(distance_rings=Mock())
    order = h.sink.order
    h.stage.distance_rings.draw.side_effect = log_call(order, "rings")
    h.annotation_engine.detect_and_annotate.side_effect = log_call(order, "annotate", make_image())

    h.stage.on_camera_image(make_cli(), make_image())

    assert order.index("rings") < order.index("annotate")
    rings_image, head_angle = h.stage.distance_rings.draw.call_args.args
    assert head_angle == 0.5  # the robot's head angle, same as detection
    assert rings_image is h.annotation_engine.detect_and_annotate.call_args.args[0]


def test_distance_rings_are_optional():
    h = make_stage()
    assert h.stage.distance_rings is None

    h.stage.on_camera_image(make_cli(), make_image())  # no rings, no crash

    assert len(h.sink.annotated) == 1


def test_processing_error_is_logged_not_raised():
    h = make_stage()
    h.detection_pipeline.detect_cubes.side_effect = ValueError("boom")

    h.stage.on_camera_image(make_cli(), make_image())  # must not raise

    h.annotation_engine.detect_and_annotate.assert_not_called()


def test_floor_disabled_skips_floor_sinks():
    h = make_stage()
    h.stage.floor_pipeline = None
    cli = make_cli()

    h.stage.on_camera_image(cli, make_image())

    cli.dispatch.assert_not_called()
    assert h.sink.floor == []
    assert h.sink.occupancy == []
    # …but the rest of the sequence still ran
    assert len(h.sink.annotated) == 1


def test_charger_detections_flow_to_events_and_world_map():
    """Charger detections are annotated, dispatched, and fused (COZMO-65)."""
    h = make_stage()
    charger = SimpleNamespace(
        x=1,
        y=1,
        width=5,
        height=4,
        confidence=0.8,
        distance=0.1,
        position_mm=np.array([10.0, 0.0, 100.0]),
        rotation_matrix=np.eye(3),
    )
    h.detection_pipeline.detect_charger.return_value = [charger]

    h.stage.on_camera_image(make_cli(), make_image())

    h.detection_pipeline.detect_charger.assert_called_once()
    # Same head angle as the cube pass
    assert h.detection_pipeline.detect_charger.call_args.args[1] == 0.5

    # …and handed to the dispatcher and the annotation engine (5th arg)
    assert h.annotation_engine.detect_and_annotate.call_args.args[4] == [charger]
    assert h.event_dispatcher.dispatch_charger_events.call_args.args[1] == [charger]
    h.world_map.update_charger_measurement.assert_called_once()
    # position, rotation, confidence, head_angle — same head angle as cubes
    args = h.world_map.update_charger_measurement.call_args.args
    assert np.allclose(args[1], np.eye(3))  # marker pose, not position-only (COZMO-67)
    assert args[3] == 0.5


# ---------------------------------------------------------------------------
# Cube motion gate — false positives while driving
# ---------------------------------------------------------------------------


def make_driving_cli(left_mmps: float, right_mmps: float, head_angle: float = 0.5):
    """Client stand-in with wheel speeds (the gate's body-yaw source)."""
    return make_cli(
        left_wheel_speed=util.Speed(mmps=left_mmps),
        right_wheel_speed=util.Speed(mmps=right_mmps),
        head_angle=SimpleNamespace(radians=head_angle),
    )


def test_motion_gate_drops_unknown_cube_while_turning():
    """A blurred blob matching a template must not enter the map while turning."""
    h = make_stage()
    h.world_map.get_cubes.return_value = {}
    h.detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=2)]

    # Wheels counter-rotate: 100 mm/s each way over a 45 mm track = 4.4 rad/s.
    h.stage.on_camera_image(make_driving_cli(-100.0, 100.0), make_image())

    assert h.fs.latest_detections == []
    h.detection_pipeline.restrict_cubes.assert_called_once_with(set())


def test_motion_gate_keeps_known_cube_while_turning():
    """An already-tracked cube keeps updating — only new IDs are refused."""
    h = make_stage()
    h.world_map.get_cubes.return_value = {0: object()}
    h.detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=0)]

    h.stage.on_camera_image(make_driving_cli(-100.0, 100.0), make_image())

    assert [d.cube_index for d in h.fs.latest_detections] == [0]
    h.detection_pipeline.restrict_cubes.assert_called_once_with({0})


def test_motion_gate_inactive_when_still():
    h = make_stage()
    h.world_map.get_cubes.return_value = {}
    h.detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=1)]

    h.stage.on_camera_image(make_driving_cli(0.0, 0.0), make_image())

    assert [d.cube_index for d in h.fs.latest_detections] == [1]
    h.detection_pipeline.restrict_cubes.assert_not_called()


def test_motion_gate_trips_on_fast_head_motion():
    """Head tilt blurs the frame too — the firmware gates it at 10 deg/s."""
    h = make_stage()
    h.world_map.get_cubes.return_value = {}
    h.detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=1)]

    # Previous frame 20 ms ago at a different head angle => 0.4 rad / 0.02 s.
    h.stage._last_head_angle = 0.0
    h.stage._last_head_time = time.monotonic() - 0.02

    h.stage.on_camera_image(make_driving_cli(0.0, 0.0, head_angle=0.4), make_image())

    assert h.fs.latest_detections == []
    h.detection_pipeline.restrict_cubes.assert_called_once_with(set())


def test_motion_gate_thresholds_are_configurable():
    h = make_stage(cube_body_rate_gate_radps=10.0)
    h.world_map.get_cubes.return_value = {}
    h.detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_index=1)]

    # 4.4 rad/s is below the raised threshold, so the cube is kept.
    h.stage.on_camera_image(make_driving_cli(-100.0, 100.0), make_image())

    assert [d.cube_index for d in h.fs.latest_detections] == [1]
