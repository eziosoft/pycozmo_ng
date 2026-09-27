"""Tests for the FrameStage — the deep per-frame vision module.

FrameStage's interface is ``configure()`` + ``on_camera_image()``; every
test crosses that seam with fakes for the injected sinks (world map,
web viewer, detectors) — no live robot, no Brain.
"""

import time
from types import SimpleNamespace
from unittest.mock import Mock

from PIL import Image

from pycozmo import event
from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.display.expressions import Happiness
from pycozmo.vision.frame_stage import FrameStage
from pycozmo.vision.frame_state import FrameState


def make_image() -> Image.Image:
    return Image.new("RGB", (8, 6), (255, 0, 0))


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


def make_stage(**configure_overrides):
    """FrameStage with fakes for every injected dependency."""
    fs = FrameState()
    ct = CubeTrackingState()
    world_map = Mock()
    web_viewer = Mock()

    detection_pipeline = Mock()
    detection_pipeline.camera_lateral_offset_mm = 0.0
    detection_pipeline.robot_center_result = None
    detection_pipeline.detect_cubes.return_value = [SimpleNamespace(cube_id=0)]
    detection_pipeline.detect_faces.return_value = []
    detection_pipeline.get_cube_measurements.return_value = [
        {
            "cube_id": 0,
            "position": (100.0, 0.0, 15.0),
            "rotation_matrix": [[1.0, 0.0, 0.0]] * 3,
            "confidence": 0.9,
            "num_faces": 2,
            "rotation_source": "pnp",
        }
    ]
    detection_pipeline.get_face_measurements.return_value = []

    annotation_engine = Mock()
    annotation_engine.detect_and_annotate.return_value = make_image()

    event_dispatcher = Mock()

    floor_pipeline = Mock()
    floor_img = object()
    occ_grid = object()
    occ_map_image = object()
    floor_pipeline.process.return_value = (floor_img, occ_grid, occ_map_image)
    floor_pipeline.occupancy_enabled = True
    floor_pipeline.occupancy_map.origin = (10, 20)
    floor_pipeline.floor_projector.floor_size = (600.0, 600.0)
    floor_pipeline.floor_projector.floor_center = (200.0, 0.0)

    stage = FrameStage(fs, ct, world_map, web_viewer)
    stage.configure(
        detection_pipeline=detection_pipeline,
        annotation_engine=annotation_engine,
        event_dispatcher=event_dispatcher,
        floor_pipeline=floor_pipeline,
        **configure_overrides,
    )
    return SimpleNamespace(
        stage=stage,
        fs=fs,
        ct=ct,
        world_map=world_map,
        web_viewer=web_viewer,
        detection_pipeline=detection_pipeline,
        annotation_engine=annotation_engine,
        event_dispatcher=event_dispatcher,
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

    # Annotated frame + detections stored, cube stamped as seen
    assert h.fs.latest_annotated_frame is not None
    assert h.fs.latest_detections == h.detection_pipeline.detect_cubes.return_value
    assert h.ct.last_seen_frame[0] == h.fs.frame_count

    # Published to the browser sink
    h.web_viewer.push_annotated_frame.assert_called_once()

    # Events dispatched
    h.event_dispatcher.dispatch_cube_events.assert_called_once()
    h.event_dispatcher.dispatch_face_events.assert_called_once()

    # World map fed with the cube measurement (head angle passed through)
    h.world_map.update_cube_measurement.assert_called_once()
    assert h.world_map.update_cube_measurement.call_args.args[4] == 0.5
    h.web_viewer.broadcast.assert_called_once()

    # Floor: event dispatched, image + occupancy + grid dims published
    cli.dispatch.assert_called_once()
    assert cli.dispatch.call_args.args[0] is event.EvtFloorProjectedImage
    h.web_viewer.push_floor_frame.assert_called_once_with(h.floor_img)
    h.web_viewer.push_occupancy_grid.assert_called_once_with(
        h.occ_map_image, 10, 20
    )
    h.web_viewer.set_floor_size.assert_called_once_with(600.0, 600.0)
    h.web_viewer.set_floor_center.assert_called_once_with(200.0, 0.0)


def test_steps_run_in_order():
    """annotate → publish → dispatch → feed world map → floor."""
    h = make_stage()
    recorder = Mock()
    recorder.attach_mock(h.annotation_engine.detect_and_annotate, "annotate")
    recorder.attach_mock(h.web_viewer.push_annotated_frame, "publish")
    recorder.attach_mock(h.event_dispatcher.dispatch_cube_events, "dispatch")
    recorder.attach_mock(h.world_map.update_cube_measurement, "feed")
    recorder.attach_mock(h.floor_pipeline.process, "floor")

    h.stage.on_camera_image(make_cli(), make_image())

    names = [c[0] for c in recorder.mock_calls]
    assert names == ["annotate", "publish", "dispatch", "feed", "floor"]


def test_unconfigured_stage_is_inert():
    """Before configure() a frame is stamped but nothing downstream runs."""
    web_viewer = Mock()
    stage = FrameStage(FrameState(), CubeTrackingState(), Mock(), web_viewer)
    cli = make_cli()
    image = make_image()

    stage.on_camera_image(cli, image)

    assert stage.fs.latest_raw_frame is image
    assert stage.fs.frame_count == 1
    web_viewer.push_annotated_frame.assert_not_called()
    web_viewer.broadcast.assert_not_called()


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

    h.detection_pipeline.detect_faces.return_value = [object()]
    h.stage.on_camera_image(cli, make_image())
    (expr,) = cli.anim_controller.set_face_expression.call_args.args
    assert isinstance(expr, Happiness)
    cli.anim_controller.clear_face_expression.assert_not_called()

    h.detection_pipeline.detect_faces.return_value = []
    h.stage.on_camera_image(cli, make_image())
    cli.anim_controller.clear_face_expression.assert_called_once()


def test_suppressed_cube_reenters_after_fresh_frames():
    h = make_stage()
    h.ct.suppress["ids"] = {0}
    h.ct.suppress["fresh_frames_needed"] = 3
    cli = make_cli()

    h.stage.on_camera_image(cli, make_image())
    h.stage.on_camera_image(cli, make_image())
    h.world_map.update_cube_measurement.assert_not_called()

    h.stage.on_camera_image(cli, make_image())
    h.world_map.update_cube_measurement.assert_called_once()
    assert 0 not in h.ct.suppress["ids"]


def test_fps_updates_after_one_second():
    h = make_stage()
    h.fs.start_time = time.time() - 2.0
    h.fs.frame_count = 9  # _stamp_raw_frame adds one → 10 over 2 s

    h.stage.on_camera_image(make_cli(), make_image())

    assert h.fs.fps > 0
    assert h.fs.frame_count == 0
    assert time.time() - h.fs.start_time < 1.0


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
    h.web_viewer.push_floor_frame.assert_not_called()
    h.web_viewer.push_occupancy_grid.assert_not_called()
    # …but the rest of the sequence still ran
    h.web_viewer.push_annotated_frame.assert_called_once()
