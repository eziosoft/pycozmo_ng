"""ArUco marker detection tests — config, pose-per-size, world map, events, viewer.

The core guarantee: each marker ID's configured physical size sets the
absolute scale of its PnP pose. The synthetic-scene tests render a real
4x4 marker into a known-geometry frame and assert the reported distance —
if the per-ID size plumbing breaks, these fail.

No live robot required (same convention as the rest of pycozmo_ng/tests).
"""

import asyncio
import json

import cv2
import numpy as np
import pytest
from aiohttp import web

from pycozmo_ng import event
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.vision.annotation import AnnotationEngine
from pycozmo_ng.vision.aruco_detector import (
    ArucoConfig,
    ArucoDetection,
    ArucoDetector,
    MarkerSizeRegistry,
    aruco_marker_png,
    load_aruco_config,
)
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.web.aruco_texture import ArucoTextureEndpoint
from pycozmo_ng.web.state_builder import build_state_json

# ---------------------------------------------------------------------------
# Synthetic scene helpers
# ---------------------------------------------------------------------------

FRAME_W, FRAME_H = 640, 480
FOCAL = 600.0


def camera_matrix() -> np.ndarray:
    """Known intrinsics so expected distances are computable in closed form."""
    return np.array(
        [
            [FOCAL, 0.0, FRAME_W / 2.0],
            [0.0, FOCAL, FRAME_H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def render_marker_frame(marker_id: int, marker_px: int) -> np.ndarray:
    """Render a marker of ``marker_px`` pixels centred on a bright frame.

    The marker's projected pixel size and the known focal length determine
    the expected distance: ``dist = size_mm * FOCAL / marker_px``.
    """
    png = aruco_marker_png(marker_id)
    marker = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert marker is not None, "marker PNG must decode"
    marker = cv2.resize(marker, (marker_px, marker_px), interpolation=cv2.INTER_NEAREST)

    frame = np.full((FRAME_H, FRAME_W, 3), 200, np.uint8)
    x0 = (FRAME_W - marker_px) // 2
    y0 = (FRAME_H - marker_px) // 2
    frame[y0 : y0 + marker_px, x0 : x0 + marker_px] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return frame


def make_detector(marker_sizes: dict[int, float], default_size: float = 50.0) -> ArucoDetector:
    config = ArucoConfig(dictionary="4X4_50", default_size_mm=default_size, marker_sizes_mm=marker_sizes)
    return ArucoDetector(camera_matrix(), np.zeros(5), config=config)


# ---------------------------------------------------------------------------
# Config: JSON size registry
# ---------------------------------------------------------------------------


class TestMarkerSizeConfig:
    def test_missing_file_returns_none(self, tmp_path):
        assert load_aruco_config(str(tmp_path / "nope.json")) is None

    def test_valid_file_parses_per_id_sizes(self, tmp_path):
        path = tmp_path / "aruco.json"
        path.write_text(
            json.dumps({"dictionary": "4X4_50", "default_size_mm": 50.0, "markers": {"0": 22.5, "7": 40.0}})
        )
        config = load_aruco_config(str(path))
        assert config is not None
        registry = MarkerSizeRegistry(config)
        assert registry.size_for_id(0) == 22.5
        assert registry.size_for_id(7) == 40.0
        assert registry.has_explicit_size(0)
        assert not registry.has_explicit_size(3)

    def test_unknown_id_falls_back_to_default(self, tmp_path):
        path = tmp_path / "aruco.json"
        path.write_text(json.dumps({"markers": {"0": 22.5}}))
        config = load_aruco_config(str(path))
        registry = MarkerSizeRegistry(config)
        assert registry.size_for_id(3) == 50.0

    def test_non_positive_size_rejected(self, tmp_path):
        path = tmp_path / "aruco.json"
        path.write_text(json.dumps({"markers": {"0": -1}}))
        with pytest.raises(ValueError):
            load_aruco_config(str(path))

    def test_wrong_root_type_rejected(self, tmp_path):
        path = tmp_path / "aruco.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(TypeError):
            load_aruco_config(str(path))

    def test_env_var_override(self, tmp_path, monkeypatch):
        from pycozmo_ng.vision.aruco_detector.aruco_config import ARUCO_CONFIG_ENV

        path = tmp_path / "custom.json"
        path.write_text(json.dumps({"markers": {"5": 33.0}}))
        monkeypatch.setenv(ARUCO_CONFIG_ENV, str(path))
        config = load_aruco_config()
        assert config is not None
        assert config.marker_sizes_mm[5] == 33.0


# ---------------------------------------------------------------------------
# Detector: ID decode + per-ID size sets pose scale
# ---------------------------------------------------------------------------


class TestArucoDetector:
    def test_detects_marker_with_configured_size(self):
        # 150px marker, size 22.5mm -> expected 22.5 * 600/150 = 90mm
        frame = render_marker_frame(0, 150)
        detections = make_detector({0: 22.5}).detect(frame)
        assert len(detections) == 1
        det = detections[0]
        assert det.marker_id == 0
        assert det.size_mm == 22.5
        assert det.position_mm[2] == pytest.approx(90.0, rel=0.05)

    def test_unsized_id_uses_default_size(self):
        # Same pixels, no configured size -> default 50mm scales to 200mm.
        # The whole point: wrong config = wrong distance, by the same factor.
        frame = render_marker_frame(1, 150)
        detections = make_detector({0: 22.5}, default_size=50.0).detect(frame)
        assert len(detections) == 1
        det = detections[0]
        assert det.marker_id == 1
        assert det.size_mm == 50.0
        assert det.position_mm[2] == pytest.approx(200.0, rel=0.05)

    def test_distance_ratio_matches_size_ratio(self):
        # Two markers at identical pixel size, sizes 22.5 vs 45 -> 2x distance.
        small = make_detector({0: 22.5, 1: 45.0})
        frame = render_marker_frame(0, 150)
        d0 = small.detect(frame)[0]
        frame = render_marker_frame(1, 150)
        d1 = small.detect(frame)[0]
        assert d1.position_mm[2] / d0.position_mm[2] == pytest.approx(2.0, rel=0.05)

    def test_no_markers_returns_empty(self):
        frame = np.full((FRAME_H, FRAME_W, 3), 128, np.uint8)
        assert make_detector({}).detect(frame) == []

    def test_marker_near_border_dropped(self):
        # Quad touching the border is clipped -> pose is garbage, dropped.
        frame = render_marker_frame(0, 150)
        shifted = np.roll(frame, 300, axis=1)  # marker half out of frame
        assert make_detector({0: 22.5}).detect(shifted) == []


# ---------------------------------------------------------------------------
# World map: fusion, persistence, reset
# ---------------------------------------------------------------------------


def fuse_marker(world_map: WorldMap, marker_id: int, size_mm: float, x: float = 0.0) -> None:
    world_map.update_aruco_measurement(
        marker_id,
        np.array([x, 0.0, 300.0]),
        np.eye(3),
        1.0,
        0.0,
        size_mm=size_mm,
    )


class TestWorldMapAruco:
    def test_marker_fused_and_serialized(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        state = wm.get_world_state()
        assert set(state["aruco_markers"]) == {0}
        assert state["aruco_markers"][0]["size_mm"] == 22.5
        assert len(state["aruco_markers"][0]["position"]) == 3

    def test_multiple_markers_kept_separately(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        fuse_marker(wm, 7, 40.0, x=50.0)
        markers = wm.get_aruco_markers()
        assert set(markers) == {0, 7}
        assert markers[0].size_mm == 22.5
        assert markers[7].size_mm == 40.0

    def test_marker_never_forgotten_without_reset(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        # No visibility-loss forgetting: state persists across many frames
        # with no new measurements.
        assert len(wm.get_aruco_markers()) == 1

    def test_reset_map_clears_markers(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        wm.reset_map()
        assert wm.get_aruco_markers() == {}

    def test_origin_change_clears_markers(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        wm.update_robot_pose(np.array([0.0, 0.0, 0.0]), 0.0, origin_id=1)
        wm.update_robot_pose(np.array([0.0, 0.0, 0.0]), 0.0, origin_id=2)
        assert wm.get_aruco_markers() == {}

    def test_repeated_fusion_smooths_position(self):
        wm = WorldMap(smoothing_alpha=0.5, rotation_alpha=0.5)
        fuse_marker(wm, 0, 22.5, x=0.0)
        fuse_marker(wm, 0, 22.5, x=100.0)
        # Camera x maps to world Y (Anki frame): sightings land at world
        # y = 0 and y = -100; confidence-weighted alpha 0.5 blends between.
        y = wm.get_aruco_markers()[0].position[1]
        assert -100.0 < y < 0.0


# ---------------------------------------------------------------------------
# Events: per-ID set diff
# ---------------------------------------------------------------------------


def make_aruco(marker_id: int) -> ArucoDetection:
    return ArucoDetection(
        marker_id=marker_id,
        size_mm=22.5,
        x=10,
        y=10,
        width=50,
        height=50,
        position_mm=np.zeros(3),
        rotation_matrix=np.eye(3),
    )


class FakeCli:
    def __init__(self):
        self.dispatched = []

    def dispatch(self, evt, *args):
        self.dispatched.append((evt, args))


class TestArucoEvents:
    def test_new_marker_dispatches_detected(self):
        cli, dispatcher = FakeCli(), DetectionEventDispatcher()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0), make_aruco(3)])
        kinds = [e for e, _ in cli.dispatched]
        assert kinds == [event.EvtArucoDetected]
        dets = cli.dispatched[0][1][1]
        assert {d.marker_id for d in dets} == {0, 3}

    def test_stable_visibility_no_transition(self):
        cli, dispatcher = FakeCli(), DetectionEventDispatcher()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])
        cli.dispatched.clear()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])
        assert cli.dispatched == []  # no transition, no event

    def test_marker_leaving_dispatches_lost_with_id(self):
        cli, dispatcher = FakeCli(), DetectionEventDispatcher()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0), make_aruco(3)])
        cli.dispatched.clear()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])
        kinds = [e for e, _ in cli.dispatched]
        assert kinds == [event.EvtArucoLost]
        assert cli.dispatched[0][1][1] == [3]

    def test_observed_throttled(self):
        cli, dispatcher = FakeCli(), DetectionEventDispatcher(obs_throttle_frames=2)
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])  # frame 1: detected only
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])  # frame 2: + observed
        kinds = [e for e, _ in cli.dispatched]
        assert event.EvtArucoObserved in kinds

    def test_reset_clears_tracking(self):
        cli, dispatcher = FakeCli(), DetectionEventDispatcher()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])
        dispatcher.reset()
        cli.dispatched.clear()
        dispatcher.dispatch_aruco_events(cli, [make_aruco(0)])
        assert [e for e, _ in cli.dispatched] == [event.EvtArucoDetected]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


class TestArucoAnnotation:
    def test_draw_does_not_crash(self):
        from PIL import Image

        image = Image.new("RGB", (320, 240), (128, 128, 128))
        det = make_aruco(0)
        det.corners = np.array([[10.0, 10.0], [60.0, 10.0], [60.0, 60.0], [10.0, 60.0]])
        annotated = AnnotationEngine().detect_and_annotate(image, [], [], None, None, [det])
        assert annotated.size == (320, 240)
        # Something magenta-ish was drawn (the marker overlay color)
        arr = np.asarray(annotated)
        assert (arr[..., 0] > 200).any() and (arr[..., 2] > 200).any()


# ---------------------------------------------------------------------------
# State builder: browser payload
# ---------------------------------------------------------------------------


class TestStateBuilderAruco:
    def test_markers_serialized_with_size(self):
        wm = WorldMap()
        fuse_marker(wm, 0, 22.5)
        state = json.loads(build_state_json(wm, {}, (600.0, 600.0), (200.0, 0.0), 0.0))
        markers = state["aruco_markers"]
        assert len(markers) == 1
        m = markers[0]
        assert m["id"] == 0
        assert m["size_mm"] == 22.5
        assert len(m["position"]) == 3
        assert len(m["rotation"]) == 3

    def test_no_markers_yields_empty_list(self):
        state = json.loads(build_state_json(WorldMap(), {}, (600.0, 600.0), (200.0, 0.0), 0.0))
        assert state["aruco_markers"] == []


# ---------------------------------------------------------------------------
# Texture endpoint
# ---------------------------------------------------------------------------


class FakeRequest:
    def __init__(self, query: dict):
        self.query = query


class TestArucoTextureEndpoint:
    def test_serves_png(self):
        ep = ArucoTextureEndpoint()
        resp = asyncio.run(ep.handle(FakeRequest({"id": "0"})))
        assert resp.content_type == "image/png"
        assert len(resp.body) > 100
        img = cv2.imdecode(np.frombuffer(resp.body, np.uint8), cv2.IMREAD_GRAYSCALE)
        assert img is not None and img.shape[0] == img.shape[1]

    def test_out_of_range_id_404(self):
        ep = ArucoTextureEndpoint()
        with pytest.raises(web.HTTPNotFound):
            asyncio.run(ep.handle(FakeRequest({"id": "9999"})))

    def test_non_integer_id_400(self):
        ep = ArucoTextureEndpoint()
        with pytest.raises(web.HTTPBadRequest):
            asyncio.run(ep.handle(FakeRequest({"id": "abc"})))

    def test_explicit_dictionary(self):
        ep = ArucoTextureEndpoint(dictionary="4X4_50")
        resp = asyncio.run(ep.handle(FakeRequest({"id": "1"})))
        assert resp.content_type == "image/png"
