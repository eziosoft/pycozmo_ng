"""Charger marker detection tests (COZMO-65).

The regression that started this work: the charger's rectangular marker
matched the cube template family and flapped cube IDs every second. The
marker now has its own detector, and both detectors carry a symmetric
``reject_markers`` guard so neither family claims the other's quads.

Detection tests run against the repository's real marker photo and the
cube face JPGs — no live robot required.
"""

import os
from typing import cast

import cv2
import numpy as np
import pytest
from PIL import Image

from pycozmo_ng import event
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.vision.annotation import AnnotationEngine
from pycozmo_ng.vision.calibration.camera_calibration import CameraCalibration
from pycozmo_ng.vision.charger_detector import (
    CHARGER_ASPECT_RANGE,
    CHARGER_HEIGHT_MM,
    CHARGER_WIDTH_MM,
    ChargerDetection,
    ChargerDetector,
    load_charger_template,
)
from pycozmo_ng.vision.cube_detector.cube_detector import CozmoCubeDetector
from pycozmo_ng.vision.cube_detector.detection_pipeline import (
    DetectionPipeline,
    load_marker_templates,
)
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.interfaces import (
    ChargerDetector as ChargerDetectorProtocol,
)
from pycozmo_ng.vision.interfaces import (
    CubeDetector,
    CubeFusionTracker,
)

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARKER_PATH = os.path.join(PKG_DIR, "vision", "charger_detector", "marker", "charger.png")
CUBE_FACES_DIR = os.path.join(PKG_DIR, "vision", "cube_detector", "cube_faces")


def camera_matrix(width: int, height: int) -> np.ndarray:
    """Plausible intrinsics for a small test frame (not robot calibration)."""
    f = float(max(width, height)) * 1.6
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def charger_template() -> np.ndarray:
    return load_charger_template()


@pytest.fixture(scope="module")
def cube_templates() -> list[np.ndarray]:
    return load_marker_templates()


@pytest.fixture(scope="module")
def marker_frame() -> np.ndarray:
    """The marker photo as a BGR frame (reproduces the original bug)."""
    frame = cv2.imread(MARKER_PATH, cv2.IMREAD_COLOR)
    assert frame is not None, f"marker asset missing: {MARKER_PATH}"
    return frame


def make_cube_detector(cube_templates, charger_template=None) -> CozmoCubeDetector:
    return CozmoCubeDetector(
        camera_matrix(92, 70),  # marker asset frame size
        np.zeros(5),
        cube_templates,
        use_clahe=True,
        denoise_strength=0,
        reject_markers=[charger_template] if charger_template is not None else None,
    )


def make_charger_detector(charger_template, cube_templates) -> ChargerDetector:
    return ChargerDetector(
        camera_matrix(92, 70),
        np.zeros(5),
        [charger_template],
        reject_markers=cube_templates,
        use_clahe=True,
        denoise_strength=0,
    )


class TestTemplate:
    def test_template_shape(self, charger_template):
        assert charger_template.shape == (32, 32)
        assert charger_template.dtype == np.uint8

    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError):
            load_charger_template("/nonexistent/charger.png")


class TestChargerDetector:
    def test_detects_marker_in_asset_frame(self, marker_frame, charger_template, cube_templates):
        detector = make_charger_detector(charger_template, cube_templates)
        results = detector.detect(marker_frame)

        assert len(results) == 1
        det = results[0]
        assert det["score"] > 0.68  # validated range on real frames: 0.71-0.84
        assert det["kind"] == "charger"
        # PnP succeeded and is facing the camera
        assert det["position"][2] > 0

    def test_result_carries_no_cube_identity(self, marker_frame, charger_template, cube_templates):
        detector = make_charger_detector(charger_template, cube_templates)
        results = detector.detect(marker_frame)
        assert results, "marker must be detected"
        for key in ("cube_index", "face_type", "rotation", "id"):
            assert key not in results[0]

    def test_aspect_gate_matches_marker(self, marker_frame, charger_template, cube_templates):
        detector = make_charger_detector(charger_template, cube_templates)
        results = detector.detect(marker_frame)
        assert results
        _, _, w, h = cv2.boundingRect(results[0]["corners"])
        aspect_min, aspect_max = CHARGER_ASPECT_RANGE
        assert aspect_min < w / h < aspect_max

    def test_configuration(self, charger_template, cube_templates):
        detector = make_charger_detector(charger_template, cube_templates)
        assert detector.pnp_flags == cv2.SOLVEPNP_IPPE
        assert detector.aspect_range == CHARGER_ASPECT_RANGE
        # Rectangular object points: +-23/2 in x, +-15/2 in y
        xs, ys = detector.obj_points[:, 0], detector.obj_points[:, 1]
        assert set(xs.tolist()) == {-CHARGER_WIDTH_MM / 2, CHARGER_WIDTH_MM / 2}
        assert set(ys.tolist()) == {-CHARGER_HEIGHT_MM / 2, CHARGER_HEIGHT_MM / 2}
        assert detector.max_detections == 1  # there is only one charger

    def test_rejects_all_cube_faces(self, charger_template, cube_templates):
        """A cube face is never reported as the charger (COZMO-65)."""
        detector = make_charger_detector(charger_template, cube_templates)
        faces = sorted(os.path.join(CUBE_FACES_DIR, f) for f in os.listdir(CUBE_FACES_DIR) if f.endswith(".jpg"))
        assert len(faces) == 18
        for path in faces:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            assert frame is not None, path
            assert detector.detect(frame) == [], f"charger matched cube face {os.path.basename(path)}"


class TestCubeDetectorRejection:
    """The original bug, and the guard that fixes it (COZMO-65)."""

    def test_without_reject_markers_marker_is_phantom_cube(self, marker_frame, cube_templates):
        """Documents the regression: the marker used to match a cube template."""
        detector = make_cube_detector(cube_templates)
        results = detector.detect(marker_frame)
        assert results, "expected the unguarded cube detector to phantom-detect the marker"
        assert 0 <= results[0]["cube_index"] <= 2  # it claimed a cube ID

    def test_with_reject_markers_marker_is_not_a_cube(self, marker_frame, cube_templates, charger_template):
        detector = make_cube_detector(cube_templates, charger_template)
        assert detector.detect(marker_frame) == []

    def test_guard_does_not_break_cube_detection(self, cube_templates, charger_template):
        """All 18 cube faces still resolve to their correct cube index."""
        detector = make_cube_detector(cube_templates, charger_template)
        faces = sorted(os.path.join(CUBE_FACES_DIR, f) for f in os.listdir(CUBE_FACES_DIR) if f.endswith(".jpg"))
        for path in faces:
            frame = cv2.imread(path, cv2.IMREAD_COLOR)
            assert frame is not None, path
            results = detector.detect(frame)
            expected = int(os.path.basename(path).split("_")[0].replace("cube", "")) - 1
            assert results, f"cube face lost: {os.path.basename(path)}"
            assert results[0]["cube_index"] == expected, os.path.basename(path)


class TestEventDispatcher:
    class FakeCli:
        def __init__(self):
            self.dispatched = []

        def dispatch(self, evt, *args):
            self.dispatched.append(evt)

    def make_detections(self) -> list[ChargerDetection]:
        return [
            ChargerDetection(
                x=1,
                y=1,
                width=5,
                height=4,
                confidence=0.8,
                distance=0.1,
                position_mm=np.zeros(3),
            )
        ]

    def test_presence_transitions(self):
        dispatcher = DetectionEventDispatcher(obs_throttle_frames=2)
        cli = self.FakeCli()
        det = self.make_detections()

        dispatcher.dispatch_charger_events(cli, det)  # first sighting
        assert cli.dispatched == [event.EvtChargerDetected]

        dispatcher.dispatch_charger_events(cli, det)  # still visible → throttled observed
        assert cli.dispatched == [event.EvtChargerDetected, event.EvtChargerObserved]

        dispatcher.dispatch_charger_events(cli, [])  # gone
        assert cli.dispatched == [event.EvtChargerDetected, event.EvtChargerObserved, event.EvtChargerLost]

        dispatcher.dispatch_charger_events(cli, [])  # still gone → nothing new
        assert len(cli.dispatched) == 3

        dispatcher.dispatch_charger_events(cli, det)  # back in view
        assert cli.dispatched[-1] is event.EvtChargerDetected

    def test_reset_clears_presence(self):
        dispatcher = DetectionEventDispatcher()
        cli = self.FakeCli()
        det = self.make_detections()

        dispatcher.dispatch_charger_events(cli, det)
        dispatcher.dispatch_charger_events(cli, [])
        dispatcher.reset()

        dispatcher.dispatch_charger_events(cli, det)
        assert cli.dispatched[-1] is event.EvtChargerDetected


class TestWorldMap:
    def test_charger_fusion_and_reset(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)

        assert wm.get_charger() is None
        assert wm.get_world_state()["charger"] is None

        wm.update_charger_measurement(np.array([100.0, 0.0, 300.0]), 0.9, 0.0)
        first = wm.get_charger()
        assert first is not None
        assert first.age == 0

        # Stationary landmark: a second identical measurement fuses in place
        wm.update_charger_measurement(np.array([100.0, 0.0, 300.0]), 0.9, 0.0)
        second = wm.get_charger()
        assert second.age == 1
        assert np.allclose(second.position, first.position)

        state = wm.get_world_state()["charger"]
        assert state is not None
        assert len(state["position"]) == 3
        assert state["confidence"] == 0.9

        wm.reset_map()
        assert wm.get_charger() is None
        assert wm.get_world_state()["charger"] is None

    def test_charger_survives_visibility_loss(self):
        """No forgetting on visibility loss — the charger does not move."""
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_charger_measurement(np.array([100.0, 0.0, 300.0]), 0.9, 0.0)

        # No further measurements (marker left the frame)
        assert wm.get_charger() is not None

    def test_localization_origin_change_clears_charger(self):
        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_charger_measurement(np.array([100.0, 0.0, 300.0]), 0.9, 0.0)
        assert wm.get_charger() is not None

        # Pick-up / put-down re-anchors the world frame → stale pose dropped
        wm.update_robot_pose([10.0, 0.0, 0.0], 0.0, origin_id=2, record_path=False)
        assert wm.get_charger() is None


class TestDetectionPipeline:
    class FakeChargerDetector:
        def detect(self, img_bgr):
            return [
                {
                    "score": 0.8,
                    "kind": "charger",
                    "corners": np.array([[[10, 10]], [[80, 10]], [[80, 60]], [[10, 60]]]),
                    "position": np.array([5.0, 0.0, 100.0]),
                }
            ]

    def make_pipeline(self, charger_detector: ChargerDetectorProtocol | None = None) -> DetectionPipeline:
        # The cube slots are irrelevant to these tests — only the charger
        # detector is exercised, so stub the rest.
        return DetectionPipeline(
            camera_calibration=cast(CameraCalibration, None),
            cube_detector=cast(CubeDetector, None),
            cube_tracker=cast(CubeFusionTracker, None),
            charger_detector=charger_detector,
        )

    def test_without_detector_returns_empty(self):
        assert self.make_pipeline().detect_charger(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_disabled_returns_empty(self):
        pipeline = self.make_pipeline(self.FakeChargerDetector())
        pipeline.enable_charger_detection = False
        assert pipeline.detect_charger(np.zeros((10, 10, 3), dtype=np.uint8)) == []

    def test_converts_raw_result_to_detection(self):
        pipeline = self.make_pipeline(self.FakeChargerDetector())
        detections = pipeline.detect_charger(np.zeros((10, 10, 3), dtype=np.uint8))

        assert len(detections) == 1
        det = detections[0]
        assert isinstance(det, ChargerDetection)
        assert (det.x, det.y, det.width, det.height) == (10, 10, 70, 50)
        assert det.confidence == 0.8
        assert det.distance == pytest.approx(0.1001, abs=1e-3)


class TestAnnotation:
    def make_detection(self) -> ChargerDetection:
        return ChargerDetection(
            x=10,
            y=10,
            width=20,
            height=15,
            confidence=0.8,
            distance=0.1,
            position_mm=np.zeros(3),
        )

    def test_draw_changes_pixels(self):
        image = Image.new("RGB", (60, 40), (0, 0, 0))
        AnnotationEngine().draw_charger_detections(image, [self.make_detection()])
        assert image.getpixel((10, 10)) != (0, 0, 0)

    def test_no_detections_leaves_image_untouched(self):
        image = Image.new("RGB", (60, 40), (0, 0, 0))
        AnnotationEngine().draw_charger_detections(image, [])
        assert image.getpixel((10, 10)) == (0, 0, 0)

    def test_detect_and_annotate_accepts_charger_detections(self):
        image = Image.new("RGB", (60, 40), (0, 0, 0))
        annotated = AnnotationEngine().detect_and_annotate(image, [], [], None, [self.make_detection()])
        assert annotated.getpixel((10, 10)) != (0, 0, 0)
