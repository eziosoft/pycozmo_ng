"""Tests for CubeFusionTracker — fuses per-face detections into cube poses.

Validates the cleaned API (COZMO-51): no dead params, stateless fusion,
no sys.path hacks.
"""

import numpy as np
import pytest

from pycozmo.vision.cube_detector.cube_detector import CubeFusionTracker


def _make_detection(
    cube_id: int,
    face_type: str = "wall",
    rotation_deg: int = 0,
    score: float = 0.9,
    position: np.ndarray | None = None,
    rotation_matrix: np.ndarray | None = None,
) -> dict:
    """Build a minimal detection dict matching CozmoCubeDetector output."""
    if position is None:
        position = np.array([0.0, 0.0, 300.0], dtype=np.float64)
    if rotation_matrix is None:
        rotation_matrix = np.eye(3, dtype=np.float64)
    return {
        "cube_id": cube_id,
        "face_type": face_type,
        "rotation": rotation_deg,
        "score": score,
        "position": position,
        "rotation_matrix": rotation_matrix,
    }


class TestCubeFusionTrackerAPI:
    def test_init_with_defaults(self):
        tracker = CubeFusionTracker()
        assert tracker.cube_size == 44.0
        assert tracker.half_size == 22.0

    def test_init_custom_cube_size(self):
        tracker = CubeFusionTracker(cube_size_mm=50.0)
        assert tracker.cube_size == 50.0
        assert tracker.half_size == 25.0

    def test_no_smoothing_alpha_attr(self):
        """COZMO-51: smoothing_alpha removed."""
        tracker = CubeFusionTracker()
        assert not hasattr(tracker, "smoothing_alpha")

    def test_no_max_age_attr(self):
        """COZMO-51: max_age_frames removed."""
        tracker = CubeFusionTracker()
        assert not hasattr(tracker, "max_age_frames")


class TestCubeFusionTrackerFusion:
    def test_single_detection_fused(self):
        tracker = CubeFusionTracker()
        det = _make_detection(cube_id=0, score=0.85)
        fused = tracker.update([det])
        assert 0 in fused
        assert fused[0]["confidence"] == pytest.approx(0.85)
        assert fused[0]["num_faces"] == 1

    def test_multiple_detections_same_cube(self):
        tracker = CubeFusionTracker()
        d1 = _make_detection(cube_id=0, score=0.9, position=np.array([10.0, 0.0, 300.0]))
        d2 = _make_detection(cube_id=0, score=0.7, position=np.array([-10.0, 0.0, 300.0]))
        fused = tracker.update([d1, d2])
        assert 0 in fused
        assert fused[0]["num_faces"] == 2
        # Center should be between the two face positions
        pos = fused[0]["position"]
        assert abs(pos[0]) < 5.0  # roughly centered

    def test_different_cubes_separated(self):
        tracker = CubeFusionTracker()
        d0 = _make_detection(cube_id=0)
        d1 = _make_detection(cube_id=1)
        fused = tracker.update([d0, d1])
        assert 0 in fused
        assert 1 in fused

    def test_empty_detections(self):
        tracker = CubeFusionTracker()
        fused = tracker.update([])
        assert fused == {}

    def test_stateless_between_frames(self):
        """Each update() call is independent — no state leaks."""
        tracker = CubeFusionTracker()
        d = _make_detection(cube_id=0, score=0.9)
        fused1 = tracker.update([d])
        fused2 = tracker.update([d])
        # Both frames produce independent results with the same confidence
        assert fused1[0]["confidence"] == fused2[0]["confidence"]

    def test_cube_center_offset_from_face(self):
        """Cube center is half_size behind the face along its normal."""
        tracker = CubeFusionTracker(cube_size_mm=44.0)
        # Face at z=300, normal toward camera (+Z), so center at z=300-22=278
        det = _make_detection(
            cube_id=0,
            position=np.array([0.0, 0.0, 300.0]),
            rotation_matrix=np.eye(3),
        )
        fused = tracker.update([det])
        # Face normal is +Z (col 2 of identity), so center = face - half_size * normal
        assert fused[0]["position"][2] == pytest.approx(278.0, abs=1e-6)


class TestCubeFusionTrackerCleanup:
    def test_no_sys_path_append(self):
        """COZMO-51: module-level sys.path.append removed."""
        import inspect

        import pycozmo.vision.cube_detector.cube_detector as mod
        source = inspect.getsource(mod)
        assert "sys.path.append" not in source
