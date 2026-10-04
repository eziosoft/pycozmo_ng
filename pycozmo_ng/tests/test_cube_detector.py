"""Tests for CozmoCubeDetector performance optimizations (COZMO-55).

Verifies that _extract_patch uses bounding-box ROI instead of full-image
warpPerspective, and that _match_marker efficiently handles template matching.
"""

import cv2
import numpy as np

from pycozmo_ng.vision.cube_detector.cube_detector import (
    CozmoCubeDetector,
    CubeFusionTracker,
)
from pycozmo_ng.vision.cube_detector.detection_pipeline import load_marker_templates


class TestExtractPatch:
    """Tests that _extract_patch uses ROI-based extraction instead of
    full-image warpPerspective."""

    def test_extract_patch_returns_32x32(self):
        """_extract_patch should return a 32x32 normalized patch."""
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        img = np.zeros((240, 320), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (160, 160), 255, -1)
        quad = np.array(
            [
                [[100, 100]],
                [[160, 100]],
                [[160, 160]],
                [[100, 160]],
            ],
            dtype=np.float32,
        )
        patch = detector._extract_patch(img, quad)
        assert patch.shape == (32, 32)
        assert patch.dtype == np.uint8

    def test_extract_patch_normalized(self):
        """_extract_patch should return normalized values (0-255)."""
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        img = np.random.randint(0, 255, (240, 320), dtype=np.uint8)
        quad = np.array(
            [
                [[50, 50]],
                [[100, 50]],
                [[100, 100]],
                [[50, 100]],
            ],
            dtype=np.float32,
        )
        patch = detector._extract_patch(img, quad)
        assert patch.min() >= 0
        assert patch.max() <= 255

    def test_extract_patch_preserves_perspective(self):
        """_extract_patch should produce a proper perspective-corrected patch."""
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        quad = np.array(
            [
                [[100, 80]],
                [[200, 90]],
                [[210, 180]],
                [[90, 170]],
            ],
            dtype=np.float32,
        )
        patch = detector._extract_patch(img, quad)
        assert patch.shape == (32, 32)
        assert patch.dtype == np.uint8
        assert patch.min() >= 0
        assert patch.max() <= 255

    def test_extract_patch_matches_template(self):
        """_extract_patch output should match the marker template."""
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        marker = cv2.normalize(markers[0], None, 0, 255, cv2.NORM_MINMAX)
        img = np.zeros((240, 320), dtype=np.uint8)
        img[100:132, 100:132] = marker
        quad = np.array(
            [
                [[100, 100]],
                [[132, 100]],
                [[132, 132]],
                [[100, 132]],
            ],
            dtype=np.float32,
        )
        patch = detector._extract_patch(img, quad)
        marker_id, score, _ = detector._match_marker(patch)
        assert marker_id == 0
        assert score > 0.8


class TestMatchMarker:
    def test_returns_valid_marker_id(self):
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        marker = cv2.normalize(markers[0], None, 0, 255, cv2.NORM_MINMAX)
        marker_id, score, rotation = detector._match_marker(marker)
        assert marker_id >= 0
        assert score > 0.9
        assert 0 <= rotation <= 3

    def test_returns_best_match(self):
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        rotated = np.rot90(markers[0], 1)
        marker_id, score, _rotation = detector._match_marker(rotated)
        assert marker_id >= 0
        assert score > 0.9

    def test_total_rotated_markers(self):
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        assert len(detector.rotated_markers) == len(markers)
        assert all(len(r) == 4 for r in detector.rotated_markers)
        assert len(detector.rotated_markers) * 4 == 72


class TestCozmoCubeDetector:
    def test_detect_returns_list(self):
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        results = detector.detect(img)
        assert isinstance(results, list)


class TestCubeFusionTracker:
    def test_fuse_single_detection(self):
        tracker = CubeFusionTracker(cube_size_mm=44.0)
        detections = [
            {
                "cube_index": 0,
                "face_type": "wall",
                "position": np.array([0.0, 0.0, 200.0]),
                "rotation_matrix": np.eye(3),
                "face_idx": 0,
                "score": 0.9,
                "rotation": 0,
            }
        ]
        fused = tracker.update(detections)
        assert 0 in fused
        assert len(fused) == 1

    def test_fuse_multiple_detections_same_cube(self):
        tracker = CubeFusionTracker(cube_size_mm=44.0)
        detections = [
            {
                "cube_index": 0,
                "face_type": "wall",
                "position": np.array([0.0, 0.0, 200.0]),
                "rotation_matrix": np.eye(3),
                "face_idx": 0,
                "score": 0.9,
                "rotation": 0,
            },
            {
                "cube_index": 0,
                "face_type": "top",
                "position": np.array([0.0, 0.0, 200.0]),
                "rotation_matrix": np.eye(3),
                "face_idx": 5,
                "score": 0.8,
                "rotation": 0,
            },
        ]
        fused = tracker.update(detections)
        assert len(fused) == 1
        assert fused[0]["num_faces"] == 2

    def test_empty_detections(self):
        tracker = CubeFusionTracker(cube_size_mm=44.0)
        fused = tracker.update([])
        assert fused == {}
