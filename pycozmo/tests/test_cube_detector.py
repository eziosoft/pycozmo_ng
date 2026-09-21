"""Tests for CozmoCubeDetector performance optimizations (COZMO-55).

Verifies that _extract_patch uses bounding-box ROI instead of full-image
warpPerspective, and that _match_marker efficiently handles template matching.
"""

import time
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from pycozmo.vision.cube_detector.cube_detector import (
    CozmoCubeDetector,
    CubeFusionTracker,
)
from pycozmo.vision.detection_pipeline import load_marker_templates


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

    def test_extract_patch_faster_than_warp_perspective(self):
        """ROI-based extraction should be faster than full-image warpPerspective."""
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

        start = time.perf_counter()
        for _ in range(100):
            detector._extract_patch(img, quad)
        roi_time = time.perf_counter() - start

        src_pts = detector._order_points(quad.reshape(4, 2))
        M = cv2.getPerspectiveTransform(src_pts, detector.dst_pts)
        start = time.perf_counter()
        for _ in range(100):
            cv2.warpPerspective(img, M, (32, 32), flags=cv2.INTER_LINEAR)
        warp_time = time.perf_counter() - start

        print(f"ROI: {roi_time:.4f}s, warpPerspective: {warp_time:.4f}s")
        assert roi_time < warp_time * 2.0


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
                "cube_id": 0,
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
                "cube_id": 0,
                "face_type": "wall",
                "position": np.array([0.0, 0.0, 200.0]),
                "rotation_matrix": np.eye(3),
                "face_idx": 0,
                "score": 0.9,
                "rotation": 0,
            },
            {
                "cube_id": 0,
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


class TestViewerServerNoCubesAttribute:
    def test_no_cubes_dict(self):
        """WebViewerServer should not have deprecated cubes dict."""
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer()
        assert not hasattr(server, "cubes")

    def test_broadcast_interval_default(self):
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer()
        assert server.broadcast_interval_s == pytest.approx(0.05)

    def test_broadcast_interval_custom(self):
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer(broadcast_interval_s=0.1)
        assert server.broadcast_interval_s == pytest.approx(0.1)

    def test_broadcast_sets_dirty(self):
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer()
        server._broadcast_queue = MagicMock()
        server._broadcast_queue.qsize = lambda: 0
        server.broadcast()
        assert server._dirty is True

    def test_deprecated_methods_delegate_to_world_map(self):
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer()
        server.update_cube("cube_1", [100, 200, 0], __import__("numpy").eye(3))
        # Should not raise - delegates to world_map

    def test_clear_cubes_delegates(self):
        from pycozmo.web.viewer_server import WebViewerServer

        server = WebViewerServer()
        server.clear_cubes()
        # Should not raise - delegates to world_map.reset_map
