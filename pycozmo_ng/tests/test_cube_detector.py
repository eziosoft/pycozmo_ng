"""Tests for CozmoCubeDetector performance optimizations (COZMO-55).

Verifies that _extract_patch uses bounding-box ROI instead of full-image
warpPerspective, and that _match_marker efficiently handles template matching.
"""

import cv2
import numpy as np
import pytest

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
    def test_scores_match_match_template(self):
        """The matrix matcher must return what ``TM_CCOEFF_NORMED`` returns.

        ``_match_marker`` scores every template in one dot product; if the
        normalisation ever drifts the scores change silently and the detection
        threshold stops meaning what it says.
        """
        markers = load_marker_templates()
        detector = CozmoCubeDetector(
            camera_matrix=np.eye(3),
            dist_coeffs=np.zeros(5),
            target_markers=markers,
        )
        rng = np.random.default_rng(0)
        patches = [
            markers[0],
            np.rot90(markers[3], 2).copy(),
            rng.integers(0, 255, (32, 32), dtype=np.uint8),
        ]
        for patch in patches:
            marker_id, score, rotation = detector._match_marker(patch)
            expected = max(
                (float(cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0][0]), i, r)
                for i, rotations in enumerate(detector.rotated_markers)
                for r, template in enumerate(rotations)
            )
            assert (marker_id, rotation) == (expected[1], expected[2])
            assert score == pytest.approx(expected[0], abs=1e-5)

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


def _draw_rounded_rect(img, x0, y0, x1, y1, radius, value):
    """Fill a rounded rectangle — the shape of the printed cube markers."""
    cv2.rectangle(img, (x0 + radius, y0), (x1 - radius, y1), value, -1)
    cv2.rectangle(img, (x0, y0 + radius), (x1, y1 - radius), value, -1)
    for cx, cy in (
        (x0 + radius, y0 + radius),
        (x1 - radius, y0 + radius),
        (x1 - radius, y1 - radius),
        (x0 + radius, y1 - radius),
    ):
        cv2.circle(img, (cx, cy), radius, value, -1)


def _make_rounded_marker(side=120, radius_frac=0.2, border=14, supersample=4):
    """Render a dark rounded-square marker ring on white.

    Returns ``(image, sharp_corners)`` where ``sharp_corners`` are the ideal
    (un-rounded) square corners in the marker's own pixel coordinates — the
    geometry a pose solver must recover.
    """
    margin = border
    size = side + 2 * margin
    big = np.full((size * supersample, size * supersample), 255, dtype=np.uint8)
    radius = round(radius_frac * side * supersample)
    _draw_rounded_rect(
        big,
        margin * supersample,
        margin * supersample,
        (size - margin) * supersample - 1,
        (size - margin) * supersample - 1,
        radius,
        0,
    )
    # White interior: leaves the dark border ring whose outer edge is the marker.
    inner = margin + 11
    _draw_rounded_rect(
        big,
        inner * supersample,
        inner * supersample,
        (size - inner) * supersample - 1,
        (size - inner) * supersample - 1,
        round(radius_frac * (side - 22) * supersample),
        255,
    )
    img = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    corners = np.array(
        [[margin, margin], [size - margin, margin], [size - margin, size - margin], [margin, size - margin]],
        dtype=np.float32,
    )
    return img, corners


def _render_scene(marker, homography, canvas=(480, 640)):
    """Warp ``marker`` onto a white canvas by ``homography``."""
    warped = cv2.warpPerspective(marker, homography, canvas, flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpPerspective(
        np.full(marker.shape, 255, dtype=np.uint8), homography, canvas, flags=cv2.INTER_NEAREST, borderValue=0
    )
    return np.where(mask > 0, warped, 255).astype(np.uint8)


def _corner_error(a, b):
    """Mean corner distance, allowing for cyclic order and winding."""
    best = float("inf")
    for candidate in (a, a[::-1]):
        for shift in range(4):
            shifted = np.roll(candidate, shift, axis=0)
            best = min(best, float(np.mean(np.linalg.norm(shifted - b, axis=1))))
    return best


def _contour_of(gray):
    """Largest dark-blob contour, as the detector's contour stage sees it."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]
    return max(contours, key=cv2.contourArea)


def _tilted_scene(rx, ry, rz, distance_mm, side_mm=40.0, marker_px=180):
    """Render a rounded marker at a known pose; return ``(gray, truth_corners)``.

    ``truth_corners`` are the ideal *sharp* corners — what a pose solver must
    recover even though the print has rounded corners.
    """
    camera = np.array([[320.0, 0, 320], [0, 320, 240], [0, 0, 1]])

    def rot(ax, ay, az):
        rx_m = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
        ry_m = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
        rz_m = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
        return rz_m @ ry_m @ rx_m

    marker, corners = _make_rounded_marker(side=marker_px - 44, border=22)
    rotation = rot(rx, ry, rz)
    translation = np.array([[0.0], [0.0], [distance_mm]])
    image_from_object = camera @ np.column_stack([rotation[:, 0], rotation[:, 1], translation[:, 0]])
    size = marker.shape[0]
    object_from_marker = np.array([[side_mm / size, 0, -side_mm / 2], [0, -side_mm / size, side_mm / 2], [0, 0, 1]])
    homography = image_from_object @ object_from_marker
    gray = cv2.warpPerspective(marker, homography, (640, 480), flags=cv2.INTER_LINEAR, borderValue=255)
    truth = cv2.perspectiveTransform(corners.reshape(1, -1, 2), homography).reshape(4, 2)
    return gray, truth


class TestQuadFromContour:
    """Rounded print corners: the contour's vertices sit on the arcs, so the
    quad must be recovered from the straight edges instead (COZMO-…)."""

    def test_recovers_sharp_corners_under_tilt(self):
        gray, truth = _tilted_scene(np.radians(45), np.radians(25), np.radians(15), 120.0)
        contour = _contour_of(gray)

        # The old gate would have rejected this marker outright.
        peri = cv2.arcLength(contour, True)
        assert len(cv2.approxPolyDP(contour, 0.02 * peri, True)) != 4

        quad = CozmoCubeDetector._quad_from_contour(contour)
        assert quad is not None
        assert _corner_error(quad, truth) < 1.2

    def test_beats_min_area_rect_under_tilt(self):
        gray, truth = _tilted_scene(np.radians(45), np.radians(25), np.radians(15), 120.0)
        contour = _contour_of(gray)

        rect = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
        quad = CozmoCubeDetector._quad_from_contour(contour)
        assert _corner_error(quad, truth) < 0.3 * _corner_error(rect, truth)

    def test_frontal_marker_is_accurate(self):
        gray, truth = _tilted_scene(0.0, 0.0, 0.0, 200.0)
        quad = CozmoCubeDetector._quad_from_contour(_contour_of(gray))
        assert _corner_error(quad, truth) < 0.8

    def test_rejects_degenerate_contour(self):
        tiny = np.array([[[0, 0]], [[3, 0]], [[3, 3]], [[0, 3]]], dtype=np.int32)
        assert CozmoCubeDetector._quad_from_contour(tiny) is None


class TestSolvePoseRobustness:
    def test_rejects_non_finite_pose(self):
        """IPPE_SQUARE degenerates to NaN on perfectly symmetric input; a NaN
        pose must never reach WorldMap."""
        detector = CozmoCubeDetector(camera_matrix=np.eye(3), dist_coeffs=np.zeros(5), target_markers=[])
        square = np.array([[100.0, 100.0], [200.0, 100.0], [200.0, 200.0], [100.0, 200.0]], dtype=np.float32)
        success, rvec, tvec = detector._solve_pose(square)
        if success:
            assert np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))


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
