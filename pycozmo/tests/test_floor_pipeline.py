"""Tests for FloorPipeline (COZMO-48).

Tests FloorPipeline construction and processing with mock camera calibration.
"""

from unittest.mock import MagicMock

import numpy as np

from pycozmo.vision.floor_pipeline import FloorPipeline


def _make_camera_calibration():
    cal = MagicMock()
    cal.camera_matrix = np.array(
        [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    cal.image_width = 320
    cal.image_height = 240
    return cal


class TestFloorPipeline:
    def test_disabled_by_default(self):
        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=False)
        assert fp.enabled is False
        assert fp.occupancy_enabled is False
        assert fp.floor_projector is None
        assert fp.occupancy_map is None

    def test_enabled_with_occupancy(self):
        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=True, enable_occupancy_map=True)
        assert fp.enabled is True
        assert fp.occupancy_enabled is True
        assert fp.floor_projector is not None
        assert fp.occupancy_map is not None

    def test_enabled_without_occupancy(self):
        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=True, enable_occupancy_map=False)
        assert fp.enabled is True
        assert fp.occupancy_enabled is False
        assert fp.floor_projector is not None
        assert fp.occupancy_map is None

    def test_process_disabled_returns_none(self):
        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=False)
        img = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        floor_img, occ_grid, occ_map_img = fp.process(img, 0.0, None)
        assert floor_img is None
        assert occ_grid is None
        assert occ_map_img is None

    def test_process_enabled_produces_output(self):
        import cv2

        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=True, enable_occupancy_map=True)
        rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        robot_pose = MagicMock()
        robot_pose.position.x = 0.0
        robot_pose.position.y = 0.0
        robot_pose.rotation.angle_z.radians = 0.0

        floor_img, occ_grid, occ_map_img = fp.process(img_bgr, 0.0, robot_pose)
        assert floor_img is not None
        assert occ_grid is not None
        assert occ_map_img is not None
        assert floor_img.ndim == 3  # BGR

    def test_process_enabled_without_occupancy(self):
        import cv2

        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=True, enable_occupancy_map=False)
        rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        robot_pose = MagicMock()
        robot_pose.position.x = 0.0
        robot_pose.position.y = 0.0
        robot_pose.rotation.angle_z.radians = 0.0

        floor_img, occ_grid, occ_map_img = fp.process(img_bgr, 0.0, robot_pose)
        assert floor_img is not None
        assert occ_grid is not None
        assert occ_map_img is None

    def test_process_none_pose_skips_occupancy(self):
        import cv2

        cal = _make_camera_calibration()
        fp = FloorPipeline(cal, enable_floor_projection=True, enable_occupancy_map=True)
        rgb = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        floor_img, occ_grid, occ_map_img = fp.process(img_bgr, 0.0, None)
        assert floor_img is not None
        assert occ_grid is not None
        assert occ_map_img is None  # No robot pose → no occupancy update
