"""Tests for typed state containers (COZMO-47).

Tests FrameState, CubeTrackingState, and DockState independently,
without constructing a live Brain.
"""

import threading
from unittest.mock import MagicMock

from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.navigation.dock_state import DockState
from pycozmo.vision.frame_state import FrameState


class TestFrameState:
    def test_defaults(self):
        fs = FrameState()
        assert fs.running is True
        assert fs.frame_count == 0
        assert fs.fps == 0.0
        assert fs.latest_raw_frame is None
        assert fs.latest_annotated_frame is None
        assert fs.latest_detections == []
        assert isinstance(fs.detections_lock, type(threading.Lock()))

    def test_mutability(self):
        fs = FrameState()
        fs.frame_count = 42
        fs.fps = 15.5
        fs.latest_detections = [MagicMock()]
        assert fs.frame_count == 42
        assert fs.fps == 15.5
        assert len(fs.latest_detections) == 1


class TestCubeTrackingState:
    def test_defaults(self):
        ct = CubeTrackingState()
        assert ct.cube_factory_id_map == {}
        assert ct.object_id_map == {}
        assert ct.last_seen_frame == {}
        assert ct.cycler is None
        assert ct.move_detectors == {}
        assert ct.suppress["fresh_frames_needed"] == 5

    def test_mutability(self):
        ct = CubeTrackingState()
        ct.cube_factory_id_map[1] = 0x12345678
        ct.object_id_map[1] = 42
        ct.last_seen_frame[1] = 100
        assert ct.cube_factory_id_map[1] == 0x12345678
        assert ct.object_id_map[1] == 42
        assert ct.last_seen_frame[1] == 100


class TestDockState:
    def test_defaults(self):
        ds = DockState()
        assert ds.on_charger is False
        assert ds.charging is False
        assert ds.dock_pose is None
        assert ds.origin_reset_origin_id is None
        assert isinstance(ds.lock, type(threading.Lock()))

    def test_mutability(self):
        ds = DockState()
        ds.on_charger = True
        ds.charging = True
        ds.dock_pose = MagicMock()
        ds.origin_reset_origin_id = 5
        assert ds.on_charger is True
        assert ds.charging is True
        assert ds.dock_pose is not None
        assert ds.origin_reset_origin_id == 5

    def test_locking(self):
        ds = DockState()
        with ds.lock:
            ds.on_charger = True
            ds.charging = True
        with ds.lock:
            assert ds.on_charger is True
            assert ds.charging is True
