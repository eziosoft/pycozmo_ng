"""Tests for state_builder (COZMO-50).

Tests build_state_json and encode_jpeg with mock world map.
"""

import json
import time
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from pycozmo.web.state_builder import build_state_json, encode_jpeg


def _make_world_map():
    wm = MagicMock()
    wm.get_world_state.return_value = {
        "robot": {"position": [100.0, 200.0, 0.0], "yaw": 0.5},
        "cubes": {
            0: {
                "position": [300.0, 100.0, 0.0],
                "rotation": np.eye(3),
            },
        },
        "path_points": [[100.0, 200.0, 0.0], [150.0, 250.0, 0.0]],
    }
    return wm


class TestEncodeJpeg:
    def test_returns_bytes(self):
        img = Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))
        result = encode_jpeg(img, quality=50)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_is_valid_jpeg(self):
        img = Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))
        result = encode_jpeg(img, quality=50)
        assert result[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_quality_affects_size(self):
        img = Image.fromarray(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        low = encode_jpeg(img, quality=10)
        high = encode_jpeg(img, quality=95)
        assert len(low) < len(high)


class TestBuildStateJson:
    def test_basic_structure(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert "robot" in state
        assert "cubes" in state
        assert "path_points" in state
        assert "telemetry" in state
        assert "floor_size" in state
        assert "floor_center" in state
        assert "timestamp" in state

    def test_robot_position_converted(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert len(state["robot"]["position"]) == 3
        assert isinstance(state["robot"]["heading"], float)

    def test_cubes_converted(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert len(state["cubes"]) == 1
        assert state["cubes"][0]["id"] == "cube_1"

    def test_telemetry_included(self):
        wm = _make_world_map()
        telemetry = {"battery_voltage": 3.7, "charging": False}
        result = build_state_json(
            wm, telemetry, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert state["telemetry"]["battery_voltage"] == 3.7
        assert state["telemetry"]["charging"] is False

    def test_occupancy_included_when_forced(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            "base64data", [10.0, 20.0], 1, -1, time.time(),
            force_occupancy=True,
        )
        state = json.loads(result)
        assert state["occupancy_png"] == "base64data"
        assert state["occupancy_origin"] == [10.0, 20.0]

    def test_occupancy_included_when_seq_changed(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            "base64data", [10.0, 20.0], 5, 4, time.time(),
        )
        state = json.loads(result)
        assert state["occupancy_png"] == "base64data"

    def test_occupancy_excluded_when_seq_same(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            "base64data", [10.0, 20.0], 5, 5, time.time(),
        )
        state = json.loads(result)
        assert "occupancy_png" not in state

    def test_occupancy_excluded_when_none(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert "occupancy_png" not in state

    def test_floor_size_and_center(self):
        wm = _make_world_map()
        result = build_state_json(
            wm, {}, (800.0, 600.0), (100.0, 50.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert state["floor_size"] == [800.0, 600.0]
        assert state["floor_center"] == [100.0, 50.0]

    def test_path_decimation(self):
        wm = MagicMock()
        wm.get_world_state.return_value = {
            "robot": {"position": [0, 0, 0], "yaw": 0},
            "cubes": {},
            "path_points": [[i, i, 0] for i in range(1000)],
        }
        result = build_state_json(
            wm, {}, (600.0, 600.0), (200.0, 0.0),
            None, None, 0, -1, time.time(),
        )
        state = json.loads(result)
        assert len(state["path_points"]) <= 801  # MAX_PATH_POINTS_SENT + 1
