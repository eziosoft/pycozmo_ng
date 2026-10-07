"""Tests for state_builder (COZMO-50).

Tests build_state_json and encode_jpeg with mock world map.
"""

import json
import time
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from pycozmo_ng.web.state_builder import (
    PATH_SCENE_LIFT_MM,
    build_state_json,
    encode_jpeg,
)


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
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
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
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert len(state["robot"]["position"]) == 3
        assert isinstance(state["robot"]["heading"], float)

    def test_cubes_converted(self):
        wm = _make_world_map()
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert len(state["cubes"]) == 1
        assert state["cubes"][0]["id"] == "cube_1"

    def test_telemetry_included(self):
        wm = _make_world_map()
        telemetry = {"battery_voltage": 3.7, "charging": False}
        result = build_state_json(
            wm,
            telemetry,
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert state["telemetry"]["battery_voltage"] == 3.7
        assert state["telemetry"]["charging"] is False

    def test_carry_status_included(self):
        """The HUD reads carry state from telemetry — it must survive."""
        wm = _make_world_map()
        telemetry = {"carry": "carrying", "carry_cube": 2, "lift_height": 92.0}
        result = build_state_json(
            wm,
            telemetry,
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert state["telemetry"]["carry"] == "carrying"
        assert state["telemetry"]["carry_cube"] == 2

    def test_occupancy_included_when_payload_given(self):
        # The changed/send decision belongs to OccupancyDelivery.take(); the
        # builder just embeds whatever that one call handed it.
        wm = _make_world_map()
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
            ("base64data", [10.0, 20.0]),
        )
        state = json.loads(result)
        assert state["occupancy_png"] == "base64data"
        assert state["occupancy_origin"] == [10.0, 20.0]

    def test_occupancy_excluded_when_none(self):
        wm = _make_world_map()
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert "occupancy_png" not in state

    def test_floor_size_and_center(self):
        wm = _make_world_map()
        result = build_state_json(
            wm,
            {},
            (800.0, 600.0),
            (100.0, 50.0),
            time.time(),
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
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert len(state["path_points"]) <= 801  # MAX_PATH_POINTS_SENT + 1

    def test_astar_path_converted_to_scene(self):
        """astar_path is 2-D world [x, y] -> 3-D scene [y, lift, x].

        Regression: passing the raw 2-element points through made
        ``waypoints[i][2]`` undefined in the browser, producing NaN in the
        Float32Array and an invisible A* line.
        """
        wm = MagicMock()
        wm.get_world_state.return_value = {
            "robot": {"position": [0, 0, 0], "yaw": 0},
            "cubes": {},
            "path_points": [],
            "astar_path": [[100.0, 50.0], [200.0, -30.0]],
        }
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        path = state["astar_path"]
        assert len(path) == 2
        assert all(len(p) == 3 for p in path)
        # world (100, 50) -> scene [50, lift, 100]
        assert path[0] == [50.0, PATH_SCENE_LIFT_MM, 100.0]
        assert path[1] == [-30.0, PATH_SCENE_LIFT_MM, 200.0]

    def test_astar_path_empty_when_absent(self):
        wm = MagicMock()
        wm.get_world_state.return_value = {
            "robot": {"position": [0, 0, 0], "yaw": 0},
            "cubes": {},
            "path_points": [],
        }
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        state = json.loads(result)
        assert state["astar_path"] == []


class TestChargerInState:
    """The charger is one static landmark — serialized or null (COZMO-65)."""

    @staticmethod
    def _build(world_extra: dict) -> dict:
        wm = MagicMock()
        base = {
            "robot": {"position": [0.0, 0.0, 0.0], "yaw": 0.0},
            "cubes": {},
            "path_points": [],
        }
        base.update(world_extra)
        wm.get_world_state.return_value = base
        result = build_state_json(
            wm,
            {},
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        return json.loads(result)

    def test_null_until_first_sighting(self):
        state = self._build({})
        assert state["charger"] is None

    def test_null_after_reset(self):
        # reset_map clears the slot; the browser removes the marker.
        state = self._build({"charger": None})
        assert state["charger"] is None

    def test_converted_to_scene_frame_with_size(self):
        state = self._build({"charger": {"position": [100.0, 200.0, 50.0], "confidence": 0.85}})
        charger = state["charger"]
        # world (x, y, z) -> scene [y, z, x]
        assert charger["position"] == [200.0, 50.0, 100.0]
        assert charger["confidence"] == 0.85
        assert charger["width"] == 23.0
        assert charger["height"] == 15.0

    def test_confidence_defaults_to_one(self):
        state = self._build({"charger": {"position": [0.0, 0.0, 30.0]}})
        assert state["charger"]["confidence"] == 1.0


class TestCarriedCube:
    """The carried cube is flagged so the browser locks it to the robot.

    The world map only holds the cube's last-seen pose — while it rides
    the forks (out of camera view) that pose is stale, so the state marks
    the cube ``carried`` and the viewer re-renders it on the forks.
    """

    @staticmethod
    def _build(telemetry: dict, world_map=None) -> dict:
        result = build_state_json(
            world_map or _make_world_map(),
            telemetry,
            (600.0, 600.0),
            (200.0, 0.0),
            time.time(),
        )
        return json.loads(result)

    def test_matching_cube_is_flagged_carried(self):
        # The map tracks cube index 0 -> label "cube_1".
        state = self._build({"carry": "carrying", "carry_cube": 1})
        carried = [c for c in state["cubes"] if c["carried"]]
        assert [c["id"] for c in carried] == ["cube_1"]

    def test_placeholder_appended_when_carried_cube_untracked(self):
        # Carrying cube 2 while the map only has cube 1 (the carried cube
        # left the frame): a placeholder must still reach the browser.
        state = self._build({"carry": "carrying", "carry_cube": 2})
        ids = [c["id"] for c in state["cubes"]]
        assert ids == ["cube_1", "cube_2"]
        placeholder = state["cubes"][1]
        assert placeholder["carried"] is True
        assert placeholder["face_idx"] == -1
        # Placeholder sits at the robot pose (the browser re-locks it to
        # the forks anyway).
        assert placeholder["position"] == state["robot"]["position"]
        # The tracked cube is not the carried one.
        assert state["cubes"][0]["carried"] is False

    def test_not_carrying_leaves_all_cubes_unflagged(self):
        state = self._build({"carry": "not_carrying", "carry_cube": 1})
        assert all(c["carried"] is False for c in state["cubes"])
        assert len(state["cubes"]) == 1  # no placeholder

    def test_carrying_without_cube_id_appends_nothing(self):
        # Defensive: carrying but no known cube id -> no flag, no placeholder.
        state = self._build({"carry": "carrying", "carry_cube": None})
        assert all(c["carried"] is False for c in state["cubes"])
        assert len(state["cubes"]) == 1

    def test_no_telemetry_at_all(self):
        state = self._build({})
        assert all(c["carried"] is False for c in state["cubes"])
