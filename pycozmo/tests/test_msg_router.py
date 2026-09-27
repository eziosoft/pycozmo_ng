"""Tests for MsgRouter (COZMO-41).

Tests message dispatch logic without any WebSocket or asyncio overhead.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from pycozmo.web.msg_router import MsgRouter


class TestMsgRouterDispatch:
    def setup_method(self):
        self.wm = MagicMock()
        # Only world-map key 0 ("cube_1") exists.
        self.wm.get_cube.side_effect = lambda idx: (
            SimpleNamespace(position=np.array([100.0, 200.0, 0.0]))
            if idx == 0
            else None
        )
        self.router = MsgRouter(world_map=self.wm)

    def test_unknown_type_ignored(self):
        self.router.route(json.dumps({"type": "unknown"}))

    def test_malformed_json_ignored(self):
        self.router.route("not json {{{")

    def test_none_json_ignored(self):
        self.router.route("")

    def test_control_message(self):
        cb = MagicMock()
        self.router.set_control_listener(cb)
        self.router.route(json.dumps({"type": "control", "keys": ["a", "b"]}))
        cb.assert_called_once_with({"a", "b"})

    def test_control_empty_keys(self):
        cb = MagicMock()
        self.router.set_control_listener(cb)
        self.router.route(json.dumps({"type": "control", "keys": []}))
        cb.assert_called_once_with(set())

    def test_control_no_keys(self):
        cb = MagicMock()
        self.router.set_control_listener(cb)
        self.router.route(json.dumps({"type": "control"}))
        cb.assert_called_once_with(set())

    def test_control_no_listener(self):
        self.router.route(json.dumps({"type": "control", "keys": ["a"]}))

    def test_control_listener_exception_swallowed(self):
        def bad_listener(keys):
            raise RuntimeError("boom")

        self.router.set_control_listener(bad_listener)
        self.router.route(json.dumps({"type": "control", "keys": ["a"]}))

    def test_navigate_to(self):
        cb = MagicMock()
        self.router.set_navigate_listener(cb)
        self.router.route(json.dumps({"type": "navigate_to", "x": 1.0, "z": 2.0}))
        cb.assert_called_once()
        args = cb.call_args[0]
        assert isinstance(args[0], float)
        assert isinstance(args[1], float)

    def test_navigate_cancel(self):
        cb = MagicMock()
        self.router.set_navigate_listener(cb)
        self.router.route(json.dumps({"type": "navigate_cancel"}))
        cb.assert_called_once_with(None)

    def test_navigate_no_listener(self):
        self.router.route(json.dumps({"type": "navigate_to", "x": 0, "z": 0}))

    def test_approach_cube_found(self):
        cb = MagicMock()
        self.router.set_approach_listener(cb)
        self.router.route(json.dumps({"type": "approach_cube", "cube_id": "cube_1"}))
        cb.assert_called_once_with(0, 100.0, 200.0)

    def test_approach_cube_not_found(self):
        cb = MagicMock()
        self.router.set_approach_listener(cb)
        self.router.route(json.dumps({"type": "approach_cube", "cube_id": "cube_99"}))
        cb.assert_not_called()

    def test_approach_cube_no_world_map(self):
        router = MsgRouter(world_map=None)
        cb = MagicMock()
        router.set_approach_listener(cb)
        router.route(json.dumps({"type": "approach_cube", "cube_id": "cube_1"}))
        cb.assert_not_called()

    def test_dock(self):
        cb = MagicMock()
        self.router.set_dock_listener(cb)
        self.router.route(json.dumps({"type": "dock"}))
        cb.assert_called_once()

    def test_dock_no_listener(self):
        self.router.route(json.dumps({"type": "dock"}))

    def test_camera_settings(self):
        cb = MagicMock()
        self.router.set_camera_settings_listener(cb)
        self.router.route(
            json.dumps(
                {
                    "type": "camera_settings",
                    "auto_exposure_enabled": False,
                    "gain": 0.5,
                    "exposure_ms": 50,
                    "color": True,
                    "resolution": "VGA",
                    "jpeg_quality": 90,
                    "head_light": False,
                }
            )
        )
        cb.assert_called_once()
        settings = cb.call_args[0][0]
        assert settings["auto_exposure_enabled"] is False
        assert settings["gain"] == 0.5
        assert settings["exposure_ms"] == 50
        assert settings["color"] is True
        assert settings["resolution"] == "VGA"
        assert settings["jpeg_quality"] == 90
        assert settings["head_light"] is False

    def test_camera_settings_defaults(self):
        cb = MagicMock()
        self.router.set_camera_settings_listener(cb)
        self.router.route(json.dumps({"type": "camera_settings"}))
        settings = cb.call_args[0][0]
        assert settings["auto_exposure_enabled"] is True
        assert settings["gain"] == 0.0
        assert settings["exposure_ms"] == 100
        assert settings["color"] is False
        assert settings["resolution"] == "QVGA"
        assert settings["jpeg_quality"] == 70
        assert settings["head_light"] is False

    def test_camera_settings_stored(self):
        cb = MagicMock()
        self.router.set_camera_settings_listener(cb)
        self.router.route(
            json.dumps(
                {
                    "type": "camera_settings",
                    "gain": 1.0,
                }
            )
        )
        assert self.router.camera_settings["gain"] == 1.0

    def test_set_camera_settings_merges(self):
        self.router.set_camera_settings({"gain": 0.5, "color": True})
        assert self.router.camera_settings["gain"] == 0.5
        assert self.router.camera_settings["color"] is True
        self.router.set_camera_settings({"color": False})
        assert self.router.camera_settings["color"] is False
        assert self.router.camera_settings["gain"] == 0.5


class TestMsgRouterListenerSwapping:
    def test_replace_listener(self):
        router = MsgRouter()
        cb1 = MagicMock()
        cb2 = MagicMock()
        router.set_control_listener(cb1)
        router.route(json.dumps({"type": "control", "keys": ["a"]}))
        cb1.assert_called_once()
        router.set_control_listener(cb2)
        router.route(json.dumps({"type": "control", "keys": ["b"]}))
        cb2.assert_called_once()
        assert cb1.call_count == 1

    def test_unsubscribe_listener(self):
        router = MsgRouter()
        cb = MagicMock()
        router.set_control_listener(cb)
        router.set_control_listener(None)
        router.route(json.dumps({"type": "control", "keys": ["a"]}))
        cb.assert_not_called()
