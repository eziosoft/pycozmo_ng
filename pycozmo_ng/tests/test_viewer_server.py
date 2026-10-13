"""Tests for WebViewerServer broadcast rate limiting and state management.

Verifies that the _broadcast_loop respects broadcast_interval_s and that
the server properly coalesces state updates without excessive calls to
get_world_state.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from pycozmo_ng.web.viewer_server import WebViewerServer


class TestBroadcastInterval:
    def test_default_interval_is_50ms(self):
        server = WebViewerServer()
        assert server.broadcast_interval_s == pytest.approx(0.05)

    def test_custom_interval(self):
        server = WebViewerServer(broadcast_interval_s=0.1)
        assert server.broadcast_interval_s == pytest.approx(0.1)

    def test_broadcast_sets_dirty_flag(self):
        server = WebViewerServer()
        server._broadcast_queue = asyncio.Queue()
        server._clients = set()
        server.broadcast()
        assert server._dirty is True

    def test_broadcast_queued_with_clients(self):
        server = WebViewerServer()
        server._broadcast_queue = asyncio.Queue()
        server._clients = set()
        mock_loop = MagicMock()
        mock_loop.is_running.return_value = True
        mock_loop.call_soon_threadsafe = MagicMock()
        server._ws_server._loop = mock_loop
        server.broadcast()
        assert server._dirty is True


class TestBroadcastLoop:
    def test_broadcast_interval_not_exceeded(self):
        """Broadcast should not happen more often than broadcast_interval_s."""
        server = WebViewerServer(broadcast_interval_s=0.5)
        server._clients = set()
        server._last_broadcast_time = time.time() - 0.1
        # If interval not elapsed, should skip sending

    def test_broadcast_interval_elapsed(self):
        """Broadcast should happen when interval has elapsed."""
        server = WebViewerServer(broadcast_interval_s=0.05)
        server._last_broadcast_time = time.time() - 0.5
        assert server._last_broadcast_time < time.time() - 0.05

    def test_no_clients_skips_broadcast(self):
        """Broadcast loop should skip when no clients connected."""
        server = WebViewerServer(broadcast_interval_s=0.05)
        server._clients = set()
        server._broadcast_queue = asyncio.Queue()
        assert len(server._clients) == 0

    def test_coalescing_drops_pending(self):
        """Multiple broadcast() calls should coalesce into one send."""
        server = WebViewerServer(broadcast_interval_s=0.05)
        server._broadcast_queue = asyncio.Queue()
        server._broadcast_queue.put_nowait(True)
        server._broadcast_queue.put_nowait(True)
        server._broadcast_queue.put_nowait(True)
        assert server._broadcast_queue.qsize() == 3
        while not server._broadcast_queue.empty():
            server._broadcast_queue.get_nowait()
        assert server._broadcast_queue.qsize() == 0


class TestStateBuilderOptimization:
    def test_world_state_copy_efficiency(self):
        """get_world_state should return dict without holding lock on return."""
        from pycozmo_ng.navigation.world_map import WorldMap

        wm = WorldMap()
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=True)
        wm.update_cube_measurement(0, [0.0, 0.0, 200.0], __import__("numpy").eye(3), 0.9, 0.0)

        state1 = wm.get_world_state()
        state2 = wm.get_world_state()

        assert "robot" in state1
        assert "cubes" in state1
        assert "path_points" in state1
        assert state1["robot"]["position"] == state2["robot"]["position"]


class TestWebViewerServerInit:
    def test_no_cubes_attribute(self):
        """WebViewerServer should not have deprecated cubes dict."""
        server = WebViewerServer()
        assert not hasattr(server, "cubes")

    def test_has_world_map(self):
        server = WebViewerServer()
        assert server.world_map is not None

    def test_has_broadcast_interval(self):
        server = WebViewerServer()
        assert hasattr(server, "broadcast_interval_s")

    def test_broadcast_interval_default(self):
        server = WebViewerServer()
        assert server.broadcast_interval_s == pytest.approx(0.05)

    def test_broadcast_sets_dirty(self):
        server = WebViewerServer()
        server._broadcast_queue = MagicMock()
        server._broadcast_queue.qsize = lambda: 0
        server.broadcast()
        assert server._dirty is True

    def test_dirty_flag_resets_after_interval(self):
        """_dirty flag should be reset when broadcast interval elapsed."""
        server = WebViewerServer(broadcast_interval_s=0.0)
        server._dirty = True
        assert server._dirty is True
        # Simulate interval elapsed
        server._last_broadcast_time = 0.0
        now = time.time()
        if now - server._last_broadcast_time >= server.broadcast_interval_s and server._dirty:
            server._dirty = False
        assert server._dirty is False

    def test_uses_injected_world_map(self):
        """The composition root injects the world map; the server never creates the shared one."""
        from pycozmo_ng.navigation.world_map import WorldMap

        wm = WorldMap()
        server = WebViewerServer(world_map=wm)
        assert server.world_map is wm

    def test_router_is_exposed_for_action_registration(self):
        """Listeners are registered on the router in one call — no mirrored setters."""
        from pycozmo_ng.navigation.world_map import WorldMap

        wm = WorldMap()
        server = WebViewerServer(world_map=wm)
        assert server.router._world_map is wm
        cb = MagicMock()
        server.router.set_actions(control=cb, dock=cb)
        assert server.router.get_action("control") is cb
        # The old per-action setters are gone.
        assert not hasattr(server, "set_control_listener")
        assert not hasattr(server, "update_camera_settings")

    def test_standalone_creates_own_world_map(self):
        server = WebViewerServer()
        assert server.world_map is not None


class TestPointCloudDelivery:
    """The motion-stereo cloud serializes only when it changed."""

    @staticmethod
    def _server():
        return WebViewerServer()

    def test_push_stores_cloud(self):
        import numpy as np

        server = self._server()
        server.push_point_cloud(np.array([[1.0, 2.0, 3.0]]))
        assert server._point_cloud is not None
        assert server._point_cloud.shape == (1, 3)

    def test_push_empty_clears_cloud(self):
        import numpy as np

        server = self._server()
        server.push_point_cloud(np.array([[1.0, 2.0, 3.0]]))
        server.push_point_cloud(np.zeros((0, 3)))
        assert server._point_cloud is None

    def test_cloud_sent_only_when_changed(self):
        import json

        import numpy as np

        server = self._server()
        server.push_point_cloud(np.array([[1.0, 2.0, 3.0]]))

        state = json.loads(server._build_state_json())
        assert state["point_cloud"] == [2, 3, 1]

        # Unchanged cloud: the next message omits the field entirely.
        state = json.loads(server._build_state_json())
        assert "point_cloud" not in state

        # A new triangulation: present again.
        server.push_point_cloud(np.array([[4.0, 5.0, 6.0]]))
        state = json.loads(server._build_state_json())
        assert state["point_cloud"] == [5, 6, 4]

    def test_force_resends_unchanged_cloud(self):
        """A newly connected client gets the current cloud without waiting."""
        import json

        import numpy as np

        server = self._server()
        server.push_point_cloud(np.array([[1.0, 2.0, 3.0]]))
        server._build_state_json()  # consumes the change

        state = json.loads(server._build_state_json(force_point_cloud=True))
        assert state["point_cloud"] == [2, 3, 1]

    def test_no_cloud_field_when_never_pushed(self):
        import json

        server = self._server()
        state = json.loads(server._build_state_json(force_point_cloud=True))
        assert "point_cloud" not in state
