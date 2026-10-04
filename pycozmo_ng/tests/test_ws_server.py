"""Tests for check_port_available (COZMO-41).

Tests the cross-platform port probe utility.
"""

import socket
import threading

from pycozmo_ng.web.ws_server import check_port_available


class TestCheckPortAvailable:
    def test_available_port(self):
        assert check_port_available("127.0.0.1", 19876) is True

    def test_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 19877))
            s.listen(1)
            assert check_port_available("127.0.0.1", 19877) is False

    def test_bad_host_returns_true(self):
        assert check_port_available("192.0.2.1", 19878, timeout=0.1) is True

    def test_custom_timeout(self):
        result = check_port_available("127.0.0.1", 19879, timeout=0.01)
        assert isinstance(result, bool)

    def test_threaded_occupied(self):
        bound = threading.Event()
        ready = threading.Event()

        def serve():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", 19880))
                s.listen(1)
                bound.set()
                ready.wait(timeout=2.0)

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        bound.wait(timeout=2.0)
        assert check_port_available("127.0.0.1", 19880) is False
        ready.set()
