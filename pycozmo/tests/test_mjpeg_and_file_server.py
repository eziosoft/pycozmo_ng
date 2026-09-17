"""Tests for MjpegEndpoint and FileServer (COZMO-41).

Tests MJPEG frame generation and static file serving.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from PIL import Image

from pycozmo.web.file_server import FileServer
from pycozmo.web.mjpeg_streamer import MjpegEndpoint


def _run_async(coro):
    """Run an async coroutine synchronously for testing."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_image():
    return Image.fromarray(np.zeros((240, 320, 3), dtype=np.uint8))


def _make_aiohttp_request():
    """Create a minimal mock aiohttp request with required attributes."""
    from aiohttp import HttpVersion

    req = MagicMock()
    req.version = HttpVersion(1, 1)
    req.keep_alive = True
    return req


class TestMjpegEndpoint:
    def test_get_frame_called(self):
        get_frame = MagicMock(return_value=(_make_image(), 1))
        endpoint = MjpegEndpoint(get_frame)
        _img, fid = endpoint._get_frame()
        get_frame.assert_called()
        assert fid == 1

    def test_encode_quality_stored(self):
        endpoint = MjpegEndpoint(lambda: (None, 0), encode_quality=85)
        assert endpoint._encode_quality == 85

    def test_default_encode_quality(self):
        endpoint = MjpegEndpoint(lambda: (None, 0))
        assert endpoint._encode_quality == 70

    def test_handle_returns_coroutine(self):
        endpoint = MjpegEndpoint(lambda: (None, 0))
        result = endpoint.handle(_make_aiohttp_request())
        assert asyncio.iscoroutine(result)
        result.close()

    @patch("pycozmo.web.mjpeg_streamer.encode_jpeg")
    def test_handle_encodes_jpeg(self, mock_encode):
        mock_encode.return_value = b"\xff\xd8" + b"\x00" * 100
        frame_counter = 0

        def get_frame():
            nonlocal frame_counter
            frame_counter += 1
            return _make_image(), frame_counter

        endpoint = MjpegEndpoint(get_frame, encode_quality=50)
        written = []

        async def fake_write(data):
            written.append(data)

        async def run_and_cancel():
            task = asyncio.ensure_future(endpoint.handle(_make_aiohttp_request()))
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        with patch("aiohttp.web.StreamResponse") as MockSR:
            mock_response = AsyncMock()
            mock_response.write = fake_write
            mock_response.prepare = AsyncMock()
            MockSR.return_value = mock_response

            _run_async(run_and_cancel())

            assert len(written) >= 2
            mock_encode.assert_called()

    def test_handle_connection_reset_stops(self):
        call_count = 0

        def get_frame():
            nonlocal call_count
            call_count += 1
            return _make_image(), call_count

        endpoint = MjpegEndpoint(get_frame)

        async def fake_write(data):
            raise ConnectionResetError("lost")

        with patch("aiohttp.web.StreamResponse") as MockSR:
            mock_response = AsyncMock()
            mock_response.write = fake_write
            mock_response.prepare = AsyncMock()
            MockSR.return_value = mock_response

            async def run():
                await endpoint.handle(_make_aiohttp_request())

            _run_async(run())
            assert call_count >= 1

    def test_handle_connection_error_stops(self):
        def get_frame():
            return _make_image(), 1

        endpoint = MjpegEndpoint(get_frame)

        async def fake_write(data):
            raise ConnectionError("closed")

        with patch("aiohttp.web.StreamResponse") as MockSR:
            mock_response = AsyncMock()
            mock_response.write = fake_write
            mock_response.prepare = AsyncMock()
            MockSR.return_value = mock_response

            async def run():
                await endpoint.handle(_make_aiohttp_request())

            _run_async(run())


class TestFileServer:
    def test_serve_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "index.html")
            with open(index_path, "w") as f:
                f.write("<html>test</html>")

            fs = FileServer(static_dir=tmpdir)
            request = MagicMock()
            response = _run_async(fs.serve_index(request))
            assert response is not None

    def test_serve_static_found(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "app.js")
            with open(file_path, "w") as f:
                f.write("console.log('hi');")

            fs = FileServer(static_dir=tmpdir)
            request = MagicMock()
            request.match_info = {"filename": "app.js"}
            response = _run_async(fs.serve_static(request))
            assert response is not None

    def test_serve_static_not_found(self):
        from aiohttp import web

        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileServer(static_dir=tmpdir)
            request = MagicMock()
            request.match_info = {"filename": "nonexistent.js"}
            try:
                _run_async(fs.serve_static(request))
                assert False, "Should have raised HTTPNotFound"
            except web.HTTPNotFound:
                pass

    def test_default_static_dir(self):
        FileServer()
        static_dir = os.path.join(os.path.dirname(__file__), "..", "web", "static")
        assert os.path.isdir(os.path.abspath(static_dir))
