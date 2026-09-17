"""MJPEG streaming endpoints for the Three.js web viewer.

Provides a reusable ``MjpegEndpoint`` that any aiohttp app can mount.
Extracted from ``WebViewerServer`` — each endpoint reads from a
thread-safe (image, frame_id) pair and JPEG-encodes on the event loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pycozmo.web.state_builder import encode_jpeg


class MjpegEndpoint:
    """Single MJPEG streaming endpoint backed by a (image, frame_id) source.

    The *get_frame* callable must return ``(image, frame_id)`` where
    *image* is a PIL Image (or ``None``) and *frame_id* is a monotonically
    increasing int.  ``encode_quality`` controls JPEG compression (1-100).
    """

    def __init__(
        self,
        get_frame: Callable[..., tuple[Any, int]],
        *,
        encode_quality: int = 70,
    ) -> None:
        self._get_frame = get_frame
        self._encode_quality = encode_quality

    async def handle(self, request):
        """aiohttp handler — streams JPEG frames as multipart/x-mixed-replace."""
        from aiohttp import web

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await response.prepare(request)

        boundary = "frame"
        last_frame_id = -1
        while True:
            try:
                image, frame_id = self._get_frame()
                if frame_id == last_frame_id or image is None:
                    await asyncio.sleep(0.01)
                    continue

                last_frame_id = frame_id
                jpeg = await asyncio.to_thread(encode_jpeg, image, self._encode_quality)
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode()
                await response.write(header + jpeg + b"\r\n")
            except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
                break

        return response
