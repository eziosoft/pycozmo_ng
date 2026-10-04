"""WebSocket lifecycle, broadcast coalescing, and HTTP setup.

Owns the asyncio loop thread, the websockets server, and (optionally)
an aiohttp HTTP server on ``port + 1``.  Composed into
``WebViewerServer`` which provides the public API.

The ``FileServer`` and ``MjpegEndpoint`` instances are mounted on the
HTTP app by the caller (``WebViewerServer._serve``).
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections.abc import Callable
from typing import Any

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from pycozmo_ng.logger import logger


class WSServer:
    """Low-level WebSocket + optional HTTP server.

    Parameters
    ----------
    host, port:
        Bind address for the WebSocket server.  The HTTP server (if
        aiohttp is available) binds to ``host:port + 1``.
    ws_handler:
        ``async def handler(websocket)`` called for each new connection.
    broadcast_fn:
        ``async def broadcast()`` invoked in a coalescing loop whenever
        the world state changes.
    register_routes:
        Optional ``def register(app)`` called to mount routes on the
        aiohttp application before the runner starts.
    """

    def __init__(
        self,
        host: str,
        port: int,
        ws_handler: Callable[..., Any],
        broadcast_fn: Callable[..., Any],
        register_routes: Callable[..., Any] | None = None,
    ) -> None:
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "The 'websockets' package is required for WebViewerServer. Install it with: pip install websockets"
            )
        self.host = host
        self.port = port
        self._ws_handler = ws_handler
        self._broadcast_fn = broadcast_fn
        self._register_routes = register_routes

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_forever: asyncio.Future | None = None

    # -- lifecycle -----------------------------------------------------------

    @property
    def loop(self):
        return self._loop

    def start(self):
        """Start the servers in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        self._run_forever = asyncio.get_running_loop().create_future()
        broadcast_task = asyncio.create_task(self._broadcast_fn())

        ws_server = await websockets.serve(
            self._ws_handler,
            self.host,
            self.port,
            reuse_address=True,
        )

        http_runner = None
        if HAS_AIOHTTP:
            app = web.Application()
            if self._register_routes is not None:
                self._register_routes(app)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port + 1)
            await site.start()
            http_runner = runner
            logger.info("Web viewer HTTP at http://%s:%s", self.host, self.port + 1)
        else:
            logger.warning("aiohttp not installed — HTTP serving disabled")
            logger.warning("Open the HTML file directly or install aiohttp: pip install aiohttp")

        bind_note = " (all interfaces)" if self.host == "0.0.0.0" else ""
        logger.info("Web viewer WS at ws://%s:%s%s", self.host, self.port, bind_note)

        try:
            await self._run_forever
        finally:
            broadcast_task.cancel()
            ws_server.close()
            if http_runner:
                await http_runner.cleanup()

    async def shutdown(self):
        """Stop the servers."""
        if self._run_forever is not None and not self._run_forever.done():
            self._run_forever.set_result(None)


def check_port_available(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if *port* is not already bound on *host*.

    Uses a lightweight TCP connect probe instead of spawning ``lsof`` or
    ``kill -9`` — cross-platform and non-destructive.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) != 0
    except OSError:
        return True
