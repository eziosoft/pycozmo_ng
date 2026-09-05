"""
WebSocket + HTTP server for the Three.js web viewer.

Serves the browser-side Three.js scene and streams cube poses + robot state
over WebSocket. Drop-in replacement for World3DViewer in main.py.

Usage:
    # Standalone (serves HTML + WebSocket on port 8765):
    python -m pycozmo.web_viewer_server

    # Integrated into main.py:
    server = WebViewerServer()
    server.update_cube("cube_1", position, rotation_matrix, cube_size=44)
    server.broadcast()
"""

import asyncio
import json
import os
import time

import numpy as np

try:
    import websockets
    import websockets.server

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from aiohttp import web

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# ---------------------------------------------------------------------------
# Coordinate transform (same as World3DViewer.update_cube)
# ---------------------------------------------------------------------------
_T = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)


def _opencv_to_gl_position(position: np.ndarray) -> list:
    """Convert OpenCV camera coords (X right, Y down, Z forward) to
    OpenGL-style coords (X right, Y up, Z forward)."""
    pos = np.asarray(position, dtype=np.float32)
    gl = np.array([-pos[0], -pos[1] + 40, pos[2]], dtype=np.float32)
    return gl.tolist()


def _opencv_to_gl_rotation(rotation_matrix: np.ndarray) -> list:
    """Convert 3x3 rotation matrix from OpenCV to OpenGL coords."""
    R = np.asarray(rotation_matrix, dtype=np.float32)
    R_gl = _T @ R @ _T.T
    return R_gl.tolist()


# ---------------------------------------------------------------------------
# WebViewerServer
# ---------------------------------------------------------------------------
class WebViewerServer:
    """Headless replacement for World3DViewer.

    Same ``update_cube`` / ``remove_cube`` / ``clear_cubes`` interface so
    ``main.py`` can swap in with minimal changes.  Broadcasts JSON state
    over WebSocket every call to ``broadcast()``.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.cubes: dict[str, dict] = {}
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_time = time.time()

        # Validate that at least websockets is available
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "The 'websockets' package is required for WebViewerServer. "
                "Install it with: pip install websockets"
            )

        # We run the server in a background thread
        self._thread: asyncio.AbstractEventLoop | None = None
        self._server_started = asyncio.Event()

    # -- public interface (mirrors World3DViewer) ---------------------------

    def update_cube(
        self,
        cube_id: str,
        position,
        rotation_matrix,
        cube_size: float = 44,
        marker_images=None,
    ):
        """Update or add a cube — same signature as World3DViewer."""
        pos_gl = _opencv_to_gl_position(position)
        rot_gl = _opencv_to_gl_rotation(rotation_matrix)
        self.cubes[cube_id] = {
            "id": cube_id,
            "position": pos_gl,
            "rotation": rot_gl,
            "size": cube_size,
        }

    def remove_cube(self, cube_id: str):
        """Remove a cube from the world."""
        self.cubes.pop(cube_id, None)

    def clear_cubes(self):
        """Remove all cubes."""
        self.cubes.clear()

    def broadcast(self):
        """Serialize current state and send to all connected clients."""
        if not self._clients:
            return
        msg = self._build_state_json()
        asyncio.run_coroutine_threadsafe(self._broadcast_json(msg), self._loop)

    def render(self) -> bool:
        """No-op for compatibility — web viewer renders in browser."""
        return True

    def close(self):
        """Stop the background server."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)

    # -- internal -----------------------------------------------------------

    def _build_state_json(self) -> str:
        cube_list = list(self.cubes.values())
        state = {
            "cubes": cube_list,
            "robot": {
                "position": [0, 40, 0],
                "head_angle": 0.0,
            },
            "timestamp": int((time.time() - self._start_time) * 1000),
        }
        return json.dumps(state)

    async def _broadcast_json(self, msg: str):
        if self._clients:
            await asyncio.gather(
                *[c.send(msg) for c in list(self._clients)],
                return_exceptions=True,
            )

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            # Send current state immediately on connect
            await websocket.send(self._build_state_json())
            async for _ in websocket:
                pass  # We don't expect client messages
        finally:
            self._clients.discard(websocket)

    async def _shutdown(self):
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)

    # -- static file serving (aiohttp) -------------------------------------

    async def _serve_index(self, request):
        """Serve index.html from the web_viewer directory."""
        viewer_dir = os.path.join(os.path.dirname(__file__), "web_viewer")
        return web.FileResponse(os.path.join(viewer_dir, "index.html"))

    async def _serve_static(self, request):
        """Serve static files from the web_viewer directory."""
        viewer_dir = os.path.join(os.path.dirname(__file__), "web_viewer")
        filename = request.match_info["filename"]
        filepath = os.path.join(viewer_dir, filename)
        if os.path.isfile(filepath):
            return web.FileResponse(filepath)
        raise web.HTTPNotFound()

    # -- server lifecycle ---------------------------------------------------

    def start(self):
        """Start the WebSocket + HTTP server in a background thread."""
        if self._thread and self._thread.is_running():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _run_server(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        # WebSocket server
        ws_server = await websockets.serve(self._handler, self.host, self.port)

        # HTTP server (if aiohttp available) on port+1
        http_runner = None
        if HAS_AIOHTTP:
            app = web.Application()
            app.router.add_get("/", self._serve_index)
            app.router.add_get("/{filename}", self._serve_static)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self.host, self.port + 1)
            await site.start()
            http_runner = runner
            print(f"✅ Web viewer HTTP  at http://{self.host}:{self.port + 1}")
        else:
            print("⚠️  aiohttp not installed — HTTP serving disabled")
            print(
                "   Open the HTML file directly or install aiohttp: pip install aiohttp"
            )

        bind_note = " (all interfaces)" if self.host == "0.0.0.0" else ""
        print(f"✅ Web viewer WS    at ws://{self.host}:{self.port}{bind_note}")

        try:
            await asyncio.Future()  # run forever
        finally:
            ws_server.close()
            if http_runner:
                await http_runner.cleanup()


import threading


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    server = WebViewerServer()
    server.start()
    print("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping web viewer server...")
        server.close()


if __name__ == "__main__":
    main()
