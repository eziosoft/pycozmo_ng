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
import threading

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


def _encode_jpeg(image, quality: int = 70) -> bytes:
    """Encode a PIL image to JPEG bytes (runs on a worker thread).

    The image is the raw frame from Cozmo — no brightness/gamma
    post-processing. If you want to tune the look, send
    ``SetCameraParams`` to Cozmo (gain, exposure, auto-exposure) via
    ``cli.set_camera_params(...)`` instead. (COZMO-24)
    """
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


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
    over WebSocket (coalesced, never queued up) every call to ``broadcast()``.
    Streams camera frames as MJPEG via ``push_frame()``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
    ):
        self.host = host
        self.port = port
        self.cubes: dict[str, dict] = {}
        self._robot_position: list[float] = [0.0, 0.0, 0.0]
        self._robot_heading: float = 0.0
        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_time = time.time()

        # Camera MJPEG streaming (encode happens in the server loop thread)
        self._latest_image = None
        self._image_id: int = 0

        # Broadcast coalescing: the loop thread keeps a queue and always
        # sends the latest state — broadcasts never pile up.
        self._broadcast_queue = None
        self._run_forever = None

        # Active-client transition callbacks. Use ``add_active_clients_listener``
        # to subscribe; the callback fires with the new count (int) whenever
        # the count changes (including going to 0 when the last client
        # disconnects). Typical use: enable Cozmo's camera on the 0→1
        # transition and disable it on the 1→0 transition, so the camera
        # only runs while someone is actually watching. (COZMO-22)
        self._active_clients_listeners: list = []
        self._last_active_count: int = 0

        # Remote-control key updates. The browser sends a
        # ``{"type": "control", "keys": [...]}`` message on every keydown/
        # keyup; the most recent set is forwarded to the listener (e.g. a
        # :class:`pycozmo.remote_control.RemoteControl` instance) which
        # translates it into Cozmo commands. (COZMO-25)
        self._control_listener = None

        # Validate that at least websockets is available
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "The 'websockets' package is required for WebViewerServer. "
                "Install it with: pip install websockets"
            )

        # We run the server in a background thread
        self._thread: threading.Thread | None = None

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

    def set_robot_state(self, position, heading):
        """Update the robot's world-frame pose for the web viewer.

        ``position`` is the robot's position in world frame in mm
        (Anki convention: X right, Y forward, Z up — same as ``cli.pose.position``).
        ``heading`` is the rotation about world Z in radians
        (``cli.pose.rotation.angle_z.radians``). The browser uses these to
        frame the initial camera view along Cozmo's heading instead of
        always facing the +Z default, so the user doesn't have to rotate
        the view on every page load.
        """
        import numpy as _np

        pos = _np.asarray(position, dtype=_np.float32).reshape(3)
        self._robot_position = [float(pos[0]), float(pos[1]), float(pos[2])]
        self._robot_heading = float(heading)

    def broadcast(self):
        """Notify the server that the world state changed.

        The server coalesces notifications and sends the latest state once,
        so rapid ``broadcast()`` calls from the client thread can never
        queue up and starve the WebSocket/MJPEG handlers on the loop.
        """
        if not self._clients or self._broadcast_queue is None:
            return
        if self._loop is None or not self._loop.is_running():
            return
        try:
            self._loop.call_soon_threadsafe(self._broadcast_queue.put_nowait, True)
        except RuntimeError:
            pass

    def render(self) -> bool:
        """No-op for compatibility — web viewer renders in browser."""
        return True

    def push_frame(self, image):
        """Publish the latest camera frame (PIL image) for MJPEG streaming.

        JPEG encoding is deferred to the server's event loop, so this never
        blocks the caller (the pycozmo packet-processing thread).
        """
        self._latest_image = image
        self._image_id += 1

    def close(self):
        """Stop the background server."""
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            except RuntimeError:
                pass  # Loop already closed

    def add_active_clients_listener(self, callback) -> None:
        """Subscribe to active-client-count transitions.

        ``callback(active_count: int)`` is called whenever the number of
        connected WebSocket clients changes (0→1, 1→0, 1→2, etc.). Multiple
        listeners may be registered. Typical use: turn Cozmo's camera on
        when the first browser tab opens and off when the last one closes,
        so the camera stream only runs while someone is watching —
        eliminates the sustained 30 fps load that drops the Wi-Fi.
        (COZMO-22)

        The callback is invoked on the server's asyncio event-loop thread;
        keep it short and non-blocking, or use ``call_soon_threadsafe`` to
        hand off work to another thread.
        """
        self._active_clients_listeners.append(callback)

    def set_control_listener(self, callback) -> None:
        """Subscribe to browser key updates for remote control.

        ``callback(keys: set[str])`` is invoked (on the asyncio loop
        thread) whenever a ``{"type": "control", "keys": [...]}`` message
        arrives from any connected browser. The set is already normalized
        to lowercase single-character keys. Pass ``None`` to unsubscribe.
        Only the most recent listener is stored — remote control is a
        single-user feature. (COZMO-25)
        """
        self._control_listener = callback

    # -- internal -----------------------------------------------------------

    def _build_state_json(self) -> str:
        cube_list = list(self.cubes.values())
        state = {
            "cubes": cube_list,
            "robot": {
                "position": list(self._robot_position),
                "heading": self._robot_heading,
            },
            "timestamp": int((time.time() - self._start_time) * 1000),
        }
        return json.dumps(state)

    async def _broadcast_loop(self):
        """Coalescing broadcaster: send the latest state, never a backlog.

        Waits for a ``broadcast()`` notification, drains any burst that
        arrived meanwhile, and sends a single, up-to-date state message.
        """
        while True:
            await self._broadcast_queue.get()
            await asyncio.sleep(0.005)  # coalesce bursts within one frame
            while not self._broadcast_queue.empty():
                self._broadcast_queue.get_nowait()
            if not self._clients:
                continue
            msg = self._build_state_json()
            await self._send_to_all(msg)

    async def _send_to_all(self, msg: str):
        try:
            await asyncio.wait_for(
                asyncio.gather(
                    *[c.send(msg) for c in list(self._clients)],
                    return_exceptions=True,
                ),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            # A client is too slow to keep up — drop this round rather than
            # blocking the event loop (and with it, the MJPEG stream).
            pass

    async def _handler(self, websocket):
        self._clients.add(websocket)
        self._fire_active_clients_change()
        try:
            # Send current state immediately on connect
            await websocket.send(self._build_state_json())
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                mtype = msg.get("type")
                if mtype == "config":
                    # Config messages from the browser were historically
                    # used to set brightness/gamma. Those are gone now
                    # (see COZMO-24) — the image is the raw Cozmo feed.
                    # Anything still arriving on this channel is from an
                    # old version of the viewer and is ignored.
                    pass
                elif mtype == "control":
                    # Remote-control key update from the browser. The
                    # listener (e.g. a RemoteControl instance) decides
                    # what to do with the keys. We hand off the raw set;
                    # the listener is responsible for any thread/async
                    # marshalling. (COZMO-25)
                    keys = msg.get("keys") or []
                    print(f"🎮 control: keys={keys}")
                    listener = self._control_listener
                    if listener is not None:
                        try:
                            listener(set(keys))
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  control listener failed: {e}")
                    else:
                        print("⚠️  control: no listener registered")
        finally:
            self._clients.discard(websocket)
            self._fire_active_clients_change()
            # Last browser tab gone. Make sure the controller stops
            # driving Cozmo (its idle-stop window will also catch this,
            # but stopping immediately avoids a brief coast). (COZMO-25)
            if not self._clients:
                listener = self._control_listener
                if listener is not None:
                    try:
                        listener(set())
                    except Exception as e:  # noqa: BLE001
                        print(f"⚠️  control listener (final stop) failed: {e}")

    def _fire_active_clients_change(self) -> None:
        """Notify listeners when the active-client count changes.

        Runs on the asyncio event-loop thread (it's called from
        ``_handler``). Listeners are run inline — they should be
        non-blocking or hand off to another thread. Errors are swallowed
        so a buggy listener doesn't break the WebSocket. (COZMO-22)
        """
        count = len(self._clients)
        if count == self._last_active_count:
            return
        self._last_active_count = count
        for listener in list(self._active_clients_listeners):
            try:
                listener(count)
            except Exception as e:
                print(f"⚠️  active-clients listener failed: {e}")

    async def _shutdown(self):
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
        if self._run_forever is not None and not self._run_forever.done():
            self._run_forever.set_result(None)

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

    async def _serve_video(self, request):
        """MJPEG streaming endpoint — pushes latest camera frame to browser."""
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
                current_id = self._image_id
                image = self._latest_image
                if current_id == last_frame_id or image is None:
                    await asyncio.sleep(0.01)
                    continue

                last_frame_id = current_id
                jpeg = await asyncio.to_thread(_encode_jpeg, image, 70)
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg)}\r\n\r\n"
                ).encode()
                await response.write(header + jpeg + b"\r\n")
            except (ConnectionResetError, ConnectionError, asyncio.CancelledError):
                break

        return response

    # -- server lifecycle ---------------------------------------------------

    def start(self):
        """Start the WebSocket + HTTP server in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._kill_stale_ports()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def _kill_stale_ports(self):
        """Kill any process still holding our ports from a previous run."""
        import socket
        import subprocess

        for port in (self.port, self.port + 1):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                result = s.connect_ex(("127.0.0.1", port))
                s.close()
                if result == 0:
                    # Port in use — find and kill the process
                    out = subprocess.run(
                        ["lsof", "-ti", f":{port}"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                        check=False,
                    )
                    for pid in out.stdout.strip().splitlines():
                        subprocess.run(
                            ["kill", "-9", pid],
                            capture_output=True,
                            timeout=3,
                            check=False,
                        )
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _run_server(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        # Broadcast queue (created and used only on the loop thread)
        self._broadcast_queue = asyncio.Queue()
        broadcast_task = asyncio.create_task(self._broadcast_loop())

        # WebSocket server
        ws_server = await websockets.serve(
            self._handler,
            self.host,
            self.port,
            reuse_address=True,
        )

        # HTTP server (if aiohttp available) on port+1
        http_runner = None
        if HAS_AIOHTTP:
            app = web.Application()
            app.router.add_get("/", self._serve_index)
            app.router.add_get("/video", self._serve_video)
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

        self._run_forever = asyncio.get_running_loop().create_future()
        try:
            await self._run_forever
        finally:
            broadcast_task.cancel()
            ws_server.close()
            if http_runner:
                await http_runner.cleanup()


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
