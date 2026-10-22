"""
WebSocket + HTTP server for the Three.js web viewer.

Serves the browser-side Three.js scene and streams world state over WebSocket.
The browser is a pure display: all localization math lives in ``WorldMap``
(pycozmo_ng/navigation/world_map.py), and this server converts the world frame to
scene frame for rendering. Drop-in replacement for World3DViewer in main.py.

The server never owns the world map: the composition root (``Brain``)
constructs it and injects it here, and every other module writes to it
directly instead of reaching through this adapter.

Usage:
    # Standalone (serves HTML + WebSocket on port 8765):
    python -m pycozmo_ng.web.viewer_server

    # Integrated (Brain owns the world map):
    server = WebViewerServer(world_map=world_map)
    server.broadcast()
"""

import asyncio
import time

from pycozmo_ng.camera.service import CameraService
from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.web.aruco_texture import ArucoTextureEndpoint
from pycozmo_ng.web.file_server import FileServer
from pycozmo_ng.web.mjpeg_streamer import MjpegEndpoint
from pycozmo_ng.web.msg_router import MsgRouter
from pycozmo_ng.web.occupancy_delivery import OccupancyDelivery
from pycozmo_ng.web.state_builder import build_state_json
from pycozmo_ng.web.ws_server import WSServer


# ---------------------------------------------------------------------------
# WebViewerServer — thin composition root
# ---------------------------------------------------------------------------
class WebViewerServer:
    """Headless replacement for World3DViewer — thin adapter over ``WorldMap``.

    Holds an injected :class:`WorldMap` (created only in standalone mode)
    and serializes it to **scene frame** on broadcast for the browser,
    which is a pure display and does no coordinate math. It exposes no
    world-map write methods: callers write to the world map directly.
    It satisfies the :class:`~pycozmo_ng.vision.frame_sink.FrameSink`
    interface that ``FrameStage`` publishes frames through.
    The injected :class:`CameraService` is read-only here — the broadcast
    reports the camera settings actually in force.

    Internally delegates to:
    - :class:`MsgRouter` — browser message dispatch (register listeners in
      one call via ``server.router.set_actions(...)``)
    - :class:`WSServer` — WebSocket + HTTP lifecycle
    - :class:`FileServer` — static file serving
    - :class:`MjpegEndpoint` — raw/annotated/floor MJPEG streams
    - :class:`OccupancyDelivery` — when the occupancy grid PNG is sent
    - :func:`build_state_json` — world→scene state serialization
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        world_map: WorldMap | None = None,
        camera: CameraService | None = None,
        broadcast_interval_s: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.world_map = world_map if world_map is not None else WorldMap()
        self.camera = camera
        self.broadcast_interval_s = broadcast_interval_s
        self._clients: set = set()
        self._start_time = time.time()
        self._last_broadcast_time = 0.0
        self._dirty = False  # state changed since last broadcast

        # Camera frame sources (image, frame_id) — thread-safe via GIL
        self._latest_raw_image = None
        self._raw_image_id: int = 0
        self._latest_annotated_image = None
        self._annotated_image_id: int = 0
        self._latest_floor_image = None
        self._floor_image_id: int = 0

        # Floor grid dimensions (mm)
        self._floor_size_mm: tuple[float, float] = (600.0, 600.0)
        self._floor_center_mm: tuple[float, float] = (200.0, 0.0)

        # Occupancy grid delivery (encode throttle + changed/send decision)
        self.occupancy = OccupancyDelivery()

        # Procedural face (base64-encoded PNG for WebSocket)
        self._latest_face_png: str | None = None

        # Broadcast coalescing
        self._broadcast_queue = None

        # Latest robot telemetry
        self._telemetry: dict = {}

        # Sub-modules
        self.router = MsgRouter(world_map=self.world_map)
        self._file_server = FileServer()
        self._aruco_texture = ArucoTextureEndpoint()

        # MJPEG endpoints — closures that return (image, frame_id)
        def _get_raw_frame():
            return self._latest_raw_image, self._raw_image_id

        def _get_annotated_frame():
            return self._latest_annotated_image, self._annotated_image_id

        def _get_floor_frame():
            return self._latest_floor_image, self._floor_image_id

        self._mjpeg_raw = MjpegEndpoint(_get_raw_frame)
        self._mjpeg_annotated = MjpegEndpoint(_get_annotated_frame)
        self._mjpeg_floor = MjpegEndpoint(_get_floor_frame)

        self._ws_server = WSServer(
            host=host,
            port=port,
            ws_handler=self._handler,
            broadcast_fn=self._broadcast_loop,
            register_routes=self._register_http_routes,
        )

    # -- public interface (display concerns) --------------------------------

    def update_telemetry(self, **kwargs) -> None:
        """Store the latest robot telemetry for the browser HUD."""
        self._telemetry.update(kwargs)

    def broadcast(self):
        """Notify the server that the world state changed."""
        self._dirty = True
        if not self._clients or self._broadcast_queue is None:
            return
        loop = self._ws_server.loop
        if loop is None or not loop.is_running():
            return
        try:
            loop.call_soon_threadsafe(self._broadcast_queue.put_nowait, True)
        except RuntimeError:
            pass

    def push_raw_frame(self, image):
        """Publish the latest raw camera frame for MJPEG streaming."""
        self._latest_raw_image = image
        self._raw_image_id += 1

    def push_annotated_frame(self, image):
        """Publish the latest annotated camera frame for MJPEG streaming."""
        self._latest_annotated_image = image
        self._annotated_image_id += 1

    def push_floor_frame(self, image):
        """Publish the latest floor-projected frame for MJPEG."""
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image

        self._latest_floor_image = Image.fromarray(rgb)
        self._floor_image_id += 1

    def push_occupancy_frame(self, image, origin_x_mm: float, origin_y_mm: float) -> None:
        """Publish the occupancy grid image + world origin for the overlay."""
        self.occupancy.push(image, origin_x_mm, origin_y_mm)

    def set_floor_grid(self, size_mm: tuple[float, float], center_mm: tuple[float, float]) -> None:
        """Set the floor grid dimensions and center (mm) for the browser overlay."""
        self._floor_size_mm = size_mm
        self._floor_center_mm = center_mm

    def push_face_image(self, pil_image) -> None:
        """Publish the procedural face as a base64-encoded PNG for WebSocket."""
        import base64
        import io

        if pil_image is None:
            return
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        self._latest_face_png = base64.b64encode(buf.getvalue()).decode("ascii")

    def close(self):
        """Stop the background server."""
        if self._ws_server.loop and self._ws_server.loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._ws_server.shutdown(), self._ws_server.loop)
            except RuntimeError:
                pass

    # -- internal -----------------------------------------------------------

    def _build_state_json(self, force_occupancy: bool = False) -> str:
        return build_state_json(
            self.world_map,
            self._telemetry,
            self._floor_size_mm,
            self._floor_center_mm,
            self._start_time,
            self.occupancy.take(force=force_occupancy),
            camera_settings=self.camera.settings if self.camera is not None else None,
            face_png=self._latest_face_png,
        )

    async def _broadcast_loop(self):
        """Coalescing broadcaster: send the latest state at most once per
        broadcast_interval_s. Skips sending if nothing changed and the
        interval hasn't elapsed.
        """
        while True:
            await self._broadcast_queue.get()
            await asyncio.sleep(0.001)
            while not self._broadcast_queue.empty():
                self._broadcast_queue.get_nowait()
            if not self._clients:
                continue
            now = time.time()
            if now - self._last_broadcast_time < self.broadcast_interval_s:
                continue
            if not self._dirty:
                continue
            self._dirty = False
            self._last_broadcast_time = now
            msg = self._build_state_json()
            await self._send_to_all(msg)

    async def _send_to_all(self, msg: str):
        """Send to each client individually with its own timeout."""
        for client in list(self._clients):
            try:
                await asyncio.wait_for(client.send(msg), timeout=1.0)
            except (asyncio.TimeoutError, ConnectionError, OSError):
                pass

    async def _handler(self, websocket):
        self._clients.add(websocket)
        try:
            await websocket.send(self._build_state_json(force_occupancy=True))
            async for raw in websocket:
                self.router.route(raw)
        finally:
            self._clients.discard(websocket)
            if not self._clients:
                listener = self.router.get_action("control")
                if listener is not None:
                    try:
                        listener(set())
                    except Exception as e:  # noqa: BLE001
                        logger.warning("control listener (final stop) failed: %s", e)

    # -- server lifecycle ---------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the WebSocket + HTTP server thread is alive."""
        thread = self._ws_server._thread
        return thread is not None and thread.is_alive()

    def start(self):
        """Start the WebSocket + HTTP server in a background thread."""
        if self._ws_server._thread and self._ws_server._thread.is_alive():
            return
        self._broadcast_queue = asyncio.Queue()
        self._ws_server.start()

    def _register_http_routes(self, app) -> None:
        """Mount FileServer + MjpegEndpoint routes on the aiohttp app."""
        app.router.add_get("/", self._file_server.serve_index)
        app.router.add_get("/raw-video", self._mjpeg_raw.handle)
        app.router.add_get("/annotated-video", self._mjpeg_annotated.handle)
        app.router.add_get("/floor-video", self._mjpeg_floor.handle)
        # Must precede the /{filename} catch-all, which would swallow it.
        app.router.add_get("/aruco-texture", self._aruco_texture.handle)
        app.router.add_get("/{filename}", self._file_server.serve_static)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
def main():
    server = WebViewerServer()
    server.start()
    logger.info("Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping web viewer server...")
        server.close()


if __name__ == "__main__":
    main()
