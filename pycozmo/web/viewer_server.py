"""
WebSocket + HTTP server for the Three.js web viewer.

Serves the browser-side Three.js scene and streams world state over WebSocket.
The browser is a pure display: all localization math lives in ``WorldMap``
(pycozmo/world_map.py), and this server converts the world frame to scene
frame for rendering. Drop-in replacement for World3DViewer in main.py.

Usage:
    # Standalone (serves HTML + WebSocket on port 8765):
    python -m pycozmo.web.viewer_server

    # Integrated into main.py:
    server = WebViewerServer(world_map)
    server.update_cube_measurement(0, position, rotation, 0.9, head_angle)
    server.broadcast()
"""

import asyncio
import time

from pycozmo.logger import logger
from pycozmo.navigation.world_map import WorldMap
from pycozmo.web.file_server import FileServer
from pycozmo.web.mjpeg_streamer import MjpegEndpoint
from pycozmo.web.msg_router import MsgRouter
from pycozmo.web.state_builder import (
    OCCUPANCY_SEND_INTERVAL_S,
    build_state_json,
)
from pycozmo.web.ws_server import WSServer


# ---------------------------------------------------------------------------
# WebViewerServer — thin composition root
# ---------------------------------------------------------------------------
class WebViewerServer:
    """Headless replacement for World3DViewer — thin adapter over ``WorldMap``.

    Owns a :class:`WorldMap` (created unless one is supplied) and exposes
    pass-through methods that ``main.py`` calls from its event handlers. On
    broadcast, it serializes the world map to **scene frame** for the browser,
    which is a pure display and does no coordinate math.

    Internally delegates to:
    - :class:`MsgRouter` — browser message dispatch
    - :class:`WSServer` — WebSocket + HTTP lifecycle
    - :class:`FileServer` — static file serving
    - :class:`MjpegEndpoint` — annotated/floor MJPEG streams
    - :func:`build_state_json` — world→scene state serialization
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        world_map: WorldMap = None,
        broadcast_interval_s: float = 0.05,
    ):
        self.host = host
        self.port = port
        self.world_map = world_map if world_map is not None else WorldMap()
        self.broadcast_interval_s = broadcast_interval_s
        self._clients: set = set()
        self._start_time = time.time()
        self._last_broadcast_time = 0.0
        self._dirty = False  # state changed since last broadcast

        # Camera frame sources (image, frame_id) — thread-safe via GIL
        self._latest_annotated_image = None
        self._annotated_image_id: int = 0
        self._latest_floor_image = None
        self._floor_image_id: int = 0

        # Floor grid dimensions (mm)
        self._floor_size_mm: tuple[float, float] = (600.0, 600.0)
        self._floor_center_mm: tuple[float, float] = (200.0, 0.0)

        # Occupancy grid (base64-encoded PNG for WebSocket)
        self._latest_occupancy_png: str | None = None
        self._occupancy_origin: list[float] | None = None
        self._occupancy_seq: int = 0
        self._occupancy_seq_sent: int = -1
        self._last_occupancy_encode_time: float = 0.0

        # Occupancy map reference for A* pathfinding
        self.occupancy_map = None

        # Procedural face (base64-encoded PNG for WebSocket)
        self._latest_face_png: str | None = None

        # Broadcast coalescing
        self._broadcast_queue = None

        # Active-client transition callbacks (COZMO-22)
        self._active_clients_listeners: list = []
        self._last_active_count: int = 0

        # Latest robot telemetry
        self._telemetry: dict = {}

        # Sub-modules
        self._msg_router = MsgRouter(world_map=self.world_map)
        self._file_server = FileServer()

        # MJPEG endpoints — closures that return (image, frame_id)
        def _get_annotated_frame():
            return self._latest_annotated_image, self._annotated_image_id

        def _get_floor_frame():
            return self._latest_floor_image, self._floor_image_id

        self._mjpeg_annotated = MjpegEndpoint(_get_annotated_frame)
        self._mjpeg_floor = MjpegEndpoint(_get_floor_frame)

        self._ws_server = WSServer(
            host=host,
            port=port,
            ws_handler=self._handler,
            broadcast_fn=self._broadcast_loop,
            register_routes=self._register_http_routes,
        )

    # -- public interface (backwards compatible with World3DViewer) ---------

    def update_cube(
        self,
        cube_id: str,
        position,
        rotation_matrix,
        cube_size: float = 44,
        marker_images=None,
    ):
        """Deprecated — position is expected in world frame now."""
        del marker_images, cube_size
        self.world_map.update_cube_measurement(
            int(cube_id.replace("cube_", "")) - 1 if "_" in cube_id else 0,
            position,
            rotation_matrix,
            0.9,
            0.0,
        )

    def remove_cube(self, cube_id: str):
        """Remove a cube (compat shim)."""
        self.world_map.forget_cube(
            int(cube_id.replace("cube_", "")) - 1 if "_" in cube_id else 0
        )

    def clear_cubes(self):
        """Remove all cubes (compat shim)."""
        self.world_map.reset_map()

    # -- WorldMap pass-through (the real API) --------------------------------

    def update_robot_pose(self, position, yaw_rad, origin_id=None, record_path=True):
        """Update the robot's world pose (mm, Anki frame) from ``cli.pose``."""
        self.world_map.update_robot_pose(
            position, yaw_rad, origin_id=origin_id, record_path=record_path
        )

    def update_cube_measurement(
        self,
        cube_id,
        position_cam,
        rotation_cam,
        confidence,
        head_angle_rad,
        num_faces=None,
        camera_lateral_offset_mm=0.0,
        rotation_source=None,
    ):
        """Fuse a camera-frame cube measurement into the world map."""
        self.world_map.update_cube_measurement(
            cube_id,
            position_cam,
            rotation_cam,
            confidence,
            head_angle_rad,
            num_faces=num_faces,
            camera_lateral_offset_mm=camera_lateral_offset_mm,
            rotation_source=rotation_source,
        )

    def forget_cube(self, cube_id):
        """Drop a cube from the world map (e.g. on IMU-move)."""
        self.world_map.forget_cube(cube_id)

    def update_face_measurement(
        self,
        face_id,
        position_cam,
        confidence,
        head_angle_rad,
        camera_lateral_offset_mm=0.0,
        label="face",
    ):
        """Fuse a camera-frame face measurement into the world map."""
        self.world_map.update_face_measurement(
            face_id,
            position_cam,
            confidence,
            head_angle_rad,
            camera_lateral_offset_mm=camera_lateral_offset_mm,
            label=label,
        )

    def forget_face(self, face_id):
        """Remove a face from the world map."""
        self.world_map.forget_face(face_id)

    def add_path_point(self, position):
        """Append a world-frame position (mm) to the path trail."""
        self.world_map.add_path_point(position)

    def clear_path(self):
        """Clear the path trail."""
        self.world_map.reset_path()

    def reset_map(self):
        """Reset the whole world map (entities + path)."""
        self.world_map.reset_map()

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

    def render(self) -> bool:
        """No-op for compatibility — web viewer renders in browser."""
        return True

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

    def push_occupancy_grid(
        self, grid_image, origin_x: float = 0.0, origin_y: float = 0.0
    ) -> None:
        """Publish the occupancy grid as a base64-encoded PNG for WebSocket."""
        import base64

        import cv2

        now = time.time()
        if now - self._last_occupancy_encode_time < OCCUPANCY_SEND_INTERVAL_S:
            return
        self._last_occupancy_encode_time = now

        _, buf = cv2.imencode(".png", grid_image)
        self._latest_occupancy_png = base64.b64encode(buf.tobytes()).decode("ascii")
        self._occupancy_origin = [origin_x, origin_y]
        self._occupancy_seq += 1

    def set_floor_size(self, width_mm: float, height_mm: float) -> None:
        """Set the floor grid dimensions (mm) for the browser overlay."""
        self._floor_size_mm = (width_mm, height_mm)

    def set_floor_center(self, center_x_mm: float, center_y_mm: float) -> None:
        """Set the floor grid center (mm, body frame) for the browser overlay."""
        self._floor_center_mm = (center_x_mm, center_y_mm)

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
                asyncio.run_coroutine_threadsafe(
                    self._ws_server.shutdown(), self._ws_server.loop
                )
            except RuntimeError:
                pass

    def add_active_clients_listener(self, callback) -> None:
        """Subscribe to active-client-count transitions."""
        self._active_clients_listeners.append(callback)

    @property
    def active_client_count(self) -> int:
        """Return the current number of connected WebSocket clients."""
        return self._last_active_count

    def set_control_listener(self, callback) -> None:
        """Subscribe to browser key updates for remote control."""
        self._msg_router.set_control_listener(callback)

    def set_navigate_listener(self, callback) -> None:
        """Subscribe to browser click-to-navigate events."""
        self._msg_router.set_navigate_listener(callback)

    def set_approach_listener(self, callback) -> None:
        """Subscribe to browser cube-click events."""
        self._msg_router.set_approach_listener(callback)

    def set_dock_listener(self, callback) -> None:
        """Subscribe to browser dock requests."""
        self._msg_router.set_dock_listener(callback)

    def set_camera_settings_listener(self, callback) -> None:
        """Subscribe to browser camera-settings changes."""
        self._msg_router.set_camera_settings_listener(callback)

    def set_say_listener(self, callback) -> None:
        """Subscribe to browser TTS say events."""
        self._msg_router.set_say_listener(callback)

    def set_test_mode_listener(self, callback) -> None:
        """Subscribe to browser test mode toggle events."""
        self._msg_router.set_test_mode_listener(callback)

    def set_astar_toggle_listener(self, callback) -> None:
        """Subscribe to browser A* toggle events."""
        self._msg_router.set_astar_toggle_listener(callback)

    def update_camera_settings(self, **kwargs) -> None:
        """Update the stored camera settings (called from the listener)."""
        self._msg_router.set_camera_settings(kwargs)

    # -- internal -----------------------------------------------------------

    def _build_state_json(self, force_occupancy: bool = False) -> str:
        msg = build_state_json(
            self.world_map,
            self._telemetry,
            self._floor_size_mm,
            self._floor_center_mm,
            self._latest_occupancy_png,
            self._occupancy_origin,
            self._occupancy_seq,
            self._occupancy_seq_sent,
            self._start_time,
            force_occupancy,
            camera_settings=self._msg_router.camera_settings,
            face_png=self._latest_face_png,
        )
        if self._latest_occupancy_png is not None and (
            force_occupancy or self._occupancy_seq != self._occupancy_seq_sent
        ):
            self._occupancy_seq_sent = self._occupancy_seq
        return msg

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
        self._fire_active_clients_change()
        try:
            await websocket.send(self._build_state_json(force_occupancy=True))
            async for raw in websocket:
                self._msg_router.route(raw)
        finally:
            self._clients.discard(websocket)
            self._fire_active_clients_change()
            if not self._clients:
                listener = self._msg_router._control_listener
                if listener is not None:
                    try:
                        listener(set())
                    except Exception as e:  # noqa: BLE001
                        logger.warning("control listener (final stop) failed: %s", e)

    def _fire_active_clients_change(self) -> None:
        count = len(self._clients)
        if count == self._last_active_count:
            return
        self._last_active_count = count
        for listener in list(self._active_clients_listeners):
            try:
                listener(count)
            except Exception as e:  # noqa: BLE001
                logger.warning("active-clients listener failed: %s", e)

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
        app.router.add_get("/annotated-video", self._mjpeg_annotated.handle)
        app.router.add_get("/floor-video", self._mjpeg_floor.handle)
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
