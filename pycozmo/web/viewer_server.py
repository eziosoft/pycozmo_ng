"""
WebSocket + HTTP server for the Three.js web viewer.

Serves the browser-side Three.js scene and streams world state over WebSocket.
The browser is a pure display: all localization math lives in ``WorldMap``
(pycozmo/world_map.py), and this server converts the world frame to scene frame
for rendering. Drop-in replacement for World3DViewer in main.py.

Usage:
    # Standalone (serves HTML + WebSocket on port 8765):
    python -m pycozmo.web.viewer_server

    # Integrated into main.py:
    server = WebViewerServer(world_map)
    server.update_cube_measurement(0, position, rotation, 0.9, head_angle)
    server.broadcast()
"""

import asyncio
import json
import os
import threading
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

from pycozmo import coords
from pycozmo.navigation.world_map import WorldMap

# How far (mm) to lift the rendered path trail above the scene ground plane
# (scene +Y). The trail is a zero-thickness line at ground level, so without a
# small offset it z-fights / clips the grid floor in the browser. This is a
# pure display artifact — it is NOT applied to world-frame data or cubes/robot.
PATH_SCENE_LIFT_MM: float = 3.0


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


# ---------------------------------------------------------------------------
# WebViewerServer
# ---------------------------------------------------------------------------
class WebViewerServer:
    """Headless replacement for World3DViewer — thin adapter over ``WorldMap``.

    Owns a :class:`WorldMap` (created unless one is supplied) and exposes
    pass-through methods that ``main.py`` calls from its event handlers. On
    broadcast, it serializes the world map to **scene frame** for the browser,
    which is a pure display and does no coordinate math.

    Same ``update_cube`` / ``remove_cube`` / ``clear_cubes`` interface is kept
    for backwards compatibility, but new code should use the WorldMap-backed
    methods below.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        world_map: WorldMap = None,
    ):
        self.host = host
        self.port = port
        self.world_map = world_map if world_map is not None else WorldMap()
        self.cubes: dict[
            str, dict
        ] = {}  # kept for compat; world_map is source of truth

        self._clients: set = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start_time = time.time()

        # Camera MJPEG streaming (encode happens in the server loop thread)
        self._latest_image = None
        self._image_id: int = 0

        # Floor MJPEG streaming
        self._latest_floor_image = None
        self._floor_image_id: int = 0

        # Floor grid dimensions (mm) — set from FloorProjector config
        self._floor_size_mm: tuple[float, float] = (600.0, 600.0)
        self._floor_center_mm: tuple[float, float] = (200.0, 0.0)

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
        # translates it into Cozmo commands. We hand off the raw set;
        # the listener is responsible for any thread/async
        # marshalling. (COZMO-25)
        self._control_listener = None

        # Navigation target listener. The browser sends a
        # ``{"type": "navigate_to", "x": scene_x, "z": scene_z}`` message
        # on ground-plane click; the listener (e.g. a navigation handler in
        # main.py) receives world-frame coordinates. (COZMO-28)
        self._navigate_listener = None

        # Cube approach listener. The browser sends a
        # ``{"type": "approach_cube", "cube_id": "cube_1"}`` message on
        # cube click; the listener receives (cube_id_int, cube_pos_world).
        # (COZMO-28)
        self._approach_listener = None

        # Dock listener. The browser sends a ``{"type": "dock"}`` message
        # on 'z' keypress; the listener drives the robot back into the
        # dock. (COZMO-28)
        self._dock_listener = None

        # Latest robot telemetry (battery, charging, speed, head/lift).
        # Updated from ``main.py``'s RobotState handler; included in the
        # broadcast state so the browser can render a telemetry HUD.
        self._telemetry: dict = {}

        # Validate that at least websockets is available
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "The 'websockets' package is required for WebViewerServer. "
                "Install it with: pip install websockets"
            )

        # We run the server in a background thread
        self._thread: threading.Thread | None = None

    # -- public interface (backwards compatible with World3DViewer) ---------

    def update_cube(
        self,
        cube_id: str,
        position,
        rotation_matrix,
        cube_size: float = 44,
        marker_images=None,
    ):
        """Deprecated — position is expected in world frame now.

        Kept to avoid crashing old callers, but the world-map path
        (``update_cube_measurement``) is preferred because it transforms and
        fuses correctly. This stores the value into scene frame directly.
        """
        del marker_images, cube_size
        pos_gl = coords.world_to_scene_point(position)
        rot_gl = coords.world_to_scene_rotation(rotation_matrix)
        self.cubes[cube_id] = {
            "id": cube_id,
            "position": [float(v) for v in pos_gl],
            "rotation": [row.tolist() for row in rot_gl],
            "size": 44,
        }

    def remove_cube(self, cube_id: str):
        """Remove a cube (compat shim)."""
        self.cubes.pop(cube_id, None)

    def clear_cubes(self):
        """Remove all cubes (compat shim)."""
        self.cubes.clear()

    # -- WorldMap pass-through (the real API) -------------------------------

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
    ):
        """Fuse a camera-frame cube measurement into the world map."""
        self.world_map.update_cube_measurement(
            cube_id,
            position_cam,
            rotation_cam,
            confidence,
            head_angle_rad,
            num_faces=num_faces,
        )

    def forget_cube(self, cube_id):
        """Drop a cube from the world map (e.g. on IMU-move)."""
        self.world_map.forget_cube(cube_id)

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
        """Store the latest robot telemetry for the browser HUD.

        Called from ``main.py``'s RobotState handler with named fields:
        ``charging`` (bool), ``on_charger`` (bool), ``battery_voltage``
        (float, V), ``left_wheel_speed`` / ``right_wheel_speed`` (float,
        mm/s), ``head_angle`` (float, degrees) and ``lift_height``
        (float, mm). Included in the next broadcast.
        """
        self._telemetry.update(kwargs)

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

    def push_floor_frame(self, image):
        """Publish the latest floor-projected frame (numpy BGR) for MJPEG."""
        import cv2

        # Convert BGR numpy to PIL RGB for the encoder
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        from PIL import Image

        self._latest_floor_image = Image.fromarray(rgb)
        self._floor_image_id += 1

    def set_floor_size(self, width_mm: float, height_mm: float) -> None:
        """Set the floor grid dimensions (mm) for the browser overlay."""
        self._floor_size_mm = (width_mm, height_mm)

    def set_floor_center(self, center_x_mm: float, center_y_mm: float) -> None:
        """Set the floor grid center (mm, body frame) for the browser overlay."""
        self._floor_center_mm = (center_x_mm, center_y_mm)

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

    def set_navigate_listener(self, callback) -> None:
        """Subscribe to browser click-to-navigate events.

        ``callback(target_world_x, target_world_z)`` is invoked (on the
        asyncio loop thread) whenever a ``{"type": "navigate_to", "x":,
        "z":}`` message arrives. Coordinates are converted from scene
        frame to world frame before delivery. Pass ``None`` to unsubscribe.

        ``callback(None)`` is invoked on ``{"type": "navigate_cancel"}``.
        (COZMO-28)
        """
        self._navigate_listener = callback

    def set_approach_listener(self, callback) -> None:
        """Subscribe to browser cube-click events.

        ``callback(cube_id_int, cube_pos_world_x, cube_pos_world_y)``
        is invoked whenever a ``{"type": "approach_cube", "cube_id": ...}``
        message arrives. Looks up the cube's current world position from
        the world map. Pass ``None`` to unsubscribe. (COZMO-28)
        """
        self._approach_listener = callback

    def set_dock_listener(self, callback) -> None:
        """Subscribe to browser dock requests.

        ``callback()`` is invoked whenever a ``{"type": "dock"}`` message
        arrives (user presses 'z'). The listener drives the robot back
        into the dock. Pass ``None`` to unsubscribe. (COZMO-28)
        """
        self._dock_listener = callback

    # -- internal -----------------------------------------------------------

    def _build_state_json(self) -> str:
        world = self.world_map.get_world_state()

        # Robot: world frame -> scene frame, with yaw mapped to scene rotation.y.
        robot_pos = coords.world_to_scene_point(world["robot"]["position"])
        robot_heading = coords.world_yaw_to_scene(world["robot"]["yaw"])

        # Cubes: world frame -> scene frame.
        cube_list = []
        for cube_id, cube_state in world["cubes"].items():
            pos_gl = coords.world_to_scene_point(cube_state["position"])
            rot_gl = coords.world_to_scene_rotation(cube_state["rotation"])
            cube_list.append(
                {
                    "id": f"cube_{cube_id + 1}",
                    "position": [float(v) for v in pos_gl],
                    "rotation": [row.tolist() for row in rot_gl],
                    "size": 44,
                }
            )

        # Path: world frame -> scene frame, lifted just above the ground so the
        # zero-thickness trail doesn't z-fight the grid floor.
        path_scene = [
            [float(v) for v in coords.world_to_scene_point(p)]
            for p in world["path_points"]
        ]
        path_scene = [[x, y + PATH_SCENE_LIFT_MM, z] for x, y, z in path_scene]

        state = {
            "cubes": cube_list,
            "robot": {
                "position": [float(v) for v in robot_pos],
                "heading": float(robot_heading),
            },
            "path_points": path_scene,
            "telemetry": dict(self._telemetry),
            "floor_size": list(self._floor_size_mm),
            "floor_center": list(self._floor_center_mm),
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
                    listener = self._control_listener
                    if listener is not None:
                        try:
                            listener(set(keys))
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  control listener failed: {e}")
                elif mtype == "navigate_to":
                    # Click-to-navigate from browser. Convert scene coords
                    # to world frame and forward to listener. (COZMO-28)
                    listener = self._navigate_listener
                    if listener is not None:
                        try:
                            scene_x = msg.get("x", 0)
                            scene_z = msg.get("z", 0)
                            world_pos = coords.scene_to_world_point(
                                np.array([scene_x, 0, scene_z])
                            )
                            # Pass (world_forward, world_left) = (pos[0], pos[1])
                            listener(float(world_pos[0]), float(world_pos[1]))
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  navigate listener failed: {e}")
                elif mtype == "navigate_cancel":
                    # Navigation cancelled by browser. (COZMO-28)
                    listener = self._navigate_listener
                    if listener is not None:
                        try:
                            listener(None)
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  navigate cancel failed: {e}")
                elif mtype == "approach_cube":
                    # Cube click from browser — look up cube position and
                    # forward to listener. (COZMO-28)
                    listener = self._approach_listener
                    if listener is not None:
                        try:
                            cube_id_str = msg.get("cube_id", "")
                            # "cube_1" -> 0 (0-indexed for world map)
                            cube_idx = int(cube_id_str.replace("cube_", "")) - 1
                            world = self.world_map.get_world_state()
                            cubes = world.get("cubes", {})
                            if cube_idx in cubes:
                                pos = cubes[cube_idx]["position"]
                                listener(cube_idx, float(pos[0]), float(pos[1]))
                            else:
                                print(
                                    f"⚠️  approach_cube: {cube_id_str} not in world map"
                                )
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  approach listener failed: {e}")
                elif mtype == "dock":
                    # User pressed 'z' — drive robot back into dock. (COZMO-28)
                    listener = self._dock_listener
                    if listener is not None:
                        try:
                            listener()
                        except Exception as e:  # noqa: BLE001
                            print(f"⚠️  dock listener failed: {e}")
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
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  active-clients listener failed: {e}")

    async def _shutdown(self):
        for ws in list(self._clients):
            await ws.close()
        self._clients.clear()
        if self._run_forever is not None and not self._run_forever.done():
            self._run_forever.set_result(None)

    # -- static file serving (aiohttp) -------------------------------------

    async def _serve_index(self, request):
        """Serve index.html from the static directory."""
        viewer_dir = os.path.join(os.path.dirname(__file__), "static")
        return web.FileResponse(os.path.join(viewer_dir, "index.html"))

    async def _serve_static(self, request):
        """Serve static files from the static directory."""
        viewer_dir = os.path.join(os.path.dirname(__file__), "static")
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

    async def _serve_floor_video(self, request):
        """MJPEG streaming endpoint — pushes floor-projected frame to browser."""
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
                current_id = self._floor_image_id
                image = self._latest_floor_image
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
            app.router.add_get("/floor-video", self._serve_floor_video)
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
