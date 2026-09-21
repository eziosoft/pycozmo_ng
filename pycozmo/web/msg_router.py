"""WebSocket message routing for the Three.js web viewer.

Owns the dispatch logic that translates raw browser JSON messages into
typed listener callbacks.  Extracted from ``WebViewerServer._handler`` for
testability — ``MsgRouter.route`` can be called with any JSON string and
the listeners, no WebSocket required.
"""

import json

import numpy as np

from pycozmo import coords


class MsgRouter:
    """Dispatch incoming browser messages to registered listeners.

    Each ``set_*_listener`` method stores a single callback (or ``None``
    to unsubscribe).  ``route`` is the hot path — called per incoming
    WebSocket frame.
    """

    def __init__(self, world_map=None):
        self._world_map = world_map
        self._control_listener = None
        self._navigate_listener = None
        self._approach_listener = None
        self._dock_listener = None
        self._camera_settings_listener = None
        self._camera_settings: dict = {
            "head_light": False,
        }

    # -- listener registration -----------------------------------------------

    def set_control_listener(self, callback) -> None:
        self._control_listener = callback

    def set_navigate_listener(self, callback) -> None:
        self._navigate_listener = callback

    def set_approach_listener(self, callback) -> None:
        self._approach_listener = callback

    def set_dock_listener(self, callback) -> None:
        self._dock_listener = callback

    def set_camera_settings_listener(self, callback) -> None:
        self._camera_settings_listener = callback

    def set_camera_settings(self, settings: dict) -> None:
        self._camera_settings.update(settings)

    @property
    def camera_settings(self) -> dict:
        return self._camera_settings

    # -- dispatch ------------------------------------------------------------

    def route(self, raw: str) -> None:
        """Parse *raw* JSON and invoke the matching listener.

        Unknown or malformed messages are silently ignored.
        """
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        mtype = msg.get("type")
        handler = {
            "control": self._on_control,
            "navigate_to": self._on_navigate_to,
            "navigate_cancel": self._on_navigate_cancel,
            "approach_cube": self._on_approach_cube,
            "dock": self._on_dock,
            "camera_settings": self._on_camera_settings,
        }.get(mtype)
        if handler is not None:
            handler(msg)

    # -- handlers ------------------------------------------------------------

    def _on_control(self, msg: dict) -> None:
        listener = self._control_listener
        if listener is not None:
            keys = msg.get("keys") or []
            try:
                listener(set(keys))
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  control listener failed: {e}")

    def _on_navigate_to(self, msg: dict) -> None:
        listener = self._navigate_listener
        if listener is not None:
            try:
                scene_x = msg.get("x", 0)
                scene_z = msg.get("z", 0)
                world_pos = coords.scene_to_world_point(np.array([scene_x, 0, scene_z]))
                listener(float(world_pos[0]), float(world_pos[1]))
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  navigate listener failed: {e}")

    def _on_navigate_cancel(self, msg: dict) -> None:
        del msg
        listener = self._navigate_listener
        if listener is not None:
            try:
                listener(None)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  navigate cancel failed: {e}")

    def _on_approach_cube(self, msg: dict) -> None:
        listener = self._approach_listener
        if listener is None:
            return
        try:
            cube_id_str = msg.get("cube_id", "")
            cube_idx = int(cube_id_str.replace("cube_", "")) - 1
            if self._world_map is None:
                return
            world = self._world_map.get_world_state()
            cubes = world.get("cubes", {})
            if cube_idx in cubes:
                pos = cubes[cube_idx]["position"]
                listener(cube_idx, float(pos[0]), float(pos[1]))
            else:
                print(f"⚠️  approach_cube: {cube_id_str} not in world map")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  approach listener failed: {e}")

    def _on_dock(self, msg: dict) -> None:
        del msg
        listener = self._dock_listener
        if listener is not None:
            try:
                listener()
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  dock listener failed: {e}")

    def _on_camera_settings(self, msg: dict) -> None:
        listener = self._camera_settings_listener
        if listener is not None:
            try:
                settings = {
                    "auto_exposure_enabled": msg.get("auto_exposure_enabled", True),
                    "gain": float(msg.get("gain", 0.0)),
                    "exposure_ms": int(msg.get("exposure_ms", 100)),
                    "color": msg.get("color", False),
                    "resolution": msg.get("resolution", "QVGA"),
                    "jpeg_quality": int(msg.get("jpeg_quality", 70)),
                    "head_light": msg.get("head_light", False),
                }
                self._camera_settings.update(settings)
                listener(settings)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  camera_settings listener failed: {e}")
