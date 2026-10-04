"""WebSocket message routing for the Three.js web viewer.

Owns the dispatch logic that translates raw browser JSON messages into
typed listener callbacks.  Extracted from ``WebViewerServer._handler`` for
testability — ``MsgRouter.route`` can be called with any JSON string and
the listeners, no WebSocket required.
"""

import json
from collections.abc import Callable
from typing import ClassVar

import numpy as np

from pycozmo_ng import coords
from pycozmo_ng.cubes.identity import parse_cube_label
from pycozmo_ng.logger import logger


class MsgRouter:
    """Dispatch incoming browser messages to registered listeners.

    ``set_actions`` registers named listeners in one call — the single
    seam for browser → library commands. ``route`` is the hot path —
    called per incoming WebSocket frame; it parses, validates, converts
    scene→world coordinates, and swallows listener exceptions.
    """

    # Every browser action this router can dispatch, by name.
    ACTIONS: ClassVar[tuple[str, ...]] = (
        "control",
        "navigate",
        "approach",
        "dock",
        "camera_settings",
        "say",
        "test_mode",
        "astar_toggle",
    )

    def __init__(self, world_map=None):
        self._world_map = world_map
        self._actions: dict[str, Callable[..., None] | None] = {name: None for name in self.ACTIONS}

    # -- listener registration -----------------------------------------------

    def set_actions(self, **listeners: Callable[..., None] | None) -> None:
        """Register (or replace, or unsubscribe with ``None``) listeners.

        Keyword names must be :attr:`ACTIONS` members; unknown names raise
        so a wiring typo fails at registration, not silently at dispatch.
        """
        unknown = [name for name in listeners if name not in self._actions]
        if unknown:
            raise ValueError(f"unknown browser action(s): {', '.join(sorted(unknown))}")
        self._actions.update(listeners)

    def get_action(self, name: str) -> Callable[..., None] | None:
        """Return the listener registered for *name*, or None."""
        return self._actions[name]

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
            "say": self._on_say,
            "test_mode": self._on_test_mode,
            "astar_toggle": self._on_astar_toggle,
        }.get(mtype)
        if handler is not None:
            handler(msg)

    # -- handlers ------------------------------------------------------------

    def _on_control(self, msg: dict) -> None:
        listener = self._actions["control"]
        if listener is not None:
            keys = msg.get("keys") or []
            try:
                listener(set(keys))
            except Exception as e:  # noqa: BLE001
                logger.warning("control listener failed: %s", e)

    def _on_navigate_to(self, msg: dict) -> None:
        listener = self._actions["navigate"]
        if listener is not None:
            try:
                scene_x = msg.get("x", 0)
                scene_z = msg.get("z", 0)
                world_pos = coords.scene_to_world_point(np.array([scene_x, 0, scene_z]))
                listener(float(world_pos[0]), float(world_pos[1]))
            except Exception as e:  # noqa: BLE001
                logger.warning("navigate listener failed: %s", e)

    def _on_navigate_cancel(self, msg: dict) -> None:
        del msg
        listener = self._actions["navigate"]
        if listener is not None:
            try:
                listener(None)
            except Exception as e:  # noqa: BLE001
                logger.warning("navigate cancel failed: %s", e)

    def _on_approach_cube(self, msg: dict) -> None:
        listener = self._actions["approach"]
        if listener is None:
            return
        try:
            cube_index = parse_cube_label(msg.get("cube_id", ""))
            if self._world_map is None:
                return
            cube = self._world_map.get_cube(cube_index)
            if cube is not None:
                pos = cube.position
                listener(cube_index, float(pos[0]), float(pos[1]))
            else:
                logger.warning("approach_cube: %s not in world map", msg.get("cube_id", ""))
        except Exception as e:  # noqa: BLE001
            logger.warning("approach listener failed: %s", e)

    def _on_dock(self, msg: dict) -> None:
        del msg
        listener = self._actions["dock"]
        if listener is not None:
            try:
                listener()
            except Exception as e:  # noqa: BLE001
                logger.warning("dock listener failed: %s", e)

    def _on_camera_settings(self, msg: dict) -> None:
        listener = self._actions["camera_settings"]
        if listener is not None:
            try:
                # Field normalization and defaults live with the settings
                # owner (CameraService), not here — pass the message on.
                listener({key: value for key, value in msg.items() if key != "type"})
            except Exception as e:  # noqa: BLE001
                logger.warning("camera_settings listener failed: %s", e)

    def _on_say(self, msg: dict) -> None:
        listener = self._actions["say"]
        if listener is not None:
            try:
                text = msg.get("text", "")
                listener(text)
            except Exception as e:  # noqa: BLE001
                logger.warning("say listener failed: %s", e)

    def _on_test_mode(self, msg: dict) -> None:
        listener = self._actions["test_mode"]
        if listener is not None:
            try:
                enabled = msg.get("enabled", False)
                listener(enabled)
            except Exception as e:  # noqa: BLE001
                logger.warning("test_mode listener failed: %s", e)

    def _on_astar_toggle(self, msg: dict) -> None:
        listener = self._actions["astar_toggle"]
        if listener is not None:
            try:
                enabled = msg.get("enabled", False)
                listener(enabled)
            except Exception as e:  # noqa: BLE001
                logger.warning("astar_toggle listener failed: %s", e)
