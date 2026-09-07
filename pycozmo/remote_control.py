"""
Browser-side remote control for Cozmo.

Translates a set of held-keys (``w``/``a``/``s``/``d``/``r``/``f``/``t``/``g``/
`` ``) into wheel / head / lift commands on a :class:`pycozmo.Client`. Used
by the web viewer to drive Cozmo from the keyboard. (COZMO-25)

Key map:

    w / s        drive forward / reverse
    a / d        rotate left / right (in place)
    r / f        head up / down
    t / g        lift up / down
    space        stop all motors

The control loop is decoupled from WebSocket I/O: a producer (typically the
:class:`WebViewerServer` <pycozmo.web_viewer_server>) calls
:meth:`RemoteControl.update_keys` whenever a fresh key set arrives, and the
:class:`RemoteControl` runs a small background thread that polls that state
at ``tick_hz`` and sends ``drive_wheels`` / ``move_head`` / ``move_lift``
commands to Cozmo. On key release, or when no keys are held for a short
idle window, motors are stopped via ``stop_all_motors`` so a stuck or
forgotten key can't drive Cozmo off the desk.

The class is intentionally library-friendly: it doesn't depend on the web
viewer. Someone can ``import pycozmo.remote_control`` and feed it key
state from any source (keyboard daemon, network, gamepad bridge, etc.).
"""

import logging
import threading
import time
from typing import Iterable, Optional

from . import robot

logger = logging.getLogger(__name__)


_DEFAULT_WHEEL_SPEED = robot.MAX_WHEEL_SPEED.mmps * 0.5
_DEFAULT_HEAD_SPEED = 1.0  # rad/s for move_head
_DEFAULT_LIFT_SPEED = 1.0  # rad/s for move_lift
_DEFAULT_TICK_HZ = 20.0
_DEFAULT_IDLE_STOP_SEC = 0.15


class RemoteControl:
    """Drive Cozmo from a set of held-key names.

    Parameters
    ----------
    cli:
        A connected :class:`pycozmo.Client`. The control loop calls its
        ``drive_wheels``, ``move_head``, ``move_lift`` and ``stop_all_motors``
        methods.
    wheel_speed:
        Wheel speed in mm/s for forward / reverse. Left / right rotation
        uses this same magnitude.
    head_speed:
        ``move_head`` speed in rad/s.
    lift_speed:
        ``move_lift`` speed in rad/s.
    tick_hz:
        How often the control loop reads the latest key state and pushes
        commands to Cozmo. 20 Hz is plenty for human-driven motion and
        keeps the Wi-Fi load light.
    idle_stop_sec:
        If no fresh key update arrives within this many seconds, the loop
        assumes the controller went away (browser tab closed, network
        drop) and stops the motors. Belt-and-braces against Cozmo driving
        off the desk because a stale "w" was held in the last frame.
    """

    _VALID_KEYS = frozenset({"w", "a", "s", "d", "r", "f", "t", "g"})

    def __init__(
        self,
        cli,
        wheel_speed: float = _DEFAULT_WHEEL_SPEED,
        head_speed: float = _DEFAULT_HEAD_SPEED,
        lift_speed: float = _DEFAULT_LIFT_SPEED,
        tick_hz: float = _DEFAULT_TICK_HZ,
        idle_stop_sec: float = _DEFAULT_IDLE_STOP_SEC,
    ):
        self._cli = cli
        self._wheel_speed = float(wheel_speed)
        self._head_speed = float(head_speed)
        self._lift_speed = float(lift_speed)
        self._tick_hz = float(tick_hz)
        self._idle_stop_sec = float(idle_stop_sec)

        self._lock = threading.Lock()
        self._keys: set[str] = set()
        self._last_update: float = 0.0
        self._last_applied: set[str] = set()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- public API --------------------------------------------------------

    def update_keys(self, keys: Iterable[str]) -> None:
        """Replace the currently-held key set.

        Called by the WebSocket message handler whenever a fresh key set
        arrives from the browser. Keys are normalized to lowercase single
        characters; unknown keys are dropped silently.
        """
        normalized = {k.lower() for k in keys if isinstance(k, str)}
        normalized &= self._VALID_KEYS
        with self._lock:
            self._keys = normalized
            self._last_update = time.monotonic()

    def start(self) -> None:
        """Start the control loop in a background thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="RemoteControl", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the control loop and cut power to the motors."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            self._cli.stop_all_motors()
        except Exception as e:  # noqa: BLE001 — best-effort during shutdown
            logger.debug("stop_all_motors on shutdown failed: %s", e)

    # -- internals ---------------------------------------------------------

    def _snapshot(self) -> tuple[set[str], float]:
        with self._lock:
            return set(self._keys), self._last_update

    def _run(self) -> None:
        period = 1.0 / self._tick_hz
        while not self._stop_event.is_set():
            try:
                keys, last_update = self._snapshot()
                now = time.monotonic()
                stale = (now - last_update) > self._idle_stop_sec
                active_keys = set() if stale else keys

                self._apply_drive(active_keys)
                if active_keys != self._last_applied:
                    self._apply_head_lift(active_keys)
                    was = self._last_applied
                    self._last_applied = active_keys
                    if active_keys:
                        print(f"🎮 keys pressed: {active_keys}")
                    elif was:
                        print("🎮 keys released")
            except Exception as e:  # noqa: BLE001 — don't kill the loop
                logger.exception("RemoteControl tick failed: %s", e)
            self._stop_event.wait(period)

    def _apply_drive(self, keys: set[str]) -> None:
        forward = ("w" in keys) - ("s" in keys)
        rotate = ("a" in keys) - ("d" in keys)
        if forward or rotate:
            lw = forward * self._wheel_speed + rotate * self._wheel_speed
            rw = forward * self._wheel_speed - rotate * self._wheel_speed
            try:
                self._cli.drive_wheels(lwheel_speed=lw, rwheel_speed=rw)
            except Exception as e:  # noqa: BLE001
                logger.warning("drive_wheels(%s, %s) failed: %s", lw, rw, e)
        else:
            try:
                self._cli.drive_wheels(lwheel_speed=0, rwheel_speed=0)
            except Exception as e:  # noqa: BLE001
                logger.warning("drive_wheels(0, 0) failed: %s", e)

    def _apply_head_lift(self, keys: set[str]) -> None:
        head_up = "r" in keys
        head_down = "f" in keys
        lift_up = "t" in keys
        lift_down = "g" in keys

        if head_up and not head_down:
            self._safe_head(self._head_speed)
        elif head_down and not head_up:
            self._safe_head(-self._head_speed)
        else:
            self._safe_head(0.0)

        if lift_up and not lift_down:
            self._safe_lift(self._lift_speed)
        elif lift_down and not lift_up:
            self._safe_lift(-self._lift_speed)
        else:
            self._safe_lift(0.0)

    def _safe_head(self, speed: float) -> None:
        try:
            self._cli.move_head(speed)
        except Exception as e:  # noqa: BLE001
            logger.warning("move_head(%s) failed: %s", speed, e)

    def _safe_lift(self, speed: float) -> None:
        try:
            self._cli.move_lift(speed)
        except Exception as e:  # noqa: BLE001
            logger.warning("move_lift(%s) failed: %s", speed, e)
