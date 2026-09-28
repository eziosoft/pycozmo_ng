"""CameraService — the single owner of the Cozmo camera lifecycle.

One settings vocabulary, one defaults table, one packet sequence, and one
record of what is actually in force. Everything that used to touch the
camera from four different places — startup enable and calibration restore
(``Brain``), browser ``camera_settings`` messages (defaults were stated
twice, in ``MsgRouter`` and ``Brain``), manual exposure during robot-centre
calibration, and facade ``enable_camera`` (which bypassed the echo store
entirely) — now goes through this class, so the settings the web viewer
broadcasts are by construction what the robot was sent.

Covers COZMO-22 (camera lifecycle) and COZMO-30 (settings from browser).
"""

from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

from pycozmo.logger import logger
from pycozmo.protocol import protocol_encoder


class CameraService:
    """Enable, disable, and configure the camera; track what is in force.

    The interface is three commands — :meth:`enable`, :meth:`disable`,
    :meth:`apply_settings` — plus read-only state (``settings``,
    ``active``, ``enabled_at``, ``resolution``). Hiding behind them: the
    resolution name↔enum map, the settings vocabulary and its defaults,
    type coercion of browser messages, and the one packet sequence
    (``enable_camera`` → ``set_camera_params`` → ``set_head_light``).

    ``cli`` is injected by ``Brain`` once connected; commands raise
    ``RuntimeError`` before that.

    Typical use::

        camera.enable()                          # standard configuration
        camera.apply_settings({"gain": 1.0})     # partial update, keeps the rest
        camera.disable()
    """

    # The settings vocabulary — names, types, and defaults live here and
    # nowhere else. Keys double as the filter for incoming messages.
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "auto_exposure_enabled": True,
        "gain": 0.0,
        "exposure_ms": 20,
        "color": False,
        "resolution": "QVGA",
        "jpeg_quality": 70,
        "head_light": False,
    }

    # Resolution names (browser/JSON side) → protocol enum values.
    RESOLUTIONS: ClassVar[dict[str, protocol_encoder.ImageResolution]] = {
        "QQQQVGA": protocol_encoder.ImageResolution.QQQQVGA,
        "QQQVGA": protocol_encoder.ImageResolution.QQQVGA,
        "QQVGA": protocol_encoder.ImageResolution.QQVGA,
        "QVGA": protocol_encoder.ImageResolution.QVGA,
        "CVGA": protocol_encoder.ImageResolution.CVGA,
        "VGA": protocol_encoder.ImageResolution.VGA,
        "SVGA": protocol_encoder.ImageResolution.SVGA,
        "XGA": protocol_encoder.ImageResolution.XGA,
        "SXGA": protocol_encoder.ImageResolution.SXGA,
        "UXGA": protocol_encoder.ImageResolution.UXGA,
        "QXGA": protocol_encoder.ImageResolution.QXGA,
        "QUXGA": protocol_encoder.ImageResolution.QUXGA,
    }
    _RESOLUTION_NAMES: ClassVar[dict[protocol_encoder.ImageResolution, str]] = {
        enum: name for name, enum in RESOLUTIONS.items()
    }

    def __init__(self) -> None:
        self.cli: Any = None  # pycozmo Client, injected by Brain once connected
        self._lock = threading.Lock()
        self._settings: dict[str, Any] = dict(self.DEFAULTS)
        self._resolution: protocol_encoder.ImageResolution = self.RESOLUTIONS[
            self.DEFAULTS["resolution"]
        ]
        self._active = False
        self._enabled_at: float | None = None

    # -- state (read-only) ----------------------------------------------------

    @property
    def settings(self) -> dict[str, Any]:
        """Copy of the settings currently in force (the browser echo)."""
        with self._lock:
            return dict(self._settings)

    @property
    def active(self) -> bool:
        """Whether the camera stream was last enabled and not since stopped."""
        with self._lock:
            return self._active

    @property
    def enabled_at(self) -> float | None:
        """Timestamp of the last successful enable, or None before the first."""
        with self._lock:
            return self._enabled_at

    @property
    def resolution(self) -> protocol_encoder.ImageResolution:
        """Resolution the camera was last configured with."""
        with self._lock:
            return self._resolution

    # -- commands -------------------------------------------------------------

    def enable(self, resolution: Any = None, **overrides: Any) -> None:
        """Enable streaming with the standard configuration.

        Resets to :attr:`DEFAULTS` — the same sequence the startup and
        post-calibration paths always used — then layers *overrides* on
        top. *resolution* accepts an ``ImageResolution`` enum or a name.

        Args:
            resolution: ImageResolution enum or name (default: QVGA).
            **overrides: Extra settings (``gain``, ``exposure_ms``, …).
        """
        requested: dict[str, Any] = dict(overrides)
        if resolution is not None:
            requested["resolution"] = resolution
        self._send(self._normalize(requested, base=self.DEFAULTS))

    def disable(self) -> None:
        """Stop the camera stream. Resolution and settings are kept."""
        cli = self._require_cli()
        with self._lock:
            resolution = self._resolution
        cli.enable_camera(enable=False, color=False, resolution=resolution)
        with self._lock:
            self._active = False
        logger.info("Camera stopped")

    def apply_settings(self, settings: dict) -> dict:
        """Merge *settings* into what is in force, send them, return the result.

        Partial updates: omitted keys keep their current values. Unknown
        keys are dropped. This is the ``camera_settings`` browser action —
        the returned dict is what the web viewer broadcasts, so the echo
        is the applied state by construction.
        """
        with self._lock:
            base = dict(self._settings)
        self._send(self._normalize(settings, base=base))
        return self.settings

    # -- implementation -------------------------------------------------------

    def _normalize(self, settings: dict, base: dict) -> dict[str, Any]:
        """Filter *settings* to known keys, merge over *base*, coerce types."""
        known = {key: value for key, value in settings.items() if key in self.DEFAULTS}
        unknown = set(settings) - set(self.DEFAULTS)
        if unknown:
            logger.debug(
                "Ignoring unknown camera setting(s): %s", ", ".join(sorted(unknown))
            )
        merged = {**self.DEFAULTS, **base, **known}
        merged["auto_exposure_enabled"] = bool(merged["auto_exposure_enabled"])
        merged["gain"] = float(merged["gain"])
        merged["exposure_ms"] = int(merged["exposure_ms"])
        merged["color"] = bool(merged["color"])
        merged["head_light"] = bool(merged["head_light"])
        merged["jpeg_quality"] = int(merged["jpeg_quality"])
        merged["resolution"] = self._resolution_name(merged["resolution"])
        return merged

    def _send(self, settings: dict[str, Any]) -> None:
        """Run the one packet sequence and record it as the state in force."""
        cli = self._require_cli()
        resolution = self.RESOLUTIONS[settings["resolution"]]
        cli.enable_camera(
            enable=True,
            color=settings["color"],
            resolution=resolution,
        )
        cli.set_camera_params(
            gain=settings["gain"],
            exposure_ms=settings["exposure_ms"],
            auto_exposure_enabled=settings["auto_exposure_enabled"],
        )
        cli.set_head_light(settings["head_light"])
        with self._lock:
            self._settings = dict(settings)
            self._resolution = resolution
            self._active = True
            self._enabled_at = time.time()
        logger.info(
            "Camera configured: %s, %s, gain=%s, exposure=%sms, auto_exposure=%s, "
            "head_light=%s",
            settings["resolution"],
            "color" if settings["color"] else "grayscale",
            settings["gain"],
            settings["exposure_ms"],
            "on" if settings["auto_exposure_enabled"] else "off",
            "on" if settings["head_light"] else "off",
        )

    def _resolution_name(self, value: Any) -> str:
        """Resolve an enum or name to a known resolution name (QVGA fallback)."""
        default = self.DEFAULTS["resolution"]
        if isinstance(value, str):
            if value in self.RESOLUTIONS:
                return value
        else:
            name = self._RESOLUTION_NAMES.get(value)
            if name is not None:
                return name
        logger.warning("Unknown camera resolution %r - using %s", value, default)
        return default

    def _require_cli(self):
        if self.cli is None:
            raise RuntimeError("Not connected - no client injected into CameraService")
        return self.cli
