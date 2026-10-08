"""HTTP endpoint serving ArUco marker textures for the 3D viewer.

Renders a marker's actual 4x4 bitmap with ``cv2.aruco.generateImageMarker``
and serves it as PNG, cached per (dictionary, marker id, size) — the browser
fetches each texture once and keeps it for the session.

The dictionary comes from the same JSON config the detector uses, so the
rendered texture always matches what detection decodes.
"""

from __future__ import annotations

from aiohttp import web

from pycozmo_ng.logger import logger
from pycozmo_ng.vision.aruco_detector.aruco_config import DEFAULT_DICTIONARY, load_aruco_config

__all__ = ["ArucoTextureEndpoint"]


class ArucoTextureEndpoint:
    """Serves ``GET /aruco-texture?id=N`` as image/png.

    A bad id (outside the dictionary) yields 404. The dictionary comes from
    the ArUco config file, falling back to the default 4x4 dictionary when
    no file exists.
    """

    def __init__(self, dictionary: str | None = None, side_px: int = 256):
        """
        Args:
            dictionary: Dictionary name (e.g. ``"4X4_50"``). When None it is
                read from the ArUco config file (falling back to the default
                4x4 dictionary when no file exists).
            side_px: Rendered texture size (square, power-of-two friendly).
        """
        if dictionary is None:
            config = load_aruco_config()
            # No config file: detection is off, but still serve textures with
            # the default 4x4 dictionary (harmless; keeps one code path).
            dictionary = config.dictionary if config is not None else DEFAULT_DICTIONARY
        self._dictionary = dictionary
        self._side_px = side_px
        self._cache: dict[int, bytes] = {}

    async def handle(self, request: web.Request) -> web.Response:
        """Serve one marker texture: ``/aruco-texture?id=0``."""
        if self._dictionary is None:
            raise web.HTTPNotFound(text="ArUco config not loaded")

        try:
            marker_id = int(request.query.get("id", ""))
        except ValueError:
            raise web.HTTPBadRequest(text="id must be an integer") from None

        png = self._cache.get(marker_id)
        if png is None:
            from pycozmo_ng.vision.aruco_detector.aruco_detector import aruco_marker_png

            try:
                png = aruco_marker_png(marker_id, self._dictionary, self._side_px)
            except Exception as e:  # cv2.error for out-of-range ids, RuntimeError for missing aruco
                logger.debug(f"ArUco texture {marker_id} unavailable: {e}")
                raise web.HTTPNotFound(text=f"no marker {marker_id}") from e
            self._cache[marker_id] = png

        return web.Response(body=png, content_type="image/png", headers={"Cache-Control": "public, max-age=3600"})
