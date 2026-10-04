"""Static file serving for the Three.js web viewer.

Serves ``index.html`` and other assets from the ``static/`` directory
alongside this module.  Extracted from ``WebViewerServer`` for single
responsibility.
"""

import os

from aiohttp import web


class FileServer:
    """Serves static files from a directory.

    Parameters
    ----------
    static_dir:
        Absolute path to the directory containing the files.  Defaults
        to ``<this package>/static/``.
    """

    def __init__(self, static_dir: str | None = None):
        if static_dir is None:
            static_dir = os.path.join(os.path.dirname(__file__), "static")
        self._static_dir = static_dir

    async def serve_index(self, request):
        """Serve ``index.html``."""
        return web.FileResponse(os.path.join(self._static_dir, "index.html"))

    async def serve_static(self, request):
        """Serve a named file from the static directory."""
        filename = request.match_info["filename"]
        filepath = os.path.join(self._static_dir, filename)
        if os.path.isfile(filepath):
            return web.FileResponse(filepath)
        raise web.HTTPNotFound()
