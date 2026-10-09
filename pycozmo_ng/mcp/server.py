"""McPServer — streamable-HTTP MCP transport over the Toolset (COZMO-68).

Serves the MCP endpoint in-process on its own daemon thread running
uvicorn (the same thread-per-server pattern as ``web/ws_server.py`` —
no asyncio anywhere else in the robot path). The ``mcp`` SDK is an
optional extra (``pycozmo-ng[mcp]``); this module imports it lazily so
the rest of the package (and this module's own import) works without it.

Transport: streamable HTTP at ``http://<host>:<port>/mcp``. Binds to
all interfaces by default so LAN clients (and LLM hosts) can reach the
robot — the tool surface can move a physical robot, so restricting it
to localhost is the explicit choice of the caller when needed.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from pycozmo_ng.logger import logger_mcp
from pycozmo_ng.mcp.safety import SafetyLimits
from pycozmo_ng.mcp.tools import Toolset

if TYPE_CHECKING:
    from pycozmo_ng.cozmo import Cozmo

#: Human-facing guidance sent in the MCP ``initialize`` handshake.
INSTRUCTIONS = (
    "PyCozmo-NG drives a physical Cozmo robot. Frames and units: world frame, "
    "millimetres, headings in radians; cube IDs are 1-3. Read tools "
    "(get_robot_state, get_world_map, get_sensors, get_detections) return "
    "snapshots and are always safe. Motion tools (navigate_to, approach_cube, "
    "dock, dock_on_aruco) return a job_id immediately — poll get_job or "
    "wait_for_job for the outcome instead of assuming success. say, "
    "set_head_angle and set_lift_height are synchronous. stop cancels all "
    "motion and jobs: call it whenever motion looks wrong. Positions come "
    "from live localization: treat them as estimates, re-read state after "
    "motion, and never navigate outside the known map."
)


class McPServer:
    """One MCP endpoint bound to one :class:`~pycozmo_ng.mcp.tools.Toolset`.

    Args:
        cozmo: The public API instance backing the tools.
        host: Bind address (``0.0.0.0`` — all interfaces — unless
            explicitly restricted).
        port: TCP port.
        limits: Safety caps forwarded to the toolset.
    """

    def __init__(
        self,
        cozmo: Cozmo,
        host: str = "0.0.0.0",
        port: int = 8767,
        limits: SafetyLimits | None = None,
    ):
        self.host = host
        self.port = port
        self.toolset = Toolset(cozmo, limits=limits or SafetyLimits())
        self._mcp: Any = None
        self._server: Any = None  # uvicorn.Server
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        """Whether the endpoint is serving requests."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def url(self) -> str:
        """The streamable-HTTP endpoint URL."""
        return f"http://{self.host}:{self.port}/mcp"

    def start(self) -> None:
        """Build the MCP app and serve it on a daemon thread.

        Raises:
            RuntimeError: If the optional ``mcp`` package is missing, or
                the port is already in use.
        """
        if self.is_running:
            return
        try:
            import uvicorn
            from mcp.server.mcpserver import MCPServer as SdkServer
        except ModuleNotFoundError as exc:
            raise RuntimeError("MCP support requires the 'mcp' package: pip install 'pycozmo-ng[mcp]'") from exc

        from pycozmo_ng.web.ws_server import check_port_available

        if not check_port_available(self.host, self.port):
            raise RuntimeError(f"Port {self.port} is already in use on {self.host}")

        mcp = SdkServer(
            name="pycozmo-ng",
            instructions=INSTRUCTIONS,
            log_level="WARNING",
            warn_on_duplicate_tools=False,
        )
        self._register_tools(mcp)
        app = mcp.streamable_http_app(
            streamable_http_path="/mcp",
            stateless_http=True,  # one request, one response — no session resumption to leak
            host=self.host,
        )
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._mcp = mcp
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="mcp-server")
        self._thread.start()
        logger_mcp.info("MCP server listening on %s", self.url)

    def stop(self) -> None:
        """Stop serving and join the server thread (briefly)."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server = None
        self._mcp = None
        logger_mcp.info("MCP server stopped")

    def _register_tools(self, mcp) -> None:
        """Expose every ``Toolset`` tool by method name.

        Explicit list rather than ``dir()`` so the private helpers
        (``_refusal``, ``_run_navigate``, ...) can never leak into the
        tool surface by accident.
        """
        names = (
            # Reads
            "get_robot_state",
            "get_world_map",
            "get_sensors",
            "get_detections",
            # Motion (job-returning)
            "navigate_to",
            "approach_cube",
            "dock",
            "dock_on_aruco",
            "stop",
            # Actuation (synchronous)
            "say",
            "set_head_angle",
            "set_lift_height",
            "set_origin",
            # Configuration
            "camera_settings",
            "astar_toggle",
            # Jobs
            "get_job",
            "wait_for_job",
            "list_jobs",
        )
        for name in names:
            mcp.add_tool(getattr(self.toolset, name), name=name, structured_output=False)
