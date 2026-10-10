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

import functools
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
    "(get_robot_state, get_world_map, get_sensors, get_detections, get_frame) return "
    "snapshots and are always safe; they carry camera_active/frame_age_s so you can "
    "tell a live frame from a stale one. Motion tools (navigate_to, move_by, turn_by, "
    "approach_cube, stack_cube, dock, dock_on_aruco) return a job_id immediately - "
    "poll get_job or wait_for_job for the outcome instead of assuming success. say, "
    "set_head_angle and set_lift_height are synchronous; wait(seconds) pauses up to "
    "10 s to let the robot settle instead of re-reading immediately. camera_toggle "
    "turns the video stream off/on - it is the recovery when get_frame returns null "
    "or get_detections is empty. stop cancels all motion and jobs: call it whenever "
    "motion looks wrong. Positions come from live localization: treat them as "
    "estimates, re-read state after motion, and never navigate outside the known map."
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
        tool surface by accident. Each tool also gets MCP annotations:
        reads are flagged read-only (clients may auto-approve them) and
        side-effect-free repeats are flagged idempotent.
        """
        from mcp_types import ToolAnnotations

        names = (
            # Reads (snapshots)
            "get_robot_state",
            "get_world_map",
            "get_sensors",
            "get_detections",
            "get_frame",
            # Motion (job-returning)
            "navigate_to",
            "move_by",
            "turn_by",
            "approach_cube",
            "stack_cube",
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
            "camera_toggle",
            "astar_toggle",
            # Jobs
            "get_job",
            "wait_for_job",
            "list_jobs",
            # Utilities
            "wait",
        )
        read_only = {
            "get_robot_state",
            "get_world_map",
            "get_sensors",
            "get_detections",
            "get_frame",
            "get_job",
            "wait_for_job",
            "list_jobs",
        }
        # Repeating these produces the same state; motion and say never do.
        idempotent = read_only | {"stop", "camera_settings", "camera_toggle", "astar_toggle"}
        for name in names:
            fn = getattr(self.toolset, name)
            if name == "get_frame":
                fn = self._get_frame_image_wrapper()
            mcp.add_tool(
                fn,
                name=name,
                structured_output=False,
                annotations=ToolAnnotations(
                    read_only_hint=name in read_only,
                    idempotent_hint=name in idempotent,
                ),
            )

    def _get_frame_image_wrapper(self):
        """Return a ``get_frame`` wrapper that emits real MCP image content.

        The Toolset stays free of the ``mcp`` SDK (its calls are JSON dicts by
        contract), so the base64 JPEG leaves the robot path as text. MCP clients
        (LLMs) can't see a base64 string as an image, so this wrapper — built
        where the SDK is already in scope — converts a populated frame into
        ``ImageContent`` plus a ``TextContent`` summary, returned as a
        ``CallToolResult``. A null return is passed through as the plain dict
        (the camera has no frame yet, there is nothing to draw).
        """
        from mcp_types import CallToolResult, ImageContent, TextContent

        frame_fn = self.toolset.get_frame

        @functools.wraps(frame_fn)
        def _wrapper(**kwargs: Any) -> Any:
            result = frame_fn(**kwargs)
            frame = result.get("frame")
            if not frame or not frame.get("jpeg_base64"):
                return result  # no frame available yet — text snapshot is right
            blocks: list[Any] = [ImageContent(type="image", data=frame["jpeg_base64"], mime_type="image/jpeg")]
            summary = frame.get("detections_summary")
            age = frame.get("age_s")
            meta = f"Frame {frame.get('width')}x{frame.get('height')}px, timestamp {frame.get('timestamp'):.3f}"
            if age is not None:
                meta += f", age {age:.1f}s"
            if summary:
                meta += f"\n{summary}"
            if frame.get("stale"):
                meta += (
                    f"\nSTALE: the newest frame is {age if age is not None else '?'}s old - the camera has "
                    "stopped producing. Call camera_toggle(enabled=true) to restart the stream."
                )
            blocks.append(TextContent(type="text", text=meta))
            return CallToolResult(content=blocks)

        _wrapper.__doc__ = (
            "Latest camera frame as MCP image content (JPEG), optimised for LLM consumption, "
            "plus a text block summarising what the vision pipeline sees in the frame.\n\n"
            "Args:\n"
            "    annotated: When true, return the annotated frame (detection boxes,\n"
            "        labels, confidence, distance drawn on the image) instead of the\n"
            "        raw camera frame.\n"
            "    max_width: Downscale the image so its width is at most this many\n"
            "        pixels (aspect ratio preserved). 0 = original resolution. Use\n"
            "        320 or 160 for smaller payloads.\n"
            "    jpeg_quality: JPEG encoding quality (1-95, default 70). Lower\n"
            "        values produce smaller payloads at the cost of fidelity.\n\n"
            "Returns an image content block plus a text summary of detections, "
            "including the frame's age - a stale marker means the camera stopped "
            "producing and camera_toggle(enabled=true) restarts it. When the camera "
            "is stopped or no frame has arrived yet, returns a text snapshot with "
            "frame: null."
        )
        return _wrapper
