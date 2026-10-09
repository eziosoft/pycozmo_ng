"""MCP adapter — let LLM clients drive Cozmo and read its state (COZMO-68).

An adapter package over the public :class:`~pycozmo_ng.Cozmo` API,
sibling to ``pycozmo_ng.web``: transport (``server``), tools
(``tools``), background jobs (``jobs``), and motion guards
(``safety``). The MCP SDK itself is an optional extra — only ``server``
imports it, lazily.

Start it from the public API::

    cozmo.mcp.start()  # http://<this-machine-ip>:8767/mcp
"""

from pycozmo_ng.mcp.jobs import Job, JobRegistry
from pycozmo_ng.mcp.safety import SafetyLimits
from pycozmo_ng.mcp.tools import Toolset

__all__ = [
    "Job",
    "JobRegistry",
    "SafetyLimits",
    "Toolset",
]
