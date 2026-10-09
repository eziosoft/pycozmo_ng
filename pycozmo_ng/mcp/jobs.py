"""JobRegistry — background job tracking for MCP-initiated motion (COZMO-68).

Long-running commands (navigate, approach, dock) are submitted here
instead of running inside the MCP tool call, so an LLM gets a
``job_id`` back immediately and never trips a client-side tool timeout.
Jobs run on daemon threads; results and status are read back with
:meth:`get` / :meth:`wait` (the ``get_job`` / ``wait_for_job`` MCP
tools).

The registry is transport-free and SDK-free — plain threads and dicts —
so it is unit-testable without the ``mcp`` package or a robot. It does
not cancel motion by itself; cancellation is the caller's business
(``Toolset.stop`` calls ``navigator.stop()``, which owns the motion
tokens), and :meth:`cancel` only flags the job as cancelled so waiting
callers stop waiting.

Statuses: ``running`` -> ``done`` | ``error``, plus a ``cancelled``
flag that stays independent of the terminal status (a cancelled job
that still finished its work reports ``done`` + ``cancelled: true``).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pycozmo_ng.logger import logger_mcp


@dataclass
class Job:
    """Snapshot of one submitted job. ``result``/``error`` are JSON-able."""

    id: str
    name: str
    status: str = "running"  # running | done | error
    result: Any = None
    error: str | None = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "name": self.name,
            "status": self.status,
            "cancelled": self.cancelled,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class JobRegistry:
    """Thread-safe registry of background jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._finished = threading.Event()  # wakes waiters when any job finishes
        self._done_callback: Callable[[Job], None] | None = None

    def set_done_callback(self, callback: Callable[[Job], None] | None) -> None:
        """Install a callback invoked (on the job thread) when a job finishes.

        Used by the MCP layer to emit completion notifications later; until
        then the default is no notification (Q4c: poll, don't push).
        """
        self._done_callback = callback

    def submit(self, name: str, fn: Callable[[], Any]) -> str:
        """Run ``fn()`` on a daemon thread under a fresh job id.

        Returns the job id immediately. Exceptions are captured as the
        job's ``error`` — they never escape to the MCP transport.
        """
        job = Job(id=uuid.uuid4().hex[:12], name=name)
        with self._lock:
            self._jobs[job.id] = job

        def _run() -> None:
            try:
                result = fn()
                with self._lock:
                    job.result = result
                    job.status = "done"
            except Exception as exc:  # noqa: BLE001 — job errors are data, not crashes
                logger_mcp.warning("MCP job %s (%s) failed: %s", job.id, name, exc)
                with self._lock:
                    job.error = str(exc)
                    job.status = "error"
            finally:
                with self._lock:
                    job.finished_at = time.time()
                self._finished.set()
                callback = self._done_callback
                if callback is not None:
                    try:
                        callback(job)
                    except Exception:  # noqa: BLE001 — a bad listener must not kill the job thread
                        logger_mcp.exception("MCP job done callback failed")

        threading.Thread(target=_run, daemon=True, name=f"mcp-job-{name}").start()
        logger_mcp.debug("MCP job %s (%s) submitted", job.id, name)
        return job.id

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return the job's current snapshot, or ``None`` for an unknown id."""
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def wait(self, job_id: str, timeout_s: float = 30.0) -> dict[str, Any] | None:
        """Block up to *timeout_s* for a job to finish; ``None`` for unknown ids.

        Returns the job snapshot either way — a still-running job simply
        has ``status: "running"`` (callers distinguish a timeout from a
        miss by that status, not by ``None``).
        """
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return None
                if job.status != "running":
                    return job.to_dict()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    job = self._jobs.get(job_id)
                    return job.to_dict() if job else None
            self._finished.wait(min(remaining, 0.2))
            self._finished.clear()

    def cancel(self, job_id: str) -> bool:
        """Flag a running job as cancelled. True when the id is known.

        Note: this does not stop robot motion — pair it with
        ``navigator.stop()`` (the ``stop`` tool does both).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancelled = True
            return True

    def list(self) -> list[dict[str, Any]]:
        """Snapshots of all known jobs, newest first."""
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [job.to_dict() for job in jobs]
