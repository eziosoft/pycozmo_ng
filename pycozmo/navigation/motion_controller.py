"""MotionController — the one place that knows which motions are running.

Every background motion job in the library (navigate, A* follow, dock,
cube approach, test mode, closed-loop drive) is registered here instead of
keeping its own cancellation flag. The interface is deliberately small:

* :meth:`begin` — open a job scope on the current thread (use as ``with``).
* :meth:`spawn` — run a job on a daemon thread.
* :meth:`cancel` — cancel the live jobs with a given name.
* :meth:`cancel_all` — cancel everything (``Navigator.stop``).
* :attr:`active_names` — which jobs are running right now.

Cancellation itself is a generation counter plus the token's parent chain, so
a token begun *after* ``cancel_all()`` starts un-cancelled: a stop() followed
by a new command re-arms movement without any explicit "clear the latch" step.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from pycozmo.navigation.motion_token import MotionToken


class MotionController:
    """Registry and cancellation authority for motion jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: list[MotionToken] = []
        self._generation = 0
        self._local = threading.local()

    @property
    def generation(self) -> int:
        """Monotonic counter, bumped by every :meth:`cancel_all`."""
        return self._generation

    @property
    def active_names(self) -> tuple[str, ...]:
        """Names of the jobs whose scope is currently open."""
        with self._lock:
            return tuple(token.name for token in self._jobs)

    # ------------------------------------------------------------------
    # Starting jobs
    # ------------------------------------------------------------------

    def begin(self, name: str) -> MotionToken:
        """Open a job scope on the current thread.

        The returned token must be scoped — either ``with controller.begin(...)``
        or ``controller.begin(...).spawn(target)`` — so it is removed from
        :attr:`active_names` when the job ends.
        """
        token = MotionToken(self, name, parent=self._current)
        with self._lock:
            self._jobs.append(token)
        return token

    def spawn(self, name: str, target: Callable[[MotionToken], None]) -> MotionToken:
        """Run ``target(token)`` on a daemon thread with ``name`` as its job."""
        return self.begin(name).spawn(target)

    def cancel(self, name: str) -> bool:
        """Cancel every live job named ``name``. True if anything was live."""
        with self._lock:
            jobs = [token for token in self._jobs if token.name == name]
        for token in jobs:
            token.cancel()
        return bool(jobs)

    def cancel_all(self) -> tuple[str, ...]:
        """Cancel every job, running or not. Returns the names cancelled.

        Tokens begun after this call belong to the new generation and are
        not cancelled — that is what makes a stopped robot re-arm on the
        next motion command.
        """
        with self._lock:
            self._generation += 1
            names = tuple(token.name for token in self._jobs)
        return names

    # ------------------------------------------------------------------
    # Implementation hooks (used by MotionToken)
    # ------------------------------------------------------------------

    def _start(
        self, token: MotionToken, target: Callable[[MotionToken], None]
    ) -> MotionToken:
        def _run() -> None:
            with token:
                target(token)

        thread = threading.Thread(target=_run, daemon=True, name=f"motion-{token.name}")
        thread.start()
        return token

    @property
    def _current(self) -> MotionToken | None:
        return getattr(self._local, "token", None)

    def _push(self, token: MotionToken) -> None:
        token._previous = self._current
        self._local.token = token

    def _pop(self, token: MotionToken) -> None:
        if self._current is token:
            self._local.token = token._previous
        with self._lock:
            if token in self._jobs:
                self._jobs.remove(token)
