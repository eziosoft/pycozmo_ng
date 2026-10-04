"""MotionController — the one place that knows which motions are running.

Every background motion job in the library (navigate, A* follow, dock,
cube approach, test mode, closed-loop drive) is registered here instead of
keeping its own cancellation flag. The interface is deliberately small:

* :meth:`begin` — open a job scope on the current thread (use as ``with``).
* :meth:`spawn` — run a job on a daemon thread.
* :meth:`cancel` — cancel the live jobs with a given name.
* :meth:`cancel_all` — cancel everything (``Navigator.stop``).
* :attr:`active_names` — which jobs are running right now.

It is also the single **driver lease**: exactly one *root* job (a job with
no enclosing job on its thread) commands the wheels at a time, and the
first thing a new root does in :meth:`begin` is take the lease from
whichever holder had it. Nested jobs ride their parent's lease — a drive
started by an approach never contends with the approach. Drivers that
aren't jobs (the browser ``RemoteControl``) take and release the lease
explicitly with :meth:`acquire_lease` / :meth:`release_lease`, so "who may
command the wheels right now" is answered here and nowhere else.

Cancellation itself is a generation counter plus the token's parent chain, so
a token begun *after* ``cancel_all()`` starts un-cancelled: a stop() followed
by a new command re-arms movement without any explicit "clear the latch" step.
``cancel_all()`` also drops the lease, so a stopped robot stays stopped until
the next driver asks for it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from pycozmo_ng.navigation.motion_token import MotionToken


class MotionController:
    """Registry and cancellation authority for motion jobs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: list[MotionToken] = []
        self._generation = 0
        self._lease: str | None = None
        self._local = threading.local()

    @property
    def generation(self) -> int:
        """Monotonic counter, bumped by every :meth:`cancel_all`."""
        return self._generation

    @property
    def lease_owner(self) -> str | None:
        """Who may command the wheels right now.

        A job name for a root motion job, ``"remote"`` for the browser
        remote control, or ``None`` when nobody holds it.
        """
        with self._lock:
            return self._lease

    def acquire_lease(self, owner: str) -> None:
        """Take the wheel lease for *owner*, cancelling the current holder.

        For non-job drivers (``RemoteControl``): root jobs take the lease
        automatically in :meth:`begin`. Taking a lease somebody else holds
        cancels their jobs — last requester wins, so fresh human input
        always preempts a running sequence.
        """
        with self._lock:
            prev = self._lease
            if prev == owner:
                return
            self._lease = owner
        if prev is not None:
            self.cancel(prev)

    def release_lease(self, owner: str) -> None:
        """Give up the wheel lease — a no-op unless *owner* holds it."""
        with self._lock:
            if self._lease == owner:
                self._lease = None

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

        A *root* token (no enclosing job on this thread) takes the wheel
        lease for ``name``, cancelling the previous holder: one top-level
        driver at a time. A nested token rides the parent's lease and
        never contends.
        """
        parent = self._current
        token = MotionToken(self, name, parent=parent)
        prev: str | None = None
        with self._lock:
            self._jobs.append(token)
            if parent is None and self._lease != name:
                prev = self._lease
                self._lease = name
        if prev is not None:
            self.cancel(prev)
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
        next motion command. The wheel lease is dropped with everything
        else: a driver that was holding it must ask again (a fresh
        RemoteControl key edge or a new root job) before commanding the
        wheels.
        """
        with self._lock:
            self._generation += 1
            self._lease = None
            names = tuple(token.name for token in self._jobs)
        return names

    # ------------------------------------------------------------------
    # Implementation hooks (used by MotionToken)
    # ------------------------------------------------------------------

    def _start(self, token: MotionToken, target: Callable[[MotionToken], None]) -> MotionToken:
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
            # A root job releases the lease when it ends — unless another
            # root of the same name is still live (navigate supersedes
            # itself by cancelling + re-spawning).
            if (
                token._parent is None
                and self._lease == token.name
                and not any(t.name == token.name and t._parent is None for t in self._jobs)
            ):
                self._lease = None
