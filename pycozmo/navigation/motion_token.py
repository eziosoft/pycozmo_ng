"""MotionToken — cancellation handle for one motion job.

A token is created by
:class:`~pycozmo.navigation.motion_controller.MotionController` and is
polled by the job it belongs to. A token is *cancelled* when:

* the job (or an enclosing job) called :meth:`cancel`,
* :meth:`MotionController.cancel_all` ran after the token was created, or
* an enclosing job was cancelled (tokens nest — a job started inside
  another job is cancelled with it).

Entering the token on a thread makes it the thread's *current* token, so
any token begun inside that thread becomes its child. Leaving the token
removes it from the controller's registry, which is why every token must
be scoped either with ``with token:`` or by :meth:`MotionController.spawn`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable

    from pycozmo.navigation.motion_controller import MotionController


class MotionToken:
    """Cancellation handle for a single motion job."""

    def __init__(
        self,
        controller: MotionController,
        name: str,
        parent: MotionToken | None = None,
    ) -> None:
        self._controller = controller
        self.name = name
        self._parent = parent
        self._generation = controller.generation
        self._cancelled = False
        # Set by MotionController._push — the token this one replaced on
        # its thread, so scopes can nest.
        self._previous: MotionToken | None = None

    @property
    def cancelled(self) -> bool:
        """True once this job must stop.

        Cheap to poll: an integer comparison, a flag, and (if nested) the
        parent's answer.
        """
        if self._cancelled or self._generation != self._controller.generation:
            return True
        if self._parent is not None and self._parent.cancelled:
            self._cancelled = True
            return True
        return False

    def cancel(self) -> None:
        """Cancel this job and every job nested inside it."""
        self._cancelled = True

    # ------------------------------------------------------------------
    # Scope — registers/unregisters the token on the controller
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        self._controller._push(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._controller._pop(self)

    def spawn(self, target: Callable[[MotionToken], None]) -> MotionToken:
        """Run ``target(self)`` on a daemon thread with this token in scope.

        Returns the token immediately; poll ``token.cancelled`` from
        ``target`` to stop the job.
        """
        return self._controller._start(self, target)
