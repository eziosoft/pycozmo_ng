"""CubeVisit — one cube visit at a time.

A *cube visit* is one trip to a cube: a grab approach
(:class:`~pycozmo_ng.navigation.cube_approach.CubeApproach`) or a stack
(:class:`~pycozmo_ng.navigation.cube_stack.CubeStack`). Both are
sequences of conflicting ``go_to_pose`` / ``drive_wheels`` commands, so
exactly one may be in flight.

This class owns that decision. It holds the lock, starts the visit on a
background motion job (so :meth:`Navigator.stop` cancels it), keeps the
lock until the job finishes, and answers truthfully whether the visit
actually started — that answer is what the public ``approach_cube`` /
``stack_cube`` booleans are built from.

``Navigator`` builds one instance and injects the *same* object into
both classes, so the two can never drift onto separate locks (which a
bare ``threading.Lock`` handed around per class allowed).
"""

import threading
from collections.abc import Callable
from typing import Any

from pycozmo_ng.navigation.motion_token import MotionToken

#: Starts a motion job — ``MotionController.spawn(name, target)``. The
#: returned token is not needed here: visits are cancelled by job name,
#: so it is typed loosely (``Any``) to keep test doubles simple.
Spawn = Callable[[str, Callable[[MotionToken], None]], Any]


class CubeVisit:
    """The lock, the job, and the truthful "did it start?" answer."""

    def __init__(self, spawn: Spawn):
        self._spawn = spawn
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """True while a cube visit is in flight."""
        return self._lock.locked()

    def begin(self, job_name: str, run: Callable[[MotionToken], None]) -> bool:
        """Start ``run`` as motion job ``job_name`` for one cube visit.

        Returns True when the visit started. Returns False when another
        cube visit is in flight — the request is ignored, never queued
        (the caller logs the request it dropped).

        The lock is held until ``run`` finishes and released in a
        ``finally``, so a raised visit cannot wedge the next one; if
        spawning itself fails, the lock is released before the error
        propagates.
        """
        if not self._lock.acquire(blocking=False):
            return False

        def _guarded(job: MotionToken) -> None:
            try:
                run(job)
            finally:
                self._lock.release()

        try:
            self._spawn(job_name, _guarded)
        except Exception:
            self._lock.release()
            raise
        return True
