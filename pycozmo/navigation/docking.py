"""Docking — drive back to the dock pose and re-anchor the pose origin.

Extracted from ``Navigator.dock()``, which was a 123-line method with four
nested closures and eight method-local constants — impossible to build or
test without a full ``Navigator``.  Same dependency-injection shape as
:class:`~pycozmo.navigation.cube_approach.CubeApproach`: ``cli`` is
injected at wiring time (``Navigator.cli`` propagates it, including on
reconnect), while the dock state, the shared
:class:`~pycozmo.navigation.motion_controller.MotionController`, and the
origin-reset callback come from the constructor.

The origin reset stays behind a callback because re-anchoring the pose
origin is charger/origin logic owned by ``Navigator``; this class only
says "we are docked — re-anchor now".  Registering the job on the
*shared* motion controller is what keeps ``Navigator.stop()`` able to
cancel docking in one call.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any, ClassVar

from pycozmo.logger import logger
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.motion_controller import MotionController
from pycozmo.navigation.motion_token import MotionToken
from pycozmo.util import Angle, Pose


class Docking:
    """Dock sequence: drive in front of the dock, reverse into contact, retry.

    :meth:`dock` is the whole interface — it validates, supersedes any
    running dock job, and spawns this one.  Everything the sequence needs
    (tuning, contact/stall detection, retries, charging checks) is hidden
    behind it and can run against a fake ``cli``.
    """

    #: Motion job name registered on the shared controller while docking.
    #: Job names live on the class that owns the work (was ``Navigator.JOB_DOCK``).
    JOB_DOCK: ClassVar[str] = "dock"

    # -- tuning (were method-local constants inside Navigator.dock) ----------
    DRIVE_DIST_MM: ClassVar[float] = 150.0
    REVERSE_SPEED_MMPS: ClassVar[float] = 80.0
    AIM_BEHIND_MM: ClassVar[float] = 40.0
    CONTACT_DIST_MM: ClassVar[float] = 15.0
    MAX_RETRIES: ClassVar[int] = 3
    RETRY_SPEED_MMPS: ClassVar[float] = 60.0
    RETRY_DRIVE_MM: ClassVar[float] = 40.0
    STALL_ITERS: ClassVar[int] = 20

    def __init__(
        self,
        dock_state: DockState,
        motion: MotionController,
        reset_origin: Callable[[], None],
    ) -> None:
        self.ds = dock_state
        self.motion = motion
        self._reset_origin = reset_origin
        self.cli: Any = None  # robot client, injected by Navigator

    # ------------------------------------------------------------------
    # Docking (COZMO-28)
    # ------------------------------------------------------------------

    def dock(self) -> None:
        """Drive the robot back into the dock in the background.

        One dock job at a time: a new request supersedes the old one.
        """
        if not self.cli:
            logger.warning("Dock: no robot connected")
            return
        dock_pose = self.ds.dock_pose
        if dock_pose is None:
            logger.warning("Dock: no start pose recorded")
            return
        # Capture the pose now, so a later origin reset cannot retarget a
        # dock job that is already running.
        self.motion.cancel(self.JOB_DOCK)
        self.motion.spawn(self.JOB_DOCK, lambda job: self._dock_job(job, dock_pose))

    # ------------------------------------------------------------------
    # Job body
    # ------------------------------------------------------------------

    def _dock_job(self, job: MotionToken, dock_pose) -> None:
        cli = self.cli
        if cli is None:
            return

        dock_x = dock_pose.position.x
        dock_y = dock_pose.position.y
        dock_heading = dock_pose.rotation.angle_z.radians

        turn_x = dock_x + self.DRIVE_DIST_MM * math.cos(dock_heading)
        turn_y = dock_y + self.DRIVE_DIST_MM * math.sin(dock_heading)

        aim_x = dock_x - self.AIM_BEHIND_MM * math.cos(dock_heading)
        aim_y = dock_y - self.AIM_BEHIND_MM * math.sin(dock_heading)

        try:
            turn_pose = Pose(turn_x, turn_y, 0, angle_z=Angle(radians=dock_heading))
            logger.info("Driving to %.0fmm in front of dock", self.DRIVE_DIST_MM)
            cli.go_to_pose(turn_pose, relative_to_robot=False)
            time.sleep(0.3)

            docked = False
            for attempt in range(1, self.MAX_RETRIES + 1):
                if job.cancelled:
                    logger.info("Docking cancelled")
                    break
                logger.info("Docking attempt %d/%d", attempt, self.MAX_RETRIES)
                if self._reverse_until_contact(job, aim_x, aim_y):
                    docked = True
                    break

                if self._is_charging():
                    docked = True
                    break

                if attempt < self.MAX_RETRIES:
                    logger.warning("No charge at contact - backing away to retry")
                    cli.drive_wheels(
                        lwheel_speed=self.RETRY_SPEED_MMPS,
                        rwheel_speed=self.RETRY_SPEED_MMPS,
                        duration=self.RETRY_DRIVE_MM / self.RETRY_SPEED_MMPS,
                    )
                    time.sleep(0.3)
                else:
                    logger.warning(
                        "Docking failed after %d attempts (dist=%.0fmm, no charge)",
                        self.MAX_RETRIES,
                        self._dist_to_dock(aim_x, aim_y),
                    )

            cli.stop_all_motors()
            if job.cancelled:
                return

            if docked or self._is_charging():
                self._reset_origin()
                logger.info("Docked and pose reset - dock is now origin")
            elif not docked:
                logger.warning("Docking did not complete - no pose reset")
        except Exception as e:  # noqa: BLE001
            logger.exception("Dock error: %s", e)

    def _reverse_until_contact(
        self, job: MotionToken, aim_x: float, aim_y: float
    ) -> bool:
        """Reverse toward the aim point until charging starts or progress stalls.

        Returns True when the charger reports contact, False on cancel,
        on contact-distance reached without charge, or when the distance
        stops decreasing (:attr:`STALL_ITERS` ticks).
        """
        prev_dist = None
        stall = 0
        while True:
            if job.cancelled:
                return False
            if self._is_charging():
                return True

            dist = self._dist_to_dock(aim_x, aim_y)

            if dist < self.CONTACT_DIST_MM:
                return False

            if prev_dist is not None and dist >= prev_dist - 1.0:
                stall += 1
                if stall >= self.STALL_ITERS:
                    return False
            else:
                stall = 0
            prev_dist = dist

            self.cli.drive_wheels(
                lwheel_speed=-self.REVERSE_SPEED_MMPS,
                rwheel_speed=-self.REVERSE_SPEED_MMPS,
            )
            time.sleep(0.05)

    def _is_charging(self) -> bool:
        with self.ds.lock:
            return self.ds.on_charger or self.ds.charging

    def _dist_to_dock(self, aim_x: float, aim_y: float) -> float:
        rp = self.cli.pose
        dx = aim_x - rp.position.x
        dy = aim_y - rp.position.y
        return math.sqrt(dx * dx + dy * dy)
