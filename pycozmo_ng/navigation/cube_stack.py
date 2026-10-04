"""CubeStack — put the carried cube on top of a second cube.

Triggered by clicking another cube in the viewer while a cube rides the
forks (``Navigator.approach_cube_at`` routes to this class) or by
``Navigator.stack_cube``. The procedure mirrors a grab approach — same
two-step standoff, same alignment checks — but the forks stay *up* the
whole way (the carried cube would be dumped by the usual forks-down
prelude), and instead of scooping, the terminal action places:

1. creep forward so the carried cube sits directly above the target,
2. lower the fork to ``PLACE_LIFT_MM`` (72 mm, live-tuned) — the carried
   cube comes down to the stack while the prongs stay clear of the 44 mm
   cube below,
3. back away: the prongs slide out from under the carried cube (never
   touching the stack), which drops it the remaining gap onto the cube
   below,
4. drop the forks fully and forget the carry observation.

The settle wait after the release command is time-based:
``set_lift_height`` returns as soon as the command is sent (the firmware
acks before the motor starts), so the job waits out the travel.
"""

import math
import time

import pycozmo_ng
from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.cube_approach import CubeApproach
from pycozmo_ng.navigation.motion_token import MotionToken


class CubeStack:
    """Stack the cube on the forks onto a second, approached cube.

    Borrows the approach's drive/alignment sequence
    (:meth:`CubeApproach.drive_and_align` with ``lower_forks=False``) and
    its :attr:`CubeApproach.approach_lock`, so an approach and a stack
    can never interleave. Requires a carry observation that believes a
    cube is on the raised forks (``carry.on_forks``).

    Constructor args (dependency injection, same seams as
    :class:`CubeApproach`):
        cli             Robot client — pose, set_lift_height,
                        stop_all_motors.
        drive_to_point  Backing away from the stack (Navigator.drive_to_point).
        move_forward    Straight-ahead creep (Navigator._move_forward).
        spawn           Starts the stack as a background motion job.
        carry           Brain's CarryDetector — checked at entry,
                        cleared once the forks are fully down.
        approach        The shared CubeApproach instance (drive_and_align
                        + approach_lock).
    """

    #: Motion job name this class registers while a stack runs.
    JOB_STACK: str = "stack"

    # ------------------------------------------------------------------
    # Tuning constants
    # ------------------------------------------------------------------
    #: Live-tuned release height: bring the carried cube down onto the
    #: stack while keeping the prongs clear of the 44 mm cube below as
    #: the robot reverses (at half-up, 46 mm, the prongs wedged against
    #: it and dragged the stack on the way back). The carried cube drops
    #: the remaining gap as the prongs withdraw.
    PLACE_LIFT_MM: float = 72.0
    #: Gentle lowering so the placed cube is not bounced off.
    PLACE_LIFT_ACCEL = 1.5
    PLACE_LIFT_SPEED = 1.5
    #: ``set_lift_height`` returns once the command is sent (and the
    #: firmware acks before the motor starts) — wait a fixed, generous
    #: travel time before backing out rather than polling.
    PLACE_TRAVEL_S = 2.0
    #: How far to reverse so the prongs clear the stacked cube.
    STACK_BACK_AWAY_MM = 160.0
    #: Slow, straight reverse — no heading change (same yaw out as in).
    BACK_AWAY_SPEED_MMPS = 40.0
    BACK_AWAY_TIMEOUT_S = 10.0

    def __init__(self, cli, drive_to_point, move_forward, spawn, carry, approach):
        self.cli = cli
        self._drive_to_point = drive_to_point
        self._move_forward = move_forward
        self._spawn = spawn
        self.carry = carry
        self._approach = approach

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def stack(self, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Carry the held cube to the given cube and stack it there.

        Runs as one background ``stack`` motion job, so the caller
        (WebSocket handler) returns immediately and ``Navigator.stop``
        cancels the whole sequence. Refused when no cube is believed to
        be on the forks, when the target *is* the held cube, or when
        another cube visit (approach/stack) is already in flight.
        """
        if self.carry is None or not self.carry.on_forks:
            logger.warning("Stack: no cube on the forks - ignoring request")
            return
        carried_id = self.carry.cube_id
        if carried_id is not None and to_cube_id(cube_idx) == carried_id:
            logger.info("Stack: cube %d is already on the forks", carried_id)
            return

        lock = self._approach.approach_lock
        if not lock.acquire(blocking=False):
            logger.warning("Stack busy - ignoring request for cube %d", to_cube_id(cube_idx))
            return

        def _run_guarded(job: MotionToken) -> None:
            try:
                self._run(job, cube_idx, cube_x, cube_y)
            finally:
                lock.release()

        try:
            self._spawn(self.JOB_STACK, _run_guarded)
        except Exception:
            lock.release()
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self, job: MotionToken, cube_idx: int, cube_x: float, cube_y: float) -> None:
        """Blocking implementation — runs inside the ``stack`` job."""
        try:
            carried_id = self.carry.cube_id if self.carry is not None else None
            logger.info(
                "Stack: carrying cube %s onto cube %d",
                carried_id,
                to_cube_id(cube_idx),
            )
            aligned = self._approach.drive_and_align(job, cube_idx, cube_x, cube_y, lower_forks=False)
            if not aligned:
                if job.cancelled:
                    logger.info("Stack cancelled - cube %s stays on the forks", carried_id)
                    self.cli.stop_all_motors()
                else:
                    logger.warning("Stack aborted - cube %s stays on the forks", carried_id)
                return
            if job.cancelled:
                logger.info("Stack cancelled - not placing")
                self.cli.stop_all_motors()
                return
            self._place(job, cube_idx, carried_id)
        except Exception as e:  # noqa: BLE001
            logger.exception("Stack error: %s", e)

    def _place(self, job: MotionToken, cube_idx: int, carried_id: int | None) -> None:
        """Terminal action: creep over the target, release height, back out."""
        # 1. Creep in so the carried cube is directly above the target
        #    (same geometry the grab uses to straddle the cube).
        logger.info(
            "Stacking: +%.0fmm over cube %d, lowering to %.0fmm",
            CubeApproach.GRAB_FORWARD_MM,
            to_cube_id(cube_idx),
            self.PLACE_LIFT_MM,
        )
        self._move_forward(CubeApproach.GRAB_FORWARD_MM)
        if job.cancelled:
            logger.info("Stack cancelled - forks left up")
            self.cli.stop_all_motors()
            return

        # 2. Release height: the carried cube comes down to the stack,
        #    prongs still clear of the cube below.
        self.cli.set_lift_height(
            self.PLACE_LIFT_MM,
            accel=self.PLACE_LIFT_ACCEL,
            max_speed=self.PLACE_LIFT_SPEED,
        )
        if not self._sleep(job, self.PLACE_TRAVEL_S):
            logger.info("Stack cancelled - forks left at %.0fmm", self.PLACE_LIFT_MM)
            self.cli.stop_all_motors()
            return

        # 3. Reverse out from under the stack.
        if not self._back_away(job):
            if job.cancelled:
                logger.info("Stack cancelled while backing away")
                self.cli.stop_all_motors()
                return
            # Backing out failed: do NOT drop the forks — that would
            # push through the stack. Leave them at the release height.
            logger.warning(
                "Stack: back-away failed - forks left at %.0fmm over cube %d",
                self.PLACE_LIFT_MM,
                to_cube_id(cube_idx),
            )
            return

        # 4. Forks fully down — the stacked cube stays behind.
        self.cli.set_lift_height(
            pycozmo_ng.MIN_LIFT_HEIGHT.mm,
            accel=self.PLACE_LIFT_ACCEL,
            max_speed=self.PLACE_LIFT_SPEED,
        )
        if self.carry is not None:
            self.carry.clear()
        logger.info(
            "Stacked cube %s on cube %d",
            carried_id,
            to_cube_id(cube_idx),
        )

    def _back_away(self, job: MotionToken) -> bool:
        """Straight reverse along the current heading. True when clear."""
        rp = self.cli.pose
        yaw = rp.rotation.angle_z.radians
        tx = rp.position.x - math.cos(yaw) * self.STACK_BACK_AWAY_MM
        ty = rp.position.y - math.sin(yaw) * self.STACK_BACK_AWAY_MM
        logger.info("Stack: backing away %.0fmm", self.STACK_BACK_AWAY_MM)
        return bool(
            self._drive_to_point(
                tx,
                ty,
                timeout_s=self.BACK_AWAY_TIMEOUT_S,
                speed_mmps=self.BACK_AWAY_SPEED_MMPS,
                final_heading=yaw,
            )
        )

    @staticmethod
    def _sleep(job: MotionToken, seconds: float) -> bool:
        """Sleep in short slices; False when the job was cancelled."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if job.cancelled:
                return False
            time.sleep(0.05)
        return True
