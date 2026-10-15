"""Navigation — navigate, approach cube, dock, reset origin, charger handlers.

The A* follow loop lives in :class:`~pycozmo_ng.navigation.astar_follower.AStarFollower`
and the dock sequence in :class:`~pycozmo_ng.navigation.docking.Docking`; this
class keeps the entry points (and the shared motion controller) so callers
still address navigation as one thing.
"""

import math
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

import pycozmo_ng
from pycozmo_ng.cubes.carry_state import CarryDetector
from pycozmo_ng.cubes.identity import to_cube_id, to_world_index
from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.aruco_docking import ArucoDocking
from pycozmo_ng.navigation.astar import AStarPlanner
from pycozmo_ng.navigation.astar_follower import AStarFollower
from pycozmo_ng.navigation.cube_approach import CubeApproach
from pycozmo_ng.navigation.cube_stack import CubeStack
from pycozmo_ng.navigation.cube_visit import CubeVisit
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.docking import Docking
from pycozmo_ng.navigation.motion_controller import MotionController
from pycozmo_ng.navigation.motion_token import MotionToken
from pycozmo_ng.navigation.steering import steer_velocities, wrap_pi
from pycozmo_ng.util import Angle, Pose

if TYPE_CHECKING:
    from pycozmo_ng.navigation.world_map import WorldMap


def _pose_signature(pose) -> tuple[float, float, float]:
    """``(x, y, yaw)`` of a firmware pose — the movement fingerprint."""
    return (pose.position.x, pose.position.y, pose.rotation.angle_z.radians)


def _pose_moved(
    anchor: tuple[float, float, float],
    current: tuple[float, float, float],
    eps_mm: float = 0.5,
    eps_rad: float = math.radians(0.3),
) -> bool:
    """Whether the pose changed beyond localization noise since *anchor*."""
    return (
        math.hypot(current[0] - anchor[0], current[1] - anchor[1]) > eps_mm
        or abs(wrap_pi(current[2] - anchor[2])) > eps_rad
    )


class Navigator:
    """Robot navigation: point-to-point drive, cube approach, docking.

    Uses :class:`DockState` for charger/dock data. Receives the injected
    ``world_map`` at construction and ``cli`` at wiring time (a property
    that propagates the client to the A* follower, docking, and cube
    approach), rather than holding a Brain or web-viewer reference.

    All motion runs as *jobs* registered on :class:`MotionController`
    (:attr:`motion`), which owns cancellation: :meth:`stop` cancels every
    job in one call, and a job begun after that is not cancelled, so the
    next motion command re-arms the robot.
    """

    #: Job names registered on ``self.motion`` (see :attr:`active_names`).
    #: Docking's job name lives on :attr:`Docking.JOB_DOCK`.
    JOB_NAVIGATE: ClassVar[str] = "navigate"
    JOB_DRIVE: ClassVar[str] = "drive"
    JOB_ALIGN: ClassVar[str] = "align"
    JOB_TEST_MODE: ClassVar[str] = "test_mode"

    #: When the robot is on the charger but its distance from the world
    #: origin exceeds this tolerance (mm) the origin must be re-anchored
    #: even if the origin id hasn't changed -- the robot drove away and
    #: came back without a lift/pick-up cycle to bump the origin id.
    #: (COZMO-72 follow-up)
    ORIGIN_AT_ORIGIN_TOLERANCE_MM: ClassVar[float] = 10.0

    def __init__(
        self,
        dock_state: DockState,
        world_map: "WorldMap | None" = None,
        motion: MotionController | None = None,
        carry: "CarryDetector | None" = None,
    ):
        self.ds = dock_state
        self.world_map = world_map
        # Carry state — created by Brain, read through the public API;
        # CubeApproach arms/clears it around a grab.
        self.carry = carry
        self._cli: Any = None
        # The shared motion seam (cancellation + driver lease). Brain
        # creates one controller and hands it to Navigator and
        # RemoteControl, so navigation jobs and keyboard input contend on
        # the same authority; a standalone Navigator gets its own.
        self.motion = motion if motion is not None else MotionController()
        # A* path following (planner + enabled flag + follow loop)
        self.astar = AStarFollower(world_map=world_map)
        # Dock sequence — shares the motion controller so stop() reaches it
        self.docking = Docking(
            dock_state=dock_state,
            motion=self.motion,
            reset_origin=self._reset_pose_origin,
        )
        # Cube approach (lazily created with DI callbacks)
        self._cube_approach: CubeApproach | None = None
        # Cube stacking — created on demand, shares the approach's
        # drive sequence.
        self._cube_stack: CubeStack | None = None
        # The one cube-visit authority: both classes take this same
        # object, so a grab and a stack can never interleave (and cannot
        # be wired onto separate locks by accident).
        self._cube_visit = CubeVisit(self.motion.spawn)
        # Serialises origin resets so the two callers that fire together
        # on a successful dock — the on_charger/on_charging event threads
        # and _on_aruco_docked — cannot each send a SetOrigin packet.
        self._origin_reset_lock = threading.Lock()

    @property
    def cli(self) -> Any:
        return self._cli

    @cli.setter
    def cli(self, value: Any) -> None:
        """Inject the client — and keep every sub-sequence in step with it."""
        self._cli = value
        self.astar.cli = value
        self.docking.cli = value
        if self._cube_approach is not None:
            self._cube_approach.cli = value
        if self._cube_stack is not None:
            self._cube_stack.cli = value

    @property
    def cube_visit(self) -> CubeVisit:
        """The one cube-visit authority (see :class:`CubeVisit`).

        Exposed so an observer can see which cube is being grabbed or
        stacked right now — the cube indicator reads ``active_cube_id``.
        """
        return self._cube_visit

    # ------------------------------------------------------------------
    # Charger / origin handlers (COZMO-28)
    # ------------------------------------------------------------------

    def on_charger_change(self, cli, state: bool) -> None:
        """Record whether the robot is physically on the charger."""
        del cli
        with self.ds.lock:
            self.ds.on_charger = state
        if state:
            logger.info("On charger detected")
            threading.Thread(target=self._anchor_on_charger, daemon=True).start()

    def on_charging_change(self, cli, state: bool) -> None:
        """Record whether the robot is actively charging."""
        del cli
        with self.ds.lock:
            self.ds.charging = state
        if state:
            logger.info("Charging detected")
            threading.Thread(target=self._anchor_on_charger, daemon=True).start()

    def _anchor_on_charger(self) -> None:
        """Origin reset + marker 0 snap — the charger-event dock anchor.

        The robot sits on the charger, so after the origin reset its pose is
        the world origin and marker 0's pose follows from geometry: 65 mm
        behind the camera (:meth:`_snap_marker_dock_pose`). The origin change
        clears the world map — marker 0 included — so the marker must be
        hard-set back right after the reset confirms, exactly like
        :meth:`_on_aruco_docked` does after an ArUco dock.

        ``on_charger_change`` and ``on_charging_change`` both land here on a
        successful dock; ``_origin_reset_lock`` serialises them with each
        other and with the ArUco path (the loser hits the reset guard and
        returns early — the snap is an idempotent overwrite of the same pose).
        """
        self._anchor_dock(self.CHARGER_MARKER_ID, "Charger event")

    def _anchor_dock(self, marker_id: int, source: str) -> None:
        """Reset the origin, then snap *marker_id* to its docked pose.

        The snap is skipped when the reset was not telemetry-confirmed: the
        robot's world pose would not be (0, 0, 0), so the marker would be
        hard-set at a wrong spot (COZMO-72 follow-up).

        Args:
            marker_id: Marker to snap after the reset.
            source: Calling path, for the not-confirmed warning.
        """
        if not self._reset_pose_origin():
            logger.warning("%s: origin reset not confirmed — skipping marker %d snap", source, marker_id)
            return
        self._snap_marker_dock_pose(marker_id)

    def _reset_pose_origin(self) -> bool:
        """Re-anchor the localization origin at the current pose.

        Sends SetOrigin so the robot's current pose becomes the new
        world frame (0,0,0) and records it as the dock pose. (COZMO-28)

        ``_origin_reset_lock`` serialises the send + confirmation so that
        the two callers that fire together on a successful dock — the
        ``on_charging_change``/``on_charger_change`` event threads and
        :meth:`_on_aruco_docked` — cannot each send a SetOrigin packet
        and double-anchor the origin.

        After sending, polls for the origin id to echo back in telemetry
        rather than sleeping a fixed 0.5 s. The world map is cleared on
        the origin change (by ``update_robot_pose`` in
        ``Brain._on_robot_state``), and :meth:`_snap_marker_dock_pose`
        runs immediately after this method returns — it must land on the
        *cleared* map, not be wiped by a delayed origin-change clear
        (COZMO-72).

        Returns:
            True when the new origin id was confirmed in telemetry (the
            origin was actually re-anchored and the world map cleared).
            False when the poll timed out — the caller should treat the
            reset as inconclusive (it may have landed or may have been
            lost on the wire) and **must not** set the DockState guard
            origin id, so that the next charger event retries the send
            instead of silently skipping it (COZMO-72 follow-up).
        """
        cli = self.cli
        if cli is None:
            return False
        with self._origin_reset_lock:
            with self.ds.lock:
                if self.ds.origin_reset_origin_id == cli.pose.origin_id:
                    # Origin id hasn't changed since the last reset.  But the
                    # robot may have driven *away* from the charger and come
                    # back without a pick-up cycle (which is what bumps the
                    # origin id via _on_robot_picked_up).  In that case the
                    # position is no longer (0, 0) even though the id is the
                    # same -- the origin must be re-anchored.  Only skip when
                    # the robot is genuinely sitting at the origin.
                    dist_from_origin = math.hypot(
                        cli.pose.position.x,
                        cli.pose.position.y,
                    )
                    if dist_from_origin < self.ORIGIN_AT_ORIGIN_TOLERANCE_MM:
                        return True  # already at origin for this origin id
                    # Fall through: robot on charger but not at (0, 0) — reset.

                # Snapshot scalars in the *old* frame: after SetOrigin every
                # pose read is the new one, so this is the only place the
                # pre-reset position (where the robot physically stands)
                # can be logged from. Values, not the object — the client
                # replaces ``pose`` per tick.
                ox, oy, oyaw = (
                    cli.pose.position.x,
                    cli.pose.position.y,
                    cli.pose.rotation.angle_z.radians,
                )
                new_origin_id = cli.pose.origin_id + 1
                reset_pkt = pycozmo_ng.protocol_encoder.SetOrigin(
                    pose_frame_id=cli.pose_frame_id + 1,
                    pose_origin_id=new_origin_id,
                )
            cli.conn.send(reset_pkt)

            # Wait for the origin change to echo back in telemetry. The
            # robot-state handler updates cli.pose atomically (new origin_id,
            # position reset to 0,0,0) before dispatching EvtRobotStateUpdated,
            # whose Brain handler then clears the world map via
            # update_robot_pose. Polling the origin_id guarantees the map
            # clear has happened before _snap_marker_dock_pose runs —
            # otherwise the marker it sets is immediately discarded.
            deadline = time.monotonic() + 2.0
            origin_confirmed = False
            while time.monotonic() < deadline:
                if cli.pose.origin_id == new_origin_id:
                    origin_confirmed = True
                    break
                time.sleep(0.05)

            with self.ds.lock:
                nx, ny, nyaw = (
                    cli.pose.position.x,
                    cli.pose.position.y,
                    cli.pose.rotation.angle_z.radians,
                )
                # Only record the confirmed origin id as "reset" so the
                # guard doesn't poison itself on a telemetry timeout:
                # if the echo never arrived (packet loss on Cozmo's Wi-Fi),
                # cli.pose.origin_id is still the *old* id — recording it
                # would make the guard skip every future charger reset
                # permanently (COZMO-72 follow-up).
                if origin_confirmed:
                    self.ds.origin_reset_origin_id = cli.pose.origin_id
                self.ds.dock_pose = cli.pose

        if origin_confirmed:
            logger.info(
                "Pose origin reset — robot was (%.0f, %.0f, %.1f deg) in the old frame, now (%.0f, %.0f, %.1f deg)",
                ox,
                oy,
                math.degrees(oyaw),
                nx,
                ny,
                math.degrees(nyaw),
            )
        else:
            logger.warning(
                "Pose origin reset timed out after %.1fs — origin id did not "
                "echo (sent id %d, telemetry still at %d). The SetOrigin "
                "packet may have been lost on the wire; the next charger "
                "event will retry the send.",
                time.monotonic() - (deadline - 2.0),
                new_origin_id,
                cli.pose.origin_id,
            )
        return origin_confirmed

    # ------------------------------------------------------------------
    # Click-to-navigate (COZMO-28 / COZMO-30)
    # ------------------------------------------------------------------

    def navigate_to(self, world_forward: float | None, world_left: float | None = None) -> None:
        """Navigate to a world position — or cancel the active navigation.

        ``world_forward`` = X (Anki frame), ``world_left`` = Y (Anki frame).
        Passing ``world_forward=None`` cancels navigation and stops the
        motors (the browser's navigate-cancel sends a single ``None``
        argument).

        When A* is enabled and a planner is set, computes an obstacle-
        avoiding path through the occupancy grid and follows it with the
        robot's built-in path follower. Otherwise drives straight to the
        target via ``go_to_pose`` (also firmware-side).

        Both variants run as one live ``navigate`` job on :attr:`motion`;
        cancel stops that job (and every job nested inside it) without
        touching unrelated motions.
        """
        if world_forward is None:
            self.motion.cancel(self.JOB_NAVIGATE)
            if self.cli:
                self.cli.stop_all_motors()
            if self.world_map is not None:
                self.world_map.clear_astar_path()
            logger.info("Navigation cancelled")
            return

        if world_left is None:
            logger.warning("Navigation: no target Y coordinate")
            return

        if not self.cli:
            logger.warning("Navigation: no robot connected")
            return

        robot_pose = self.cli.pose
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y

        dx = world_forward - robot_x
        dy = world_left - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        logger.info("Navigating to (%.0f, %.0f) - dist=%.0fmm", world_forward, world_left, dist)

        def _navigate(job: MotionToken) -> None:
            try:
                if self.astar.enabled and self.astar.planner is not None:
                    self.astar.follow(world_forward, world_left, job)
                elif dist < 10.0:
                    logger.info("Already at target")
                else:
                    heading = math.atan2(dy, dx)
                    target_pose = Pose(
                        world_forward,
                        world_left,
                        0,
                        angle_z=Angle(radians=heading),
                    )
                    ok = self.cli.go_to_pose(
                        target_pose,
                        relative_to_robot=False,
                        should_interrupt=lambda: job.cancelled,
                    )
                    if job.cancelled:
                        logger.info("Navigation cancelled")
                    elif ok:
                        logger.info("Arrived at target")
                    else:
                        logger.warning("Navigation failed — path rejected or timed out")
            except Exception as e:  # noqa: BLE001
                logger.exception("Navigation error: %s", e)

        # One navigate job at a time: a new request supersedes the old one.
        self.motion.cancel(self.JOB_NAVIGATE)
        self.motion.spawn(self.JOB_NAVIGATE, _navigate)

    def set_astar_planner(self, planner: AStarPlanner | None) -> None:
        """Inject or remove the A* planner (state lives on the follower)."""
        self.astar.planner = planner

    def set_astar_enabled(self, enabled: bool) -> None:
        """Enable or disable A* pathfinding for navigate requests."""
        self.astar.enabled = enabled
        if not enabled:
            self.motion.cancel(self.JOB_NAVIGATE)
        logger.info("A* %s", "enabled" if enabled else "disabled")

    # ------------------------------------------------------------------
    # Closed-loop drive (COZMO-35)
    # ------------------------------------------------------------------

    def drive_to_point(
        self,
        tx: float,
        ty: float,
        *,
        stop_mm: float = 20.0,
        timeout_s: float = 15.0,
        speed_mmps: float = 60.0,
        final_heading: float | None = None,
        align_tol_rad: float = 0.06,
        stall_s: float | None = None,
    ) -> bool:
        """Closed-loop differential drive toward a point, reversing if needed.

        The bearing to the target is re-aimed at the *current* robot pose every
        tick, so the robot converges onto the target (the old code steered
        along the line-of-sight captured at click time and never re-aimed,
        missing any target not exactly on that ray). Ends by turning in place
        to ``final_heading`` (if given) once close.

        Each call runs as its own ``drive`` job on :attr:`motion`, so a
        :meth:`stop` that happened before the call cannot block it, while a
        :meth:`stop` during the loop ends it at the next tick.

        Args:
            tx, ty: Target position (mm, world frame).
            stop_mm: Stop when within this distance of the target.
            timeout_s: Overall drive timeout.
            speed_mmps: Cruise wheel speed.
            final_heading: World heading (rad) to align to after arrival
                (turning in place).
            align_tol_rad: Heading tolerance for the final alignment.
            stall_s: Give up early when the pose stops changing for this
                long — a picked-up or physically blocked robot never
                moves, and waiting out ``timeout_s`` for that says
                nothing the caller can act on.
                ``None`` (default) waits out the timeout.

        Returns:
            True if the target was reached (within tolerance), False on
            timeout, stall, or when interrupted by :meth:`stop`.
        """
        cli = self.cli
        if cli is None:
            logger.warning("Drive: no robot connected")
            return False

        t0 = time.monotonic()
        reached = False
        with self.motion.begin(self.JOB_DRIVE) as job:
            try:
                anchor = _pose_signature(cli.pose)
                last_moved = time.monotonic()
                while time.monotonic() - t0 <= timeout_s:
                    if job.cancelled:
                        logger.info("Drive interrupted by stop()")
                        break
                    pose = cli.pose
                    signature = _pose_signature(pose)
                    if _pose_moved(anchor, signature):
                        anchor, last_moved = signature, time.monotonic()
                    elif stall_s is not None and time.monotonic() - last_moved > stall_s:
                        logger.warning("Drive: no robot motion for %.1fs - wheels not responding", stall_s)
                        break
                    ex, ey = tx - pose.position.x, ty - pose.position.y
                    if math.hypot(ex, ey) <= stop_mm:
                        reached = True
                        break
                    vl, vr = steer_velocities(
                        pose.position.x,
                        pose.position.y,
                        pose.rotation.angle_z.radians,
                        tx,
                        ty,
                        speed_mmps=speed_mmps,
                    )
                    cli.drive_wheels(lwheel_speed=vl, rwheel_speed=vr)
                    time.sleep(0.05)

                if not reached and time.monotonic() - t0 > timeout_s:
                    logger.warning("Nav timeout")
                if reached and final_heading is not None:
                    self._align_heading(final_heading, align_tol_rad)
            finally:
                cli.stop_all_motors()

        if reached:
            logger.info("Arrived at (%.0f, %.0f)", tx, ty)

        return reached

    def _align_heading(
        self,
        target_heading: float,
        tol_rad: float = 0.06,
        timeout_s: float = 3.0,
        turn_speed_mmps: float = 30.0,
        stall_s: float | None = None,
    ) -> None:
        """Turn in place to ``target_heading`` (world frame, radians).

        ``stall_s`` aborts early when the heading stops changing for that
        long — a picked-up or physically blocked robot never turns, so
        waiting out ``timeout_s`` for that only delays an answer the
        caller cannot act on anyway.
        """
        cli = self.cli
        if cli is None:
            return

        t0 = time.monotonic()
        with self.motion.begin(self.JOB_ALIGN) as job:
            try:
                anchor_yaw = cli.pose.rotation.angle_z.radians
                last_moved = time.monotonic()
                while time.monotonic() - t0 <= timeout_s:
                    if job.cancelled:
                        break
                    yaw = cli.pose.rotation.angle_z.radians
                    if abs(wrap_pi(yaw - anchor_yaw)) > math.radians(0.25):
                        anchor_yaw, last_moved = yaw, time.monotonic()
                    elif stall_s is not None and time.monotonic() - last_moved > stall_s:
                        logger.warning("Align: no heading response for %.1fs - wheels not responding", stall_s)
                        break
                    err = wrap_pi(target_heading - yaw)
                    if abs(err) < tol_rad:
                        break
                    omega = max(-1.0, min(1.0, err * 2.2)) * turn_speed_mmps
                    cli.drive_wheels(lwheel_speed=-omega, rwheel_speed=omega)
                    time.sleep(0.03)
            finally:
                cli.stop_all_motors()

    def _move_forward(self, dist_mm: float, timeout_s: float = 6.0) -> bool:
        """Drive straight ahead ``dist_mm`` using the onboard planner.

        ``go_to_pose`` (relative to the robot) is used instead of the Python
        closed loop (``drive_to_point``) for these very short moves: right
        after a ``go_to_pose`` the robot can still be finalizing its path and
        ignores raw ``drive_wheels`` commands. (COZMO-37)

        Returns True if the robot actually arrived within ~4mm.
        """
        cli = self.cli
        if cli is None:
            return False
        start_x = cli.pose.position.x
        start_y = cli.pose.position.y
        start_yaw = cli.pose.rotation.angle_z.radians
        cli.go_to_pose(
            Pose(
                dist_mm,
                0,
                0,
                angle_z=Angle(radians=0.0),
            ),
            relative_to_robot=True,
        )
        t0 = time.monotonic()
        while time.monotonic() - t0 <= timeout_s:
            rp = cli.pose
            moved = math.hypot(rp.position.x - start_x, rp.position.y - start_y)
            err = abs(wrap_pi(start_yaw - rp.rotation.angle_z.radians))
            if moved >= dist_mm - 4.0 and err <= math.radians(3.0):
                return True
            time.sleep(0.05)
        logger.warning("Forward move of %.0fmm not reached", dist_mm)
        return False

    # ------------------------------------------------------------------
    # Relative motion (COZMO-69)
    # ------------------------------------------------------------------

    def move_by(
        self,
        distance_mm: float,
        speed_mmps: float = 60.0,
        timeout_s: float | None = None,
        stall_s: float | None = 2.0,
    ) -> bool:
        """Drive straight forward (positive) or reverse (negative) by ``distance_mm``.

        The target is projected from the current pose along the current
        heading, and the drive ends aligned to that same heading — so the
        move changes position, not orientation. It runs as the shared
        ``drive`` job on :attr:`motion`, which :meth:`stop` cancels.

        Args:
            distance_mm: Signed distance to cover (mm).
            speed_mmps: Cruise wheel speed (mm/s); the caller caps it.
            timeout_s: Overall drive timeout; ``None`` scales with the
                distance at the requested speed.
            stall_s: Give up when the pose stops changing for this long
                (wheels not responding — see :meth:`drive_to_point`).

        Returns:
            True when the robot covered the requested distance, False
            when not connected, stalled, interrupted, or timed out.
        """
        cli = self.cli
        if cli is None:
            logger.warning("Move: no robot connected")
            return False
        pose = cli.pose
        yaw = pose.rotation.angle_z.radians
        tx = pose.position.x + distance_mm * math.cos(yaw)
        ty = pose.position.y + distance_mm * math.sin(yaw)
        # Scale the arrival radius with the request: 20 mm is too coarse
        # for a 100 mm move, 5 mm needlessly strict for a metre.
        stop_mm = max(5.0, min(20.0, abs(distance_mm) * 0.15))
        if timeout_s is None:
            timeout_s = max(5.0, abs(distance_mm) / max(speed_mmps, 1.0) * 2.0 + 3.0)
        return self.drive_to_point(
            tx,
            ty,
            stop_mm=stop_mm,
            timeout_s=timeout_s,
            speed_mmps=speed_mmps,
            final_heading=yaw,
            stall_s=stall_s,
        )

    def turn_by(
        self,
        angle_rad: float,
        tol_rad: float = 0.06,
        timeout_s: float | None = None,
        stall_s: float | None = 2.0,
    ) -> bool:
        """Turn in place by ``angle_rad`` (positive = CCW/left).

        The target heading is the current heading plus the angle, wrapped
        into [-pi, pi) — an angle past ±180° takes the short way around.
        The turn runs as the ``align`` job on :attr:`motion`, which
        :meth:`stop` cancels.

        Args:
            angle_rad: Relative angle to turn (radians).
            tol_rad: Heading error considered aligned.
            timeout_s: Turn timeout; ``None`` scales with the angle at
                the fixed turn speed (~38°/s).
            stall_s: Give up when the heading stops changing for this
                long (wheels not responding — the robot is likely picked
                up or obstructed). ``None`` waits out the timeout.

        Returns:
            True when the heading settled within ``tol_rad``, False when
            not connected, stalled, cancelled, or timed out.
        """
        cli = self.cli
        if cli is None:
            logger.warning("Turn: no robot connected")
            return False
        if timeout_s is None:
            timeout_s = 3.0 + abs(math.degrees(angle_rad)) / 30.0
        target = wrap_pi(cli.pose.rotation.angle_z.radians + angle_rad)
        self._align_heading(target, tol_rad=tol_rad, timeout_s=timeout_s, stall_s=stall_s)
        residual = abs(wrap_pi(target - cli.pose.rotation.angle_z.radians))
        return residual <= tol_rad

    # ------------------------------------------------------------------
    # Cube approach (COZMO-28 / COZMO-40)
    # ------------------------------------------------------------------

    def _get_cube_approach(self) -> CubeApproach:
        """Lazily create the CubeApproach instance with DI callbacks."""
        if self._cube_approach is None:
            world_map = self.world_map
            self._cube_approach = CubeApproach(
                cli=self.cli,
                get_cube_state=lambda idx: world_map.get_cubes().get(idx) if world_map is not None else None,
                drive_to_point=self.drive_to_point,
                move_forward=self._move_forward,
                visit=self._cube_visit,
                carry=self.carry,
            )
        return self._cube_approach

    def _get_cube_stack(self) -> CubeStack:
        """Lazily create the CubeStack with DI callbacks (shares the approach)."""
        if self._cube_stack is None:
            self._cube_stack = CubeStack(
                cli=self.cli,
                drive_to_point=self.drive_to_point,
                move_forward=self._move_forward,
                visit=self._cube_visit,
                carry=self.carry,
                approach=self._get_cube_approach(),
            )
        return self._cube_stack

    @staticmethod
    def _pick_face_normal(
        cube_pos: np.ndarray,
        cube_rot: np.ndarray,
        robot_pos: tuple[float, float],
        robot_yaw: float,
    ) -> tuple[float, float, float] | None:
        """Pick the cube face to approach — delegates to CubeApproach."""
        return CubeApproach.pick_face_normal(cube_pos, cube_rot, robot_pos, robot_yaw)

    def approach_cube(self, cube_id: int) -> bool:
        """Approach and align with the cube with the given ID (1-3).

        Resolves the cube's world position from the world map, then starts
        the two-step approach in the background.

        Returns:
            True if a cube visit was actually started (a grab, or a stack
            when a cube is riding the forks), False otherwise — the cube is
            unknown, there is no robot/map, or another cube visit is
            already in flight.
        """
        if not self.cli:
            logger.warning("Approach: no robot connected")
            return False
        wm = self.world_map
        if wm is None:
            logger.warning("Approach: world map not available")
            return False
        cube_index = to_world_index(cube_id)
        cube = wm.get_cube(cube_index)
        if cube is None:
            logger.warning("Approach: cube %d not in world map", cube_id)
            return False
        position = cube.position
        return self.approach_cube_at(cube_index, float(position[0]), float(position[1]))

    def approach_cube_at(self, cube_idx: int, cube_x: float, cube_y: float) -> bool:
        """Approach the cube at world position (x, y) — grab or stack.

        ``cube_idx`` is the 0-indexed world-map key; the browser message
        router resolves the pose and calls this directly. Use
        :meth:`approach_cube` to address a cube by its ID (1-3).

        Normally this starts a :class:`CubeApproach` (drive up, grab).
        While a cube is believed to be riding the raised forks
        (``carry.on_forks``) and the target is a *different* cube, it
        routes to :class:`CubeStack` instead — so clicking a second cube
        in the viewer stacks the held cube on it rather than dumping it.

        Returns True when the visit started, False when it was refused
        (no robot, or another cube visit — approach or stack — is already
        in flight).
        """
        if not self.cli:
            logger.warning("Approach: no robot connected")
            return False
        carry = self.carry
        if carry is not None and carry.on_forks and to_cube_id(cube_idx) != carry.cube_id:
            return self._get_cube_stack().stack(cube_idx, cube_x, cube_y)
        return self._get_cube_approach().approach(cube_idx, cube_x, cube_y)

    def stack_cube(self, cube_id: int) -> bool:
        """Approach the cube with the given ID (1-3) and stack the held cube on it.

        Returns True if the stack was started, False when refused: no cube
        on the raised forks, the target *is* the held cube, the target is
        not in the world map, or another cube visit is already in flight.
        The same routing happens automatically for click-to-approach while
        carrying — this method is the explicit library API.
        """
        carry = self.carry
        if carry is None or not carry.on_forks:
            logger.warning("Stack: no cube on the forks - not stacking")
            return False
        if carry.cube_id == cube_id:
            logger.warning("Stack: cube %d is already on the forks", cube_id)
            return False
        return self.approach_cube(cube_id)

    # ------------------------------------------------------------------
    # Docking (COZMO-28)
    # ------------------------------------------------------------------

    def dock(self) -> None:
        """Drive robot back into the dock (in the background).

        Delegates to :class:`Docking`, which owns the sequence (approach,
        reverse into contact, retries) and registers the shared ``dock``
        job that :meth:`stop` cancels.
        """
        self.docking.dock()

    # ------------------------------------------------------------------
    # ArUco docking  (COZMO-66)
    # ------------------------------------------------------------------

    CHARGER_MARKER_ID: ClassVar[int] = 0

    def dock_on_aruco(self, marker_id: int = 0) -> bool:
        """Dock on an ArUco marker: approach, refresh pose, reverse in.

        Drives to a standoff on the marker's normal (25 cm out, facing
        it) so the robot arrives on the charger's centreline, dwells
        until that pose converges, verifies the robot is centred on the
        marker (the reverse is open loop), then turns 180° and reverses
        into the charger until charging is detected (the real stop) or
        the robot has deliberately overshot the docked distance to bump
        the back of the charger (15 s timeout).

        The marker sits at the *end* of the docking station, on its
        centreline — the standoff uses the marker's outward normal to
        land on that centreline (see
        :mod:`pycozmo_ng.navigation.aruco_docking`).

        After a successful dock the pose origin is reset — the robot's
        docked position becomes the world origin (the point 65 mm in
        front of marker 0) — and the marker's world-map pose is snapped
        to its exact docked position (65 mm behind the camera).

        Args:
            marker_id: ArUco dictionary ID of the charger marker
                (default ``CHARGER_MARKER_ID`` = 0).

        Returns:
            True if a dock attempt was **started** (marker has been
            fused into the world map), False otherwise.
        """
        if not self.cli:
            logger.warning("ArUco dock: no robot connected")
            return False
        wm = self.world_map
        if wm is None:
            logger.warning("ArUco dock: world map not available")
            return False
        markers = wm.get_aruco_markers()
        if marker_id not in markers:
            logger.warning("ArUco dock: marker %d not in world map", marker_id)
            return False

        aruco_docking = ArucoDocking(
            cli=self.cli,
            get_marker_state=lambda mid: wm.get_aruco_markers().get(mid),
            dock_state=self.ds,
            motion=self.motion,
            on_docked=lambda: self._on_aruco_docked(marker_id),
        )
        return aruco_docking.dock(marker_id)

    def _on_aruco_docked(self, marker_id: int) -> None:
        """Post-dock anchoring — runs once the charger contacts detect.

        Resets the pose origin (the docked robot pose becomes the world
        origin — the point 65 mm in front of the marker), then snaps the
        marker to its exact docked position.

        If the origin reset did not confirm (telemetry timeout), the
        robot's world pose is not at (0, 0, 0) and snapping the marker
        would place it at a wrong spot — so the snap is skipped and the
        marker stays at whatever the vision pipeline last fused.
        """
        self._anchor_dock(marker_id, "ArUco dock")

    def _snap_marker_dock_pose(self, marker_id: int = 0) -> None:
        r"""Hard-set marker *marker\_id* to its exact docked position.

        Called after a successful origin reset (:meth:`_reset_pose_origin`)
        while the robot is docked on the charger.  The robot's world pose
        is known (origin + yaw), and the marker sits at a fixed offset
        65 mm behind the camera — so its exact world pose is computed
        from geometry, not from vision.
        """
        wm = self.world_map
        cli = self.cli
        if wm is None or cli is None:
            return
        yaw = cli.pose.rotation.angle_z.radians
        dist = ArucoDocking.DOCKED_MARKER_DIST_MM  # 65 mm
        # Marker body offset: straight behind the robot (camera and body
        # origin share the same XY horizontally — camera is directly above).
        offset = np.array([-dist, 0.0, 0.0], dtype=np.float64)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        # World position = R_z(yaw) @ offset (robot at origin after reset)
        pos = np.array(
            [
                cos_y * offset[0] - sin_y * offset[1],
                sin_y * offset[0] + cos_y * offset[1],
                0.0,
            ],
            dtype=np.float64,
        )
        # Marker normal (z) points toward the robot = + (cos yaw, sin yaw)
        zn = np.array([cos_y, sin_y, 0.0], dtype=np.float64)
        y_up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_up, zn)
        rot = np.column_stack([x_axis, y_up, zn])
        wm.set_aruco_world_pose(
            marker_id=marker_id,
            position_world=pos,
            rotation_world=rot,
            size_mm=22.5,
        )
        logger.info(
            "Marker %d dock pose snapped to (%.1f, %.1f) — %.0f mm behind camera",
            marker_id,
            pos[0],
            pos[1],
            ArucoDocking.DOCKED_MARKER_DIST_MM,
        )

    # ------------------------------------------------------------------
    # Stop (public motion control)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop all movement.

        Cancels every motion job (navigate, dock, approach, test mode,
        closed-loop drive) in one call, clears the published A* path, and
        stops the motors. Jobs begun after the stop are not cancelled, so
        the next motion command re-arms movement.
        """
        cancelled = self.motion.cancel_all()
        if cancelled:
            logger.info("Cancelled motion jobs: %s", ", ".join(sorted(set(cancelled))))
        if self.world_map is not None:
            self.world_map.clear_astar_path()
        if self.cli is not None:
            self.cli.stop_all_motors()
        logger.info("Stopped")

    # ------------------------------------------------------------------
    # Test mode — loop approach on cubes with 3 sec pauses
    # ------------------------------------------------------------------

    def set_test_mode(self, enabled: bool) -> None:
        """Start or stop the test mode loop."""
        if not enabled:
            self.motion.cancel(self.JOB_TEST_MODE)
            logger.info("Test mode OFF")
            return
        if self.JOB_TEST_MODE in self.motion.active_names:
            return
        self.motion.spawn(self.JOB_TEST_MODE, self._test_mode_loop)
        logger.info("Test mode ON")

    def _test_mode_loop(self, job: MotionToken) -> None:
        """Continuously approach cubes with 3 second waits."""
        while not job.cancelled:
            if not self.cli or self.world_map is None:
                self._sleep_while_running(job, 1.0)
                continue
            cubes = self.world_map.get_cubes()
            if not cubes:
                self._sleep_while_running(job, 1.0)
                continue
            cube_index = next(iter(cubes))
            cube = cubes[cube_index]
            cx, cy, _ = cube.position
            logger.info("Test mode: approaching cube %d", to_cube_id(cube_index))
            self.approach_cube_at(cube_index, float(cx), float(cy))
            self._sleep_while_running(job, 3.0)

    @staticmethod
    def _sleep_while_running(job: MotionToken, seconds: float) -> None:
        """Sleep in short slices so a cancelled job notices within ~100ms."""
        deadline = time.monotonic() + seconds
        while not job.cancelled and time.monotonic() < deadline:
            time.sleep(0.1)
