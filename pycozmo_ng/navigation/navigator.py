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
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def on_charging_change(self, cli, state: bool) -> None:
        """Record whether the robot is actively charging."""
        del cli
        with self.ds.lock:
            self.ds.charging = state
        if state:
            logger.info("Charging detected")
            threading.Thread(target=self._reset_pose_origin, daemon=True).start()

    def _reset_pose_origin(self) -> None:
        """Re-anchor the localization origin at the current pose.

        Sends SetOrigin so the robot's current pose becomes the new
        world frame (0,0,0) and records it as the dock pose. (COZMO-28)
        """
        cli = self.cli
        if cli is None:
            return
        with self.ds.lock:
            if self.ds.origin_reset_origin_id == cli.pose.origin_id:
                return

        reset_pkt = pycozmo_ng.protocol_encoder.SetOrigin(
            pose_frame_id=cli.pose_frame_id + 1,
            pose_origin_id=cli.pose.origin_id + 1,
        )
        cli.conn.send(reset_pkt)
        time.sleep(0.5)
        with self.ds.lock:
            self.ds.origin_reset_origin_id = cli.pose.origin_id
            self.ds.dock_pose = cli.pose
        logger.info("Pose origin reset - current pose is now origin")

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

        Returns:
            True if the target was reached (within tolerance), False on
            timeout or when interrupted by :meth:`stop`.
        """
        cli = self.cli
        if cli is None:
            logger.warning("Drive: no robot connected")
            return False

        t0 = time.monotonic()
        reached = False
        with self.motion.begin(self.JOB_DRIVE) as job:
            try:
                while time.monotonic() - t0 <= timeout_s:
                    if job.cancelled:
                        logger.info("Drive interrupted by stop()")
                        break
                    pose = cli.pose
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
    ) -> None:
        """Turn in place to ``target_heading`` (world frame, radians)."""
        cli = self.cli
        if cli is None:
            return

        t0 = time.monotonic()
        with self.motion.begin(self.JOB_ALIGN) as job:
            try:
                while time.monotonic() - t0 <= timeout_s:
                    if job.cancelled:
                        break
                    yaw = cli.pose.rotation.angle_z.radians
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

        Drives to a point 20 cm in front of the marker facing it so the
        camera sees it and the world map refreshes its fused pose,
        dwells until that pose converges, corrects the approach point,
        verifies the robot is centred on the marker's normal line (the
        reverse is open loop), then turns 180° and reverses into the
        charger until charging is detected (the real stop) or the robot
        has deliberately overshot the docked distance to bump the back
        of the charger (15 s timeout).

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
        """
        self._reset_pose_origin()
        self._snap_marker_dock_pose(marker_id)

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
