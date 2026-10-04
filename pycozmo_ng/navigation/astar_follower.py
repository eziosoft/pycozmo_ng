"""AStarFollower — plan an A* path and follow it with the firmware follower.

Extracted from ``Navigator`` (click-to-navigate, COZMO-28/COZMO-30): one
feature — planning, publishing the path for the 3D viewer, executing it
with ``Client.execute_polyline``, the occupancy interrupt poll, and the
built-in ``go_to_pose`` fallback — lived as ~170 lines of private methods
and module constants on ``Navigator``.  As its own class it needs only a
client and a world-map reference, so the follow loop is testable without
constructing a navigation subsystem.

``Navigator.navigate_to`` remains the entry point: it decides *whether*
to navigate (cancel, distance, job lifecycle) and delegates *how* to
follow an A* path here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

from pycozmo_ng.logger import logger
from pycozmo_ng.navigation.astar import AStarPlanner, PathPoint
from pycozmo_ng.navigation.motion_token import MotionToken
from pycozmo_ng.util import Angle, Pose

if TYPE_CHECKING:
    from pycozmo_ng.navigation.world_map import WorldMap


class AStarFollower:
    """Compute, publish, and follow an A* path through the occupancy grid.

    State: ``planner`` (the A* planner, injected by ``Brain``), ``enabled``
    (whether navigate requests use A* at all — the browser toggle), and
    ``waypoints`` (the path currently being followed, owned by the
    navigate job's thread and read by its interrupt poll).

    The rest of the behaviour sits behind :meth:`follow`, which runs until
    the goal is reached, the job is cancelled, or replanning gives up.
    """

    #: Considered arrived within this distance of the goal (mm).
    GOAL_TOL_MM: ClassVar[float] = 30.0
    #: Maximum consecutive replans before giving up on a goal.
    MAX_REPLAN_ATTEMPTS: ClassVar[int] = 10
    #: Waypoints closer to the robot than this are skipped by the occupancy
    #: monitor — localization noise near the body must not trigger replans.
    MONITOR_SKIP_MM: ClassVar[float] = 75.0

    def __init__(self, world_map: WorldMap | None = None) -> None:
        self.cli: Any = None  # robot client, injected by Navigator
        self.world_map = world_map
        self.planner: AStarPlanner | None = None
        self.enabled = False
        # Owned by the navigate job's thread; read by its interrupt poll.
        self.waypoints: list[PathPoint] = []

    # ------------------------------------------------------------------
    # Following
    # ------------------------------------------------------------------

    def follow(self, goal_x: float, goal_y: float, job: MotionToken) -> None:
        """Compute an A* path and follow it with the firmware path follower.

        The remaining path is sent as a single built-in Cozmo path
        (``Client.execute_polyline``), so the robot runs its own
        high-rate trajectory generator instead of the Python closed loop
        (WiFi latency made ``drive_to_point`` unreliable). The occupancy
        map is monitored while the path executes — if a remaining
        waypoint becomes occupied the path is interrupted and a new A*
        search is issued from the current pose. The active path is
        published to WorldMap for 3D visualization and cleared when
        navigation ends. Cancelling ``job`` ends the loop and clears the
        published path.
        """
        cli = self.cli
        if cli is None:
            return

        if self.planner is None:
            logger.warning("A*: no planner — falling back to built-in navigation")
            self.go_to_pose_goal(goal_x, goal_y, job)
            return

        robot_pose = cli.pose
        start_x = robot_pose.position.x
        start_y = robot_pose.position.y

        self.waypoints = self.planner.find_path(start_x, start_y, goal_x, goal_y) or []
        self._publish_path()
        logger.info("A*: computed %d waypoints", len(self.waypoints))

        if not self.waypoints:
            logger.warning("A*: no path found — falling back to built-in navigation")
            self._clear_path()
            self.go_to_pose_goal(goal_x, goal_y, job)
            return

        try:
            attempts = 0
            while not job.cancelled:
                pose = cli.pose
                if (
                    math.hypot(goal_x - pose.position.x, goal_y - pose.position.y)
                    <= self.GOAL_TOL_MM
                ):
                    logger.info("A*: reached goal")
                    break

                attempts += 1
                if attempts > self.MAX_REPLAN_ATTEMPTS:
                    logger.warning("A*: giving up after max replan attempts")
                    break

                pts = [(w.x, w.y) for w in self.waypoints]
                arrived = cli.execute_polyline(
                    pts,
                    final_heading_rad=self._final_heading(
                        pts, start_x, start_y, goal_x, goal_y
                    ),
                    should_interrupt=lambda: self.should_interrupt(job),
                )

                if job.cancelled:
                    logger.info("A*: path following cancelled")
                    break

                pose = cli.pose
                if arrived and (
                    math.hypot(goal_x - pose.position.x, goal_y - pose.position.y)
                    <= self.GOAL_TOL_MM
                ):
                    logger.info("A*: reached goal")
                    break

                # Interrupted (new obstacle) or fell short — replan from here.
                new_path = self.planner.find_path(
                    pose.position.x, pose.position.y, goal_x, goal_y
                )
                if not new_path:
                    logger.warning("A*: replan failed — no path to goal")
                    break
                self.waypoints = new_path
                self._publish_path()
                logger.info("A*: replanned — %d waypoints", len(new_path))
        finally:
            self._clear_path()

    @staticmethod
    def _final_heading(
        pts: list[tuple[float, float]],
        start_x: float,
        start_y: float,
        goal_x: float,
        goal_y: float,
    ) -> float:
        """World heading facing the final approach into the goal."""
        if len(pts) >= 2:
            hx = goal_x - pts[-2][0]
            hy = goal_y - pts[-2][1]
            if math.hypot(hx, hy) >= 1.0:
                return math.atan2(hy, hx)
        return math.atan2(goal_y - start_y, goal_x - start_x)

    def should_interrupt(self, job: MotionToken) -> bool:
        """Poll callback for ``Client.execute_polyline``.

        Returns True when path execution must stop: the navigate job was
        cancelled, or a remaining waypoint has become occupied in the
        updated occupancy map (a replan follows). Waypoints close to the
        robot are skipped — localization noise near the body must not
        trigger spurious replans.
        """
        if job.cancelled:
            return True
        planner = self.planner
        cli = self.cli
        if planner is None or cli is None:
            return False
        pose = cli.pose
        for w in self.waypoints:
            if (
                math.hypot(w.x - pose.position.x, w.y - pose.position.y)
                < self.MONITOR_SKIP_MM
            ):
                continue
            if not planner.is_traversable(w.x, w.y):
                logger.warning(
                    "A*: waypoint (%.0f, %.0f) now blocked — replanning", w.x, w.y
                )
                return True
        return False

    def go_to_pose_goal(
        self, goal_x: float, goal_y: float, job: MotionToken | None = None
    ) -> None:
        """Drive straight to the goal with the robot's built-in path follower."""
        cli = self.cli
        if cli is None:
            return
        pose = cli.pose
        heading = math.atan2(goal_y - pose.position.y, goal_x - pose.position.x)
        ok = cli.go_to_pose(
            Pose(goal_x, goal_y, 0, angle_z=Angle(radians=heading)),
            relative_to_robot=False,
            should_interrupt=(None if job is None else lambda: job.cancelled),
        )
        if ok:
            logger.info("Arrived at target")
        else:
            logger.warning("Navigation failed — path rejected or timed out")

    # ------------------------------------------------------------------
    # Path publication (3D viewer)
    # ------------------------------------------------------------------

    def _publish_path(self) -> None:
        if self.world_map is not None:
            self.world_map.set_astar_path(
                [[float(w.x), float(w.y)] for w in self.waypoints]
            )

    def _clear_path(self) -> None:
        if self.world_map is not None:
            self.world_map.clear_astar_path()
