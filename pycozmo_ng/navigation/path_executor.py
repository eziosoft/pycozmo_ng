"""PathExecutor — the firmware path-following state machine.

Extracted from ``Client`` (a 1175-line class where handshake, robot-state
decode, image-chunk reassembly, actuator commands, audio and path execution
sat side by side). The path cluster — settle, event-id allocation,
``ClearPath`` → ``AppendPathSeg*`` → ``ExecutePath``, waiting for a
``PathFollowingEvent`` *or* pose proximity, and the chunked polyline loop —
is ~450 lines that were reachable only through a constructed ``Client``,
which needs a live ``conn.Connection``. So it had no tests at all, while
:class:`~pycozmo_ng.navigation.astar_follower.AStarFollower`,
:class:`~pycozmo_ng.navigation.docking.Docking`,
:class:`~pycozmo_ng.navigation.cube_approach.CubeApproach` and
:class:`~pycozmo_ng.navigation.navigator.Navigator` all stand on it.

Interface: :meth:`PathExecutor.go_to_pose` and
:meth:`PathExecutor.execute_polyline`. Everything else — the path lock, the
event-id counter, settling, chunking, and the two wait strategies — sits
behind them. The single port is :class:`PathRobot`: send a packet, stop the
motors, and read the live pose / wheel speeds / handler registry.
:class:`~pycozmo_ng.client.Client` satisfies it, so ``Client.go_to_pose`` and
``Client.execute_polyline`` stay one-line delegations and no navigation
caller changed.

Why the lock lives here: the firmware path buffer holds one path at a time,
so event-id allocation + build-and-wait must be serialized. Two concurrent
callers (e.g. a navigate and a dock job) would otherwise wipe each other's
segments or steal each other's completion events.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from threading import Event, Lock
from typing import TYPE_CHECKING, Any, Protocol

from pycozmo_ng.logger import logger
from pycozmo_ng.protocol import protocol_encoder
from pycozmo_ng.util import Pose, Speed

if TYPE_CHECKING:
    from pycozmo_ng.event import Handler

__all__ = [
    "MAX_PATH_SEGMENTS",
    "PathExecutor",
    "PathRobot",
    "build_polyline_packets",
    "chunk_polyline_points",
]

#: Firmware path-buffer capacity per ``ExecutePath`` — appending more
#: segments than this is silently rejected
#: (``Path.AppendLine.ExceededMaxSize: MAX_NUM_PATH_SEGMENTS: 6``).
MAX_PATH_SEGMENTS = 6


class PathRobot(Protocol):
    """The slice of the robot client a :class:`PathExecutor` talks to.

    Deliberately narrow: the wire, the stop command, the state the wait
    loops poll, and the handler registry the completion waiters install
    themselves on. ``Client`` satisfies it; a test double needs no
    connection.
    """

    pose: Pose
    left_wheel_speed: Speed
    right_wheel_speed: Speed

    def send(self, pkt: protocol_encoder.Packet) -> None: ...

    def stop_all_motors(self) -> None: ...

    def add_handler(self, event: Any, f: Callable[..., None], one_shot: bool = False) -> Handler: ...

    def del_handler(self, event: Any, handler: Handler) -> None: ...


def _filter_polyline_points(
    points: Sequence[tuple[float, float]],
    start: tuple[float, float],
) -> list[tuple[float, float]]:
    """Drop consecutive points closer than 10 mm to the previous kept point.

    The first point is measured from *start*. This keeps the firmware
    from seeing zero-length segments.
    """
    waypoints: list[tuple[float, float]] = []
    fx, fy = float(start[0]), float(start[1])
    for x, y in points:
        x, y = float(x), float(y)
        if math.hypot(x - fx, y - fy) >= 10.0:
            waypoints.append((x, y))
            fx, fy = x, y
    return waypoints


def chunk_polyline_points(
    points: Sequence[tuple[float, float]],
    *,
    final_heading_rad: float | None = None,
    max_segments: int = MAX_PATH_SEGMENTS,
) -> list[list[tuple[float, float]]]:
    """Split *points* into chunks that each fit the firmware path buffer.

    A chunk of *n* points costs *n* line segments, plus one point-turn
    segment for the final chunk when *final_heading_rad* is set — the sum
    must stay within *max_segments* or the robot drops the overflow
    (``Path.AppendLine.ExceededMaxSize``).

    Chunks are taken from the end so the final chunk reserves room for
    the turn while every other chunk may use the full capacity. Point
    order is preserved across chunks. Returns ``[]`` for empty input.
    """
    n = len(points)
    if n == 0:
        return []
    last_cap = max(max_segments - (1 if final_heading_rad is not None else 0), 1)
    chunks_rev: list[list[tuple[float, float]]] = []
    hi = n
    final = True
    while hi > 0:
        cap = last_cap if final else max_segments
        take = min(cap, hi)
        chunks_rev.append([(float(x), float(y)) for x, y in points[hi - take : hi]])
        hi -= take
        final = False
    chunks_rev.reverse()
    return chunks_rev


def build_polyline_packets(
    points: Sequence[tuple[float, float]],
    start: tuple[float, float],
    event_id: int,
    speed_mmps: float = 90.0,
    accel_mmps2: float = 40.0,
    decel_mmps2: float = 40.0,
    final_heading_rad: float | None = None,
) -> list:
    """Build the packet sequence for a multi-segment firmware path.

    Returns ``[ClearPath, AppendPathSegLine..., (AppendPathSegPointTurn),
    ExecutePath]``. Consecutive points closer than 10 mm to the previous
    kept point (starting from *start*) are dropped so the firmware never
    sees zero-length segments. Empty list when there is nothing to
    execute.
    """
    waypoints = _filter_polyline_points(points, start)

    if not waypoints and final_heading_rad is None:
        return []

    packets: list = [protocol_encoder.ClearPath()]
    px, py = float(start[0]), float(start[1])
    for tx, ty in waypoints:
        packets.append(
            protocol_encoder.AppendPathSegLine(
                from_x=px,
                from_y=py,
                to_x=tx,
                to_y=ty,
                speed_mmps=speed_mmps,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
            )
        )
        px, py = tx, ty

    if final_heading_rad is not None:
        packets.append(
            protocol_encoder.AppendPathSegPointTurn(
                x=px,
                y=py,
                angle_rad=float(final_heading_rad),
                angle_tolerance_rad=0.05,
                speed_mmps=25.0,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
            )
        )

    packets.append(protocol_encoder.ExecutePath(event_id=event_id))
    return packets


class PathExecutor:
    """Drive the robot along firmware paths.

    Two entry points, both serialized on the same lock because the firmware
    buffer holds one path at a time:

    - :meth:`go_to_pose` — one straight segment plus an in-place turn to a
      target pose.
    - :meth:`execute_polyline` — a waypoint run, split into batches that
      each fit the firmware buffer and rebuilt from the measured pose
      between batches.

    Each blocks until the path resolves — a ``PathFollowingEvent`` for its
      event id, or pose proximity (the firmware often omits the event for
      simple paths, COZMO-53) — and honours ``should_interrupt`` so a
      cancelled motion job or a stolen driver lease ends the path.
    """

    def __init__(self, robot: PathRobot) -> None:
        self.robot = robot
        # Monotonic event id (modulo 65535) so a new path never reuses the
        # id of a PathFollowingEvent that may still be in flight.
        self._event_id = 100
        # Serializes the firmware path protocol: the path buffer holds one
        # path at a time, so event-id allocation + ClearPath → Append… →
        # ExecutePath + the wait all run under this lock. Two concurrent
        # callers (e.g. a navigate and a dock job) can no longer wipe each
        # other's segments or steal each other's completion events.
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Single-pose paths
    # ------------------------------------------------------------------

    def go_to_pose(
        self,
        pose: Pose,
        relative_to_robot: bool = False,
        timeout_s: float | None = 30.0,
        position_tol_mm: float = 15.0,
        angle_tol_rad: float = 0.15,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Move to a specific pose (position and orientation).

        Settles the robot first (the firmware rejects paths that start
        while moving), clears any leftover buffered path, then sends
        ``AppendPathSegLine`` + ``AppendPathSegPointTurn`` +
        ``ExecutePath`` and blocks until either:
        1. A ``PathFollowingEvent`` for this event id reports
           completion, **or**
        2. The robot pose is within ``position_tol_mm`` mm and
           ``angle_tol_rad`` rad of the target (pose polling — Cozmo
           firmware often omits ``PathFollowingEvent`` for simple
           paths, COZMO-53).

        The whole build-and-wait is serialized on an internal lock: the
        firmware path buffer holds one path at a time, so a second caller
        starts only after the first resolves instead of wiping its
        segments mid-flight.

        Args:
            pose: Target pose (world frame, or robot-relative).
            relative_to_robot: Interpret *pose* relative to the robot.
            timeout_s: Overall wait bound.
            position_tol_mm: Pose-proximity completion tolerance.
            angle_tol_rad: Heading completion tolerance.
            should_interrupt: Polled while waiting; the first True stops
                the motors and returns False. Motion jobs pass
                ``lambda: job.cancelled`` so a stolen driver lease (or
                ``Navigator.stop()``) actually ends the path.

        Returns:
            True when the path completed (event or pose proximity),
            False when it was interrupted, rejected, or timed out
            short of the target. On timeout a warning is logged — the
            robot may still be moving.
        """
        with self._lock:
            return self._go_to_pose(
                pose,
                relative_to_robot=relative_to_robot,
                timeout_s=timeout_s,
                position_tol_mm=position_tol_mm,
                angle_tol_rad=angle_tol_rad,
                should_interrupt=should_interrupt,
            )

    def _go_to_pose(
        self,
        pose: Pose,
        *,
        relative_to_robot: bool = False,
        timeout_s: float | None = 30.0,
        position_tol_mm: float = 15.0,
        angle_tol_rad: float = 0.15,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Body of :meth:`go_to_pose` — runs with ``_lock`` held."""
        robot = self.robot
        self._settle_robot()

        if relative_to_robot:
            robot_pose = robot.pose
            pose = Pose(
                robot_pose.position.x,
                robot_pose.position.y,
                robot_pose.position.z,
                angle_z=robot_pose.rotation.angle_z,
            ).define_pose_relative_this(pose)

        if should_interrupt is not None and should_interrupt():
            logger.info("go_to_pose: cancelled before the path was sent")
            return False

        event_id = self._next_event_id()

        e = Event()
        outcome: list[bool] = []

        def event_wait(_, pkt2: protocol_encoder.PathFollowingEvent):
            if pkt2.event_id != event_id:
                return
            if pkt2.event_type == protocol_encoder.PathEventType.PATH_COMPLETED:
                outcome.append(True)
                e.set()
            elif pkt2.event_type == protocol_encoder.PathEventType.PATH_INTERRUPTED:
                outcome.append(False)
                e.set()
            # PATH_STARTED: keep waiting.

        handler = robot.add_handler(protocol_encoder.PathFollowingEvent, event_wait)
        completed = False
        interrupted = False
        dist = float("inf")
        angle_err = float("inf")
        try:
            # Drop leftover segments from a previous or rejected path —
            # appended segments would chain onto them and start
            # discontinuously from the robot's position.
            robot.send(protocol_encoder.ClearPath())
            start_x = robot.pose.position.x
            start_y = robot.pose.position.y
            # Same 10 mm floor as _filter_polyline_points: a zero-length
            # line segment makes the firmware assert in trig_fast.cpp
            # (atan2(0, 0) — "y != 0 || x != 0") and crash. A degenerate
            # target still gets its point-turn below.
            if math.hypot(pose.position.x - start_x, pose.position.y - start_y) >= 10.0:
                pkt: protocol_encoder.Packet = protocol_encoder.AppendPathSegLine(
                    from_x=start_x,
                    from_y=start_y,
                    to_x=pose.position.x,
                    to_y=pose.position.y,
                    speed_mmps=100.0,
                    accel_mmps2=20.0,
                    decel_mmps2=20.0,
                )
                robot.send(pkt)
            pkt = protocol_encoder.AppendPathSegPointTurn(
                x=pose.position.x,
                y=pose.position.y,
                angle_rad=pose.rotation.angle_z.radians,
                angle_tolerance_rad=0.01,
                speed_mmps=25.0,
                accel_mmps2=20.0,
                decel_mmps2=20.0,
            )
            robot.send(pkt)
            pkt = protocol_encoder.ExecutePath(event_id=event_id)
            robot.send(pkt)

            # Wait for either the event OR pose proximity — whichever comes first.
            deadline = time.perf_counter() + (timeout_s or 30.0)
            while time.perf_counter() < deadline:
                # Fast path: terminal event arrived
                if e.wait(timeout=0.25):
                    completed = bool(outcome and outcome[0])
                    break
                # Cancellation: a motion job cancelled or the driver lease
                # moved on — stop the wheels and give the robot back.
                if should_interrupt is not None and should_interrupt():
                    robot.stop_all_motors()
                    interrupted = True
                    break
                # Pose-polling fallback: check if we're close enough
                dx = robot.pose.position.x - pose.position.x
                dy = robot.pose.position.y - pose.position.y
                dist = math.hypot(dx, dy)
                angle_err = abs(
                    math.remainder(
                        robot.pose.rotation.angle_z.radians - pose.rotation.angle_z.radians,
                        2 * math.pi,
                    )
                )
                if dist <= position_tol_mm and angle_err <= angle_tol_rad:
                    completed = True
                    break
        finally:
            robot.del_handler(protocol_encoder.PathFollowingEvent, handler)

        if interrupted:
            logger.info("go_to_pose: interrupted — motors stopped")
            return False

        if not completed:
            if outcome and not outcome[0]:
                logger.info(
                    "go_to_pose: path interrupted — dist=%.1f mm, angle_err=%.1f deg",
                    dist,
                    math.degrees(angle_err),
                )
            else:
                logger.warning(
                    "go_to_pose: timed out after %.0fs — dist=%.1f mm, angle_err=%.1f deg",
                    timeout_s if timeout_s is not None else 0,
                    dist,
                    math.degrees(angle_err),
                )
        return completed

    # ------------------------------------------------------------------
    # Waypoint polylines
    # ------------------------------------------------------------------

    def execute_polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        speed_mmps: float = 90.0,
        accel_mmps2: float = 40.0,
        decel_mmps2: float = 40.0,
        final_heading_rad: float | None = None,
        timeout_s: float | None = None,
        position_tol_mm: float = 20.0,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Follow a multi-segment path with the robot's built-in path follower.

        The firmware path buffer holds at most ``MAX_PATH_SEGMENTS``
        (6) segments per ``ExecutePath``, so long paths are split into
        sequential batches: each batch sends ``ClearPath`` +
        ``AppendPathSegLine``... + ``ExecutePath`` and blocks until it
        finishes before the next batch starts from the robot's actual
        pose (rebuilt after a settle so the follower never sees a
        discontinuous start).

        Unlike the Python-side closed loop
        (``Navigator.drive_to_point``), the trajectory — accel/decel
        ramps and wheel control — runs on the robot at high rate,
        independent of WiFi latency. This is the recommended way to drive
        along a planned (e.g. A*) polyline.

        Args:
            points: Waypoints in world frame (mm, X forward / Y left), in
                order. The first segment starts at the robot's current
                pose. Degenerate segments (< 10 mm) are dropped.
            speed_mmps: Cruise speed for the line segments.
            accel_mmps2: Acceleration limit (mm/s²).
            decel_mmps2: Deceleration limit (mm/s²).
            final_heading_rad: If set, finish with an in-place turn to
                this world heading at the last point.
            timeout_s: Overall timeout across all batches. Defaults to
                a length-based estimate (3× straight-line time at
                *speed_mmps*, min 10 s).
            position_tol_mm: Pose-proximity tolerance for the polling
                fallback (firmware sometimes omits ``PathFollowingEvent``
                for simple paths — COZMO-53).
            should_interrupt: Polled ~10× per second while waiting (and
                between batches); when it returns True the motors are
                stopped and the method returns False. Used for
                cancellation and mid-path replanning.

        Returns:
            True when every batch completed (firmware event or pose
            proximity), False when interrupted or timed out short of
            the goal.

        The whole call is serialized with :meth:`go_to_pose` on the path
        lock: the firmware buffer holds one path at a time.
        """
        with self._lock:
            return self._execute_polyline(
                points,
                speed_mmps=speed_mmps,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
                final_heading_rad=final_heading_rad,
                timeout_s=timeout_s,
                position_tol_mm=position_tol_mm,
                should_interrupt=should_interrupt,
            )

    def _execute_polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        speed_mmps: float = 90.0,
        accel_mmps2: float = 40.0,
        decel_mmps2: float = 40.0,
        final_heading_rad: float | None = None,
        timeout_s: float | None = None,
        position_tol_mm: float = 20.0,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Body of :meth:`execute_polyline` — runs with ``_lock`` held."""
        robot = self.robot
        self._settle_robot()

        start = (robot.pose.position.x, robot.pose.position.y)
        waypoints = _filter_polyline_points(points, start)
        if not waypoints and final_heading_rad is None:
            # Nothing to drive — already at (or within 10 mm of) the goal.
            return True

        path_len = 0.0
        fx, fy = start
        for x, y in waypoints:
            path_len += math.hypot(x - fx, y - fy)
            fx, fy = x, y
        end_x, end_y = waypoints[-1] if waypoints else start

        if timeout_s is None:
            timeout_s = max(10.0, path_len / max(speed_mmps, 1.0) * 3.0 + 5.0)
        deadline = time.perf_counter() + timeout_s

        chunks = chunk_polyline_points(waypoints, final_heading_rad=final_heading_rad)
        if not chunks:
            # All points degenerate but a final heading was requested —
            # a single turn-only batch at the current pose.
            chunks = [[]]
        for index, chunk in enumerate(chunks):
            if should_interrupt is not None and should_interrupt():
                robot.stop_all_motors()
                return False
            if index > 0:
                # Previous batch ended near its last point; wait for rest
                # and rebuild the first segment from the actual pose so
                # the follower never sees a discontinuous start.
                self._settle_robot(timeout_s=0.5)
            batch_start = (robot.pose.position.x, robot.pose.position.y)
            event_id = self._next_event_id()
            packets = build_polyline_packets(
                chunk,
                batch_start,
                event_id,
                speed_mmps=speed_mmps,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
                final_heading_rad=final_heading_rad if index == len(chunks) - 1 else None,
            )
            if not packets:
                continue

            batch_end = batch_start
            for pkt in packets:
                if isinstance(pkt, protocol_encoder.AppendPathSegLine):
                    batch_end = (pkt.to_x, pkt.to_y)
                elif isinstance(pkt, protocol_encoder.AppendPathSegPointTurn):
                    batch_end = (pkt.x, pkt.y)

            status = self._wait_path(
                packets,
                event_id,
                batch_end,
                deadline,
                position_tol_mm,
                should_interrupt,
            )
            if status == "completed":
                continue
            if status == "interrupted":
                return False
            # Overall deadline expired — report pose proximity to the
            # final goal so the caller can decide to replan.
            dx = robot.pose.position.x - end_x
            dy = robot.pose.position.y - end_y
            return math.hypot(dx, dy) <= position_tol_mm

        return True

    def _wait_path(
        self,
        packets: list,
        event_id: int,
        end: tuple[float, float],
        deadline: float,
        position_tol_mm: float,
        should_interrupt: Callable[[], bool] | None,
    ) -> str:
        """Send *packets* and block until this path batch resolves.

        Returns ``"completed"`` (firmware event or pose proximity),
        ``"interrupted"`` (firmware abort or *should_interrupt*), or
        ``"timeout"`` (the shared overall *deadline* expired).
        """
        robot = self.robot
        state: dict = {"event": None}

        def on_event(_, pkt2: protocol_encoder.PathFollowingEvent) -> None:
            if pkt2.event_id == event_id:
                state["event"] = pkt2.event_type

        handler = robot.add_handler(protocol_encoder.PathFollowingEvent, on_event)
        t0 = time.perf_counter()
        try:
            for pkt in packets:
                robot.send(pkt)

            while time.perf_counter() < deadline:
                ev = state["event"]
                if ev == protocol_encoder.PathEventType.PATH_COMPLETED:
                    return "completed"
                if ev == protocol_encoder.PathEventType.PATH_INTERRUPTED:
                    return "interrupted"
                if should_interrupt is not None and should_interrupt():
                    robot.stop_all_motors()
                    return "interrupted"
                # Pose-polling fallback (COZMO-53). Gated by a 1 s grace
                # period so a path that starts within the tolerance (e.g.
                # a turn-only command) isn't reported done before the
                # firmware has had a chance to run.
                if time.perf_counter() - t0 >= 1.0:
                    dx = robot.pose.position.x - end[0]
                    dy = robot.pose.position.y - end[1]
                    if math.hypot(dx, dy) <= position_tol_mm:
                        return "completed"
                time.sleep(0.1)
            return "timeout"
        finally:
            robot.del_handler(protocol_encoder.PathFollowingEvent, handler)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_event_id(self) -> int:
        """Allocate the next ``ExecutePath`` event id (wraps at 65535)."""
        self._event_id = self._event_id % 65535 + 1
        return self._event_id

    def _settle_robot(self, timeout_s: float = 1.0) -> None:
        """Stop the robot and wait until the wheels are (almost) still.

        The firmware path follower refuses a path whose first segment's
        ``from_x``/``from_y`` does not match the robot's position at
        ``ExecutePath`` time (``PathFollower...PathIsDiscontinuous``).
        That happens whenever a new path is sent while the robot is
        still rolling — e.g. a second navigation click that interrupts
        the first. Stop, wait for measured wheel speeds to reach ~0,
        then let the caller re-read the pose.
        """
        robot = self.robot
        robot.stop_all_motors()
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            if abs(robot.left_wheel_speed.mmps) < 5.0 and abs(robot.right_wheel_speed.mmps) < 5.0:
                return
            time.sleep(0.05)
