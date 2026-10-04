"""Tests for the PathExecutor seam (extracted from ``Client``).

``PathExecutor`` owns the firmware path state machine — settle, event-id
allocation, ``ClearPath`` → ``AppendPathSeg*`` → ``ExecutePath``, the wait
for a ``PathFollowingEvent`` *or* pose proximity, and the chunked polyline
loop — behind a two-method interface and a narrow :class:`PathRobot` port.
That makes all of it testable against a ~40-line fake robot: no
``conn.Connection``, no live Cozmo.

What this covers (none of it was reachable before the extraction):
interruption actually stops the motors, the pose-poll fallback (COZMO-53)
completes a path the firmware never reported, the shared deadline survives
batch boundaries, event ids do not collide, and the path lock keeps two
concurrent callers from interleaving packets into one path buffer.
"""

import threading
import time
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from pycozmo_ng.client import Client
from pycozmo_ng.event import Handler
from pycozmo_ng.navigation.path_executor import PathExecutor
from pycozmo_ng.protocol import protocol_encoder
from pycozmo_ng.util import Angle, Pose, Speed


def pose(x: float, y: float = 0.0, yaw: float = 0.0) -> Pose:
    """``Pose`` with the ``angle_z`` it insists on."""
    return Pose(x, y, 0.0, angle_z=Angle(radians=yaw))


class FakeRobot:
    """A :class:`PathRobot` test double.

    Records every packet, exposes a mutable pose and wheel speeds, and can
    fire ``PathFollowingEvent``s at whatever the executor registered — the
    synchronous equivalent of the robot answering on the wire. ``on_send``
    is a hook called inside :meth:`send`, so a test can complete, move, or
    cancel at the exact moment a packet goes out.
    """

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.pose = Pose(x, y, 0.0, angle_z=Angle(radians=yaw))
        self.left_wheel_speed = Speed(mmps=0.0)
        self.right_wheel_speed = Speed(mmps=0.0)
        self.sent: list = []
        self.stops = 0
        self.on_send: Callable[[Any], None] | None = None
        self._handlers: dict = {}
        # Concurrency probe: how many sends overlapped (the path lock's job
        # is to keep this at 1).
        self._send_lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    # -- PathRobot port -------------------------------------------------

    def send(self, pkt) -> None:
        with self._send_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            self.sent.append(pkt)
            if self.on_send is not None:
                self.on_send(pkt)
        finally:
            with self._send_lock:
                self._active -= 1

    def stop_all_motors(self) -> None:
        self.stops += 1

    def add_handler(self, event, f, one_shot: bool = False) -> Handler:
        handler = Handler(f, one_shot=one_shot)
        self._handlers.setdefault(event, []).append(handler)
        return handler

    def del_handler(self, event, handler: Handler) -> None:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)

    # -- test helpers ---------------------------------------------------

    def fire(self, event_type: protocol_encoder.PathEventType, event_id: int) -> None:
        """Deliver a PathFollowingEvent to the registered waiter(s)."""
        pkt = protocol_encoder.PathFollowingEvent(event_id=event_id, event_type=event_type)
        for handler in list(self._handlers.get(protocol_encoder.PathFollowingEvent, [])):
            handler.f(self, pkt)

    @property
    def execute_ids(self) -> list[int]:
        return [p.event_id for p in self.sent if isinstance(p, protocol_encoder.ExecutePath)]

    @property
    def lines(self) -> list:
        return [p for p in self.sent if isinstance(p, protocol_encoder.AppendPathSegLine)]

    @property
    def turns(self) -> list:
        return [p for p in self.sent if isinstance(p, protocol_encoder.AppendPathSegPointTurn)]

    @property
    def handler_count(self) -> int:
        """Handlers still registered — a waiter that was never removed leaks."""
        return sum(len(h) for h in self._handlers.values())


def complete_on_execute(robot: FakeRobot) -> None:
    """Wire the fake up so every ExecutePath is answered PATH_COMPLETED."""

    def hook(pkt) -> None:
        if isinstance(pkt, protocol_encoder.ExecutePath):
            robot.fire(protocol_encoder.PathEventType.PATH_COMPLETED, pkt.event_id)

    robot.on_send = hook


class TestSettle:
    def test_stops_the_motors_even_when_already_still(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        executor._settle_robot()
        assert robot.stops == 1

    def test_gives_up_after_the_timeout_when_wheels_keep_spinning(self):
        robot = FakeRobot()
        robot.left_wheel_speed = Speed(mmps=200.0)
        executor = PathExecutor(robot)
        t0 = time.perf_counter()
        executor._settle_robot(timeout_s=0.2)
        elapsed = time.perf_counter() - t0
        assert elapsed >= 0.2
        assert robot.stops == 1

    def test_returns_as_soon_as_the_wheels_rest(self):
        robot = FakeRobot()
        robot.left_wheel_speed = Speed(mmps=200.0)
        executor = PathExecutor(robot)

        def stop_spinning() -> None:
            time.sleep(0.15)
            robot.left_wheel_speed = Speed(mmps=0.0)

        thread = threading.Thread(target=stop_spinning, daemon=True)
        thread.start()
        t0 = time.perf_counter()
        executor._settle_robot(timeout_s=5.0)
        elapsed = time.perf_counter() - t0
        thread.join()
        assert elapsed < 1.0


class TestGoToPose:
    def test_builds_the_expected_packet_sequence(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        complete_on_execute(robot)

        ok = executor.go_to_pose(pose(400.0, yaw=1.0))

        assert ok is True
        assert isinstance(robot.sent[0], protocol_encoder.ClearPath)
        assert isinstance(robot.sent[-1], protocol_encoder.ExecutePath)
        assert len(robot.lines) == 1
        assert (robot.lines[0].from_x, robot.lines[0].from_y) == (0.0, 0.0)
        assert (robot.lines[0].to_x, robot.lines[0].to_y) == (400.0, 0.0)
        assert len(robot.turns) == 1
        assert robot.turns[0].angle_rad == 1.0
        assert robot.handler_count == 0

    def test_firmware_interrupt_event_reports_failure(self):
        robot = FakeRobot()

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                robot.fire(protocol_encoder.PathEventType.PATH_INTERRUPTED, pkt.event_id)

        robot.on_send = hook
        executor = PathExecutor(robot)
        assert executor.go_to_pose(pose(400.0)) is False
        assert robot.handler_count == 0

    def test_pose_polling_completes_when_the_firmware_says_nothing(self):
        # COZMO-53: the firmware often omits PathFollowingEvent. The pose
        # reaches the target while the wait loop is still waiting on it.
        robot = FakeRobot()

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                robot.pose = pose(400.0, yaw=0.5)

        robot.on_send = hook
        executor = PathExecutor(robot)
        ok = executor.go_to_pose(pose(400.0, yaw=0.5), timeout_s=5.0)
        assert ok is True
        assert robot.handler_count == 0

    def test_timeout_returns_false_and_logs(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        ok = executor.go_to_pose(pose(5000.0), timeout_s=0.6)
        assert ok is False
        # Still registered at timeout? No — the waiter is always removed.
        assert robot.handler_count == 0

    def test_interrupt_stops_the_motors_and_returns_false(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        cancel = {"armed": False}

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                cancel["armed"] = True

        robot.on_send = hook
        # Settle stops once; the interruption must stop the wheels again.
        ok = executor.go_to_pose(
            pose(400.0),
            timeout_s=5.0,
            should_interrupt=lambda: cancel["armed"],
        )
        assert ok is False
        assert robot.stops >= 2
        assert robot.handler_count == 0

    def test_cancelled_before_the_path_is_sent_sends_nothing(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        ok = executor.go_to_pose(pose(400.0), should_interrupt=lambda: True)
        assert ok is False
        assert robot.sent == []
        assert robot.handler_count == 0

    def test_relative_to_robot_offsets_the_target(self):
        robot = FakeRobot(x=100.0, y=0.0, yaw=0.0)
        executor = PathExecutor(robot)
        complete_on_execute(robot)
        executor.go_to_pose(pose(50.0, 10.0), relative_to_robot=True)
        assert (robot.lines[0].to_x, robot.lines[0].to_y) == (150.0, 10.0)

    def test_event_ids_never_repeat_back_to_back(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        complete_on_execute(robot)
        executor.go_to_pose(pose(400.0))
        executor.go_to_pose(pose(400.0))
        assert len(robot.execute_ids) == 2
        assert robot.execute_ids[0] != robot.execute_ids[1]
        assert all(1 <= i <= 65535 for i in robot.execute_ids)


class TestExecutePolyline:
    def test_single_batch_ends_with_execute_path(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        complete_on_execute(robot)
        ok = executor.execute_polyline([(100.0, 0.0), (200.0, 0.0), (300.0, 0.0)])
        assert ok is True
        assert len(robot.execute_ids) == 1
        assert [p.to_x for p in robot.lines] == [100.0, 200.0, 300.0]
        assert robot.handler_count == 0

    def test_long_path_is_batched_and_the_next_batch_starts_from_the_pose(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        # 7 points → 2 chunks (buffer holds 6 segments).
        points = [(float(i * 50), 0.0) for i in range(1, 8)]

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                # The robot drove the batch it was just sent.
                robot.pose = pose(200.0)
                robot.fire(protocol_encoder.PathEventType.PATH_COMPLETED, pkt.event_id)

        robot.on_send = hook
        assert executor.execute_polyline(points) is True
        assert len(robot.execute_ids) == 2
        assert robot.execute_ids[0] != robot.execute_ids[1]
        # 7 points split as [1 point] + [6 points]: the first batch is the
        # odd point out, the second fills the firmware buffer.
        assert len(robot.lines) == 7
        # First batch starts at the original pose…
        assert robot.lines[0].from_x == 0.0
        # …and the second rebuilds its first segment from where the robot
        # actually ended up, not from the end of the previous batch.
        assert robot.lines[1].from_x == 200.0
        assert robot.handler_count == 0

    def test_final_turn_is_only_on_the_last_batch(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        points = [(float(i * 50), 0.0) for i in range(1, 8)]
        complete_on_execute(robot)
        executor.execute_polyline(points, final_heading_rad=0.7)
        assert len(robot.execute_ids) == 2
        assert len(robot.turns) == 1
        assert robot.turns[0].angle_rad == 0.7

    def test_degenerate_points_with_a_heading_still_turns(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        complete_on_execute(robot)
        # Within 10 mm of the start — no line segments, but the turn runs.
        assert executor.execute_polyline([(5.0, 0.0)], final_heading_rad=0.5) is True
        assert robot.lines == []
        assert len(robot.turns) == 1
        assert len(robot.execute_ids) == 1

    def test_nothing_to_drive_returns_true_without_sending(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        assert executor.execute_polyline([(5.0, 0.0)]) is True
        assert robot.sent == []
        assert robot.stops == 1  # settle still ran

    def test_interrupt_between_batches_stops_the_motors(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        cancel = {"armed": False}

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                cancel["armed"] = True
                robot.fire(protocol_encoder.PathEventType.PATH_COMPLETED, pkt.event_id)

        robot.on_send = hook
        points = [(float(i * 50), 0.0) for i in range(1, 8)]
        ok = executor.execute_polyline(points, should_interrupt=lambda: cancel["armed"])
        assert ok is False
        # Settled once, then stopped again on the interrupt — and the second
        # batch was never sent.
        assert robot.stops >= 2
        assert len(robot.execute_ids) == 1
        assert robot.handler_count == 0

    def test_firmware_interrupt_event_fails_the_run(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                robot.fire(protocol_encoder.PathEventType.PATH_INTERRUPTED, pkt.event_id)

        robot.on_send = hook
        assert executor.execute_polyline([(400.0, 0.0)]) is False
        assert robot.handler_count == 0

    def test_deadline_expiry_reports_pose_proximity_to_the_goal(self):
        # The shared deadline expired — the caller is told whether it is
        # close enough to the final goal to stop replanning.
        robot = FakeRobot()
        executor = PathExecutor(robot)

        def reach_goal(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                robot.pose = pose(400.0)

        robot.on_send = reach_goal
        assert executor.execute_polyline([(400.0, 0.0)], timeout_s=0.3) is True

        far = FakeRobot()
        executor_far = PathExecutor(far)
        assert executor_far.execute_polyline([(4000.0, 0.0)], timeout_s=0.3) is False

    def test_waypoint_tolerated_by_the_firmware_gap_filter(self):
        robot = FakeRobot()
        executor = PathExecutor(robot)
        complete_on_execute(robot)
        # Second point is < 10 mm past the first — dropped from the path.
        assert executor.execute_polyline([(100.0, 0.0), (105.0, 0.0), (400.0, 0.0)])
        assert [p.to_x for p in robot.lines] == [100.0, 400.0]


class TestPathLock:
    def test_concurrent_callers_never_interleave_packets(self):
        """The firmware buffer holds one path — the lock keeps two callers out."""
        robot = FakeRobot()

        def hook(pkt) -> None:
            if isinstance(pkt, protocol_encoder.ExecutePath):
                time.sleep(0.05)  # widen the window for a would-be race
                robot.fire(protocol_encoder.PathEventType.PATH_COMPLETED, pkt.event_id)

        robot.on_send = hook
        executor = PathExecutor(robot)

        results: list[bool] = []

        def run() -> None:
            results.append(executor.go_to_pose(pose(400.0)))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results == [True, True]
        assert robot.max_active == 1
        assert len(robot.execute_ids) == 2
        assert robot.handler_count == 0


class TestClientDelegation:
    """``Client`` keeps the public path API — the state machine lives behind it."""

    @staticmethod
    def _client() -> Client:
        # Build without __init__: no connection, no TTS, nothing but the seam.
        cli = Client.__new__(Client)
        cli.paths = MagicMock()
        return cli

    def test_go_to_pose_forwards_every_argument(self):
        cli = self._client()
        target = pose(400.0)
        cli.go_to_pose(
            target,
            relative_to_robot=True,
            timeout_s=3.0,
            position_tol_mm=7.0,
            angle_tol_rad=0.2,
            should_interrupt=lambda: False,
        )
        args, kwargs = cli.paths.go_to_pose.call_args
        assert args == (target,)
        assert kwargs["relative_to_robot"] is True
        assert kwargs["timeout_s"] == 3.0
        assert kwargs["position_tol_mm"] == 7.0
        assert kwargs["angle_tol_rad"] == 0.2
        assert kwargs["should_interrupt"]() is False

    def test_execute_polyline_forwards_every_argument(self):
        cli = self._client()
        points = [(100.0, 0.0), (200.0, 0.0)]
        cli.execute_polyline(
            points,
            speed_mmps=55.0,
            accel_mmps2=30.0,
            decel_mmps2=25.0,
            final_heading_rad=0.7,
            timeout_s=12.0,
            position_tol_mm=9.0,
            should_interrupt=lambda: True,
        )
        args, kwargs = cli.paths.execute_polyline.call_args
        assert args == (points,)
        assert kwargs["speed_mmps"] == 55.0
        assert kwargs["accel_mmps2"] == 30.0
        assert kwargs["decel_mmps2"] == 25.0
        assert kwargs["final_heading_rad"] == 0.7
        assert kwargs["timeout_s"] == 12.0
        assert kwargs["position_tol_mm"] == 9.0
        assert kwargs["should_interrupt"]() is True
