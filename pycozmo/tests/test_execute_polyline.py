"""Tests for firmware polyline path execution and the A* follower.

``build_polyline_packets`` is pure (packet building), and the Navigator A*
follower is exercised with a mock CLI whose ``execute_polyline`` side
effect simulates arrival / interruption — no live robot needed.

Motivation: the Python-side closed loop (``drive_to_point``) was unreliable
because wheel commands and pose updates travel over WiFi; the firmware path
follower runs the trajectory at high rate on the robot.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from pycozmo.client import build_polyline_packets, chunk_polyline_points
from pycozmo.navigation.astar import AStarPlanner, PathPoint
from pycozmo.navigation.dock_state import DockState
from pycozmo.navigation.navigator import Navigator
from pycozmo.protocol import protocol_encoder
from pycozmo.vision.occupancy_map import FREE, OCCUPIED, OccupancyMap


class TestBuildPolylinePackets:
    def test_basic_sequence(self):
        pkts = build_polyline_packets(
            [(100.0, 0.0), (100.0, 100.0)], start=(0.0, 0.0), event_id=7
        )
        assert isinstance(pkts[0], protocol_encoder.ClearPath)
        assert isinstance(pkts[-1], protocol_encoder.ExecutePath)
        assert pkts[-1].event_id == 7
        lines = [p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegLine)]
        assert len(lines) == 2
        # Segments chain: start -> wp1 -> wp2.
        assert (lines[0].from_x, lines[0].from_y) == (0.0, 0.0)
        assert (lines[0].to_x, lines[0].to_y) == (100.0, 0.0)
        assert (lines[1].from_x, lines[1].from_y) == (100.0, 0.0)
        assert (lines[1].to_x, lines[1].to_y) == (100.0, 100.0)

    def test_degenerate_points_dropped(self):
        # First point is within 10 mm of the start — skipped; the line
        # starts from the robot and goes straight to the real waypoint.
        pkts = build_polyline_packets(
            [(5.0, 0.0), (400.0, 0.0)], start=(0.0, 0.0), event_id=1
        )
        lines = [p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegLine)]
        assert len(lines) == 1
        assert (lines[0].from_x, lines[0].to_x) == (0.0, 400.0)

    def test_final_turn_appended(self):
        pkts = build_polyline_packets(
            [(400.0, 0.0)],
            start=(0.0, 0.0),
            event_id=1,
            final_heading_rad=1.5,
        )
        turns = [
            p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegPointTurn)
        ]
        assert len(turns) == 1
        assert (turns[0].x, turns[0].y) == (400.0, 0.0)
        assert turns[0].angle_rad == 1.5

    def test_nothing_to_do_returns_empty(self):
        assert build_polyline_packets([], start=(0.0, 0.0), event_id=1) == []
        # Degenerate point and no final heading -> nothing to execute.
        assert build_polyline_packets([(1.0, 0.0)], start=(0.0, 0.0), event_id=1) == []

    def test_turn_only_when_close_but_heading_requested(self):
        pkts = build_polyline_packets(
            [(5.0, 0.0)],
            start=(0.0, 0.0),
            event_id=3,
            final_heading_rad=0.5,
        )
        lines = [p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegLine)]
        turns = [
            p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegPointTurn)
        ]
        assert lines == []
        assert len(turns) == 1
        assert (turns[0].x, turns[0].y) == (0.0, 0.0)  # in place, at start
        assert isinstance(pkts[-1], protocol_encoder.ExecutePath)

    def test_speed_propagates_to_segments(self):
        pkts = build_polyline_packets(
            [(400.0, 0.0)],
            start=(0.0, 0.0),
            event_id=1,
            speed_mmps=55.0,
            accel_mmps2=30.0,
            decel_mmps2=25.0,
        )
        line = next(
            p for p in pkts if isinstance(p, protocol_encoder.AppendPathSegLine)
        )
        assert line.speed_mmps == 55.0
        assert line.accel_mmps2 == 30.0
        assert line.decel_mmps2 == 25.0


class TestChunkPolylinePoints:
    """Firmware buffer holds MAX_NUM_PATH_SEGMENTS (6) appends per path."""

    def test_empty(self):
        assert chunk_polyline_points([]) == []

    def test_short_path_single_chunk(self):
        pts = [(100.0, 0.0), (200.0, 0.0)]
        assert chunk_polyline_points(pts) == [pts]

    def test_six_points_fit_in_one_chunk(self):
        pts = [(float(i * 50), 0.0) for i in range(1, 7)]
        assert chunk_polyline_points(pts) == [pts]

    def test_seven_points_split_and_order_preserved(self):
        pts = [(float(i * 50), 0.0) for i in range(1, 8)]
        chunks = chunk_polyline_points(pts)
        assert [len(c) for c in chunks] == [1, 6]
        assert [p for c in chunks for p in c] == pts

    def test_final_turn_reserves_segment(self):
        pts = [(float(i * 50), 0.0) for i in range(1, 12)]  # 11 points
        chunks = chunk_polyline_points(pts, final_heading_rad=0.5)
        assert len(chunks) == 2
        for i, chunk in enumerate(chunks):
            segs = len(chunk) + (1 if i == len(chunks) - 1 else 0)
            assert segs <= 6
        assert [p for c in chunks for p in c] == pts

    def test_five_points_plus_turn_fit_in_one_chunk(self):
        pts = [(float(i * 50), 0.0) for i in range(1, 6)]
        chunks = chunk_polyline_points(pts, final_heading_rad=0.2)
        assert len(chunks) == 1
        assert len(chunks[0]) + 1 <= 6

    def test_large_path_all_chunks_within_limit(self):
        pts = [(float(i * 50), float(i % 3 * 40.0)) for i in range(1, 32)]
        for heading in (None, 1.0):
            chunks = chunk_polyline_points(pts, final_heading_rad=heading)
            assert chunks
            for i, chunk in enumerate(chunks):
                is_last = i == len(chunks) - 1
                segs = len(chunk) + (1 if heading is not None and is_last else 0)
                assert 1 <= segs <= 6
            assert [p for c in chunks for p in c] == pts


def _free_map(size_mm: float = 1000.0, cell_mm: float = 50.0) -> OccupancyMap:
    occ = OccupancyMap(grid_size_mm=size_mm, cell_size_mm=cell_mm)
    occ._grid[:] = FREE
    return occ


def _make_nav(x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    """Navigator with a mock CLI (controllable pose) and free occupancy map."""
    nav = Navigator(DockState(), world_map=MagicMock())
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=yaw)),
    )
    cli = MagicMock()
    type(cli).pose = property(lambda self: pose)
    nav.cli = cli
    nav.astar_planner = AStarPlanner(_free_map(), allow_unknown=False)
    return nav, cli, pose


class TestFollowAstarPath:
    def test_follows_path_via_firmware(self):
        nav, cli, pose = _make_nav()

        def arrive(*args, **kwargs):
            del args, kwargs
            pose.position.x, pose.position.y = 400.0, 0.0
            return True

        cli.execute_polyline.side_effect = arrive
        nav._follow_astar_path(400.0, 0.0)

        assert cli.execute_polyline.call_count == 1
        args, kwargs = cli.execute_polyline.call_args
        pts = args[0]
        assert (400.0, 0.0) in pts  # the exact goal is on the path
        assert "should_interrupt" in kwargs
        assert "final_heading_rad" in kwargs
        # Path was published to the world map, then cleared on completion.
        wm = nav.world_map
        assert wm.set_astar_path.called
        assert wm.clear_astar_path.called
        assert nav._path_following is False

    def test_no_path_falls_back_to_go_to_pose(self):
        nav, cli, _pose = _make_nav()
        # Occupy the goal cell so find_path fails (goal not traversable).
        col, row = nav.astar_planner._world_to_grid(400.0, 0.0)
        nav.astar_planner._grid[row, col] = OCCUPIED

        nav._follow_astar_path(400.0, 0.0)

        cli.go_to_pose.assert_called_once()
        cli.execute_polyline.assert_not_called()
        assert nav._path_following is False

    def test_no_planner_falls_back(self):
        nav, cli, _pose = _make_nav()
        nav.astar_planner = None
        nav._follow_astar_path(400.0, 0.0)
        cli.go_to_pose.assert_called_once()
        cli.execute_polyline.assert_not_called()

    def test_cancel_stops_without_replan(self):
        nav, cli, _pose = _make_nav()

        def cancel(*args, **kwargs):
            del args, kwargs
            nav._path_following = False  # simulate navigate_cancel
            return False

        cli.execute_polyline.side_effect = cancel
        nav._follow_astar_path(400.0, 0.0)
        # Interrupted once, then the loop saw _path_following == False.
        assert cli.execute_polyline.call_count == 1
        assert nav.world_map.clear_astar_path.called

    def test_replans_when_execute_falls_short(self):
        nav, cli, pose = _make_nav()
        calls = {"n": 0}

        def partial(*args, **kwargs):
            del args, kwargs
            calls["n"] += 1
            if calls["n"] == 1:
                pose.position.x = 200.0  # fell short, still en route
                return False
            pose.position.x, pose.position.y = 400.0, 0.0
            return True

        cli.execute_polyline.side_effect = partial
        nav._follow_astar_path(400.0, 0.0)
        assert calls["n"] == 2
        assert nav.world_map.set_astar_path.call_count == 2


class TestPathShouldInterrupt:
    def test_true_when_not_following(self):
        nav, _cli, _pose = _make_nav()
        nav._path_following = False
        assert nav._path_should_interrupt() is True

    def test_false_when_path_clear(self):
        nav, _cli, _pose = _make_nav()
        nav._path_following = True
        nav._path_waypoints = [PathPoint(400.0, 0.0)]
        assert nav._path_should_interrupt() is False

    def test_true_when_waypoint_becomes_occupied(self):
        nav, _cli, _pose = _make_nav()
        nav._path_following = True
        nav._path_waypoints = [PathPoint(400.0, 0.0)]
        col, row = nav.astar_planner._world_to_grid(400.0, 0.0)
        nav.astar_planner._grid[row, col] = OCCUPIED
        assert nav._path_should_interrupt() is True

    def test_skips_waypoints_near_robot(self):
        nav, _cli, _pose = _make_nav(x=380.0, y=0.0)
        nav._path_following = True
        # Occupied but within ASTAR_MONITOR_SKIP_MM of the robot — trusted.
        nav._path_waypoints = [PathPoint(400.0, 0.0)]
        col, row = nav.astar_planner._world_to_grid(400.0, 0.0)
        nav.astar_planner._grid[row, col] = OCCUPIED
        assert nav._path_should_interrupt() is False
