"""Tests for the origin-reset + marker-snap race (COZMO-72).

When the robot docks on the charger two callers fire ``_reset_pose_origin``
at the same time:

* the ``on_charging_change`` event thread (contacts detect), and
* ``_on_aruco_docked`` (the ArucoDocking job that just finished its reverse).

The bug: ``_reset_pose_origin`` slept a fixed 0.5 s and then read
``cli.pose.origin_id`` -- if the origin-change telemetry hadn't arrived yet,
``origin_reset_origin_id`` was set to the *stale* id, and
``_snap_marker_dock_pose`` (which runs immediately after) ran before the
world map's ``update_robot_pose`` had cleared on the origin change -- so the
marker it just set was wiped a moment later by the late telemetry.

The fix: ``_reset_pose_origin`` now polls for the new origin id in telemetry
instead of a blind sleep, serialises the send + confirmation behind
``_origin_reset_lock`` so the two callers never both send SetOrigin, and
returns a bool so callers can skip the marker snap when the origin was not
confirmed (e.g. telemetry lost on Cozmo's lossy Wi-Fi).

Follow-up bug (still COZMO-72): on poll timeout the old code set
``origin_reset_origin_id = cli.pose.origin_id`` *before* the echo arrived --
i.e. to the stale id -- which permanently poisoned the guard and made every
subsequent charger event skip the origin reset.  The fix leaves the guard
untouched on timeout so the next event retries, and only snaps the marker
when the reset was confirmed.
"""

import math
import threading
import time
from unittest.mock import MagicMock

from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.navigator import Navigator
from pycozmo_ng.navigation.world_map import WorldMap
from pycozmo_ng.util import Angle, Pose


class _FakeCli:
    """Mock CLI whose ``pose.origin_id`` advances only when telemetry
    echoes back -- never immediately on ``send``.

    ``delay_s`` controls when the echo arrives (default 0 = immediate),
    simulating the telemetry round-trip that the real robot incurs after
    a SetOrigin packet.  This is exactly the window in which the old code
    used to read a stale origin id.
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, origin_id=1, delay_s=0.0):
        self._x = x
        self._y = y
        self._yaw = yaw
        self._origin_id = origin_id
        self._pending_origin_id: int | None = None
        self.pose_frame_id = 1
        self.delay_s = delay_s
        self.set_origin_count = 0
        self._conn = MagicMock()
        self._conn.send.side_effect = self._on_send

    @property
    def pose(self) -> Pose:
        return Pose(
            self._x,
            self._y,
            0.0,
            angle_z=Angle(radians=self._yaw),
            origin_id=self._origin_id,
            is_accurate=True,
        )

    @property
    def conn(self):
        return self._conn

    def _on_send(self, pkt) -> None:
        """Called by conn.send() -- mimics the firmware accepting SetOrigin."""
        self.set_origin_count += 1
        self._pending_origin_id = pkt.pose_origin_id
        if self.delay_s > 0:
            threading.Timer(self.delay_s, self._apply_telemetry).start()
        else:
            self._apply_telemetry()

    def _apply_telemetry(self) -> None:
        """Simulate the RobotState telemetry carrying the new origin id."""
        if self._pending_origin_id is not None:
            self._origin_id = self._pending_origin_id
            self._pending_origin_id = None
            # After SetOrigin the robot reports itself at (0, 0).
            self._x = 0.0
            self._y = 0.0


def _make_nav(
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    origin_id: int = 1,
    delay_s: float = 0.0,
    world_map=None,
) -> tuple[Navigator, _FakeCli]:
    ds = DockState()
    nav = Navigator(dock_state=ds, world_map=world_map)
    cli = _FakeCli(x=x, y=y, yaw=yaw, origin_id=origin_id, delay_s=delay_s)
    nav.cli = cli
    return nav, cli


# -------------------------------------------------------------------
# _reset_pose_origin
# -------------------------------------------------------------------


class TestResetPoseOrigin:
    def test_sends_set_origin_and_updates_tracking(self):
        """Basic path: SetOrigin is sent and origin_reset_origin_id is
        updated to the *new* (telemetry-confirmed) origin id."""
        nav, cli = _make_nav(origin_id=5)
        nav.ds.origin_reset_origin_id = None  # not yet reset for this origin

        nav._reset_pose_origin()

        assert cli.set_origin_count == 1
        assert nav.ds.origin_reset_origin_id == 6  # confirmed, not stale 5
        assert nav.ds.dock_pose is not None

    def test_returns_true_on_success(self):
        """Successful reset returns True (confirmed via telemetry)."""
        nav, _cli = _make_nav(origin_id=5)
        nav.ds.origin_reset_origin_id = None
        result = nav._reset_pose_origin()
        assert result is True

    def test_guard_skips_already_reset_origin(self):
        """A second reset for the same origin id sends no SetOrigin and
        returns True when the robot is still at (0, 0)."""
        nav, cli = _make_nav(origin_id=5)
        nav.ds.origin_reset_origin_id = 5  # already reset for this origin

        result = nav._reset_pose_origin()

        assert result is True
        assert cli.set_origin_count == 0

    def test_guard_resets_when_robot_drove_away_and_came_back(self):
        """The 'resets only once' bug: after the first charger reset the
        robot drives away, walks around, and returns.  The origin_id hasn't
        changed (no pick-up to bump it via _on_robot_picked_up), but the
        position is no longer (0, 0).  The guard must detect this and
        re-anchor the origin -- not silently skip because the origin_id
        matches.
        """
        nav, cli = _make_nav(x=100.0, y=0.0, origin_id=5, delay_s=0.1)
        nav.ds.origin_reset_origin_id = 5  # already reset for this origin

        result = nav._reset_pose_origin()

        assert result is True
        assert cli.set_origin_count == 1, "guard skipped — origin never re-anchored"
        assert nav.ds.origin_reset_origin_id == 6  # new origin confirmed

    def test_double_sender_only_one_set_origin(self):
        """Two callers racing on the same origin id must not double-send.

        This is the core of the race: on_charging_change and
        _on_aruco_docked both call _reset_pose_origin when the contacts
        detect during a dock_on_aruco sequence.
        """
        nav, cli = _make_nav(origin_id=5)
        nav.ds.origin_reset_origin_id = None

        barrier = threading.Barrier(2)
        results: list[bool] = []

        def call():
            barrier.wait()
            try:
                nav._reset_pose_origin()
                results.append(True)
            except Exception:  # noqa: BLE001
                results.append(False)

        t1 = threading.Thread(target=call)
        t2 = threading.Thread(target=call)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert cli.set_origin_count == 1, "double SetOrigin: origin re-anchored twice, map cleared twice"
        assert all(results)

    def test_polls_for_telemetry_not_fixed_sleep(self):
        """_reset_pose_origin must wait for origin_id to echo in telemetry,
        not just sleep 0.5 s. With a 1 s telemetry delay the old 0.5 s
        sleep would have read a stale origin id and left
        origin_reset_origin_id at the *old* value."""
        nav, cli = _make_nav(origin_id=5, delay_s=1.0)
        nav.ds.origin_reset_origin_id = None

        t0 = time.monotonic()
        nav._reset_pose_origin()
        elapsed = time.monotonic() - t0

        # Waited for the ~1 s telemetry delay, not just 0.5 s.
        assert elapsed >= 0.9, f"returned too early ({elapsed:.2f}s)"
        assert cli.set_origin_count == 1
        assert nav.ds.origin_reset_origin_id == 6  # confirmed via telemetry


# -------------------------------------------------------------------
# _reset_pose_origin -- timeout handling (COZMO-72 follow-up)
# -------------------------------------------------------------------


class TestResetTimeout:
    """When telemetry doesn't echo within the poll window (Cozmo's Wi-Fi
    drops the packet), the old code set origin_reset_origin_id to the stale
    id and permanently poisoned the guard.  These tests verify the fix."""

    def test_timeout_returns_false(self):
        """Poll timeout returns False so callers know the reset was not
        confirmed."""
        nav, cli = _make_nav(origin_id=5, delay_s=5.0)  # never echoes
        nav.ds.origin_reset_origin_id = None

        t0 = time.monotonic()
        result = nav._reset_pose_origin()
        elapsed = time.monotonic() - t0

        assert result is False
        assert elapsed < 3.0, f"poll should have timed out at ~2s, took {elapsed:.2f}s"
        assert cli.set_origin_count == 1

    def test_timeout_leaves_guard_untouched(self):
        """On timeout, origin_reset_origin_id must NOT be set to the stale
        origin id -- otherwise the guard permanently blocks future resets.

        This is the core of the follow-up: the old code did
        ``ds.origin_reset_origin_id = cli.pose.origin_id`` unconditionally,
        which after a timeout set it to the *old* id (the echo never arrived),
        making the guard return True forever.
        """
        nav, _cli = _make_nav(origin_id=5, delay_s=5.0)
        nav.ds.origin_reset_origin_id = None

        nav._reset_pose_origin()  # times out

        # Guard must NOT be poisoned: origin_reset_origin_id stays None,
        # so the next call will see None != 5 and send SetOrigin again.
        assert nav.ds.origin_reset_origin_id is None

    def test_timeout_then_retry_sends_set_origin(self):
        """After a timeout, a subsequent _reset_pose_origin call must send
        SetOrigin again (the guard did not permanently skip it)."""
        nav, cli = _make_nav(origin_id=5, delay_s=5.0)
        nav.ds.origin_reset_origin_id = None

        # First call times out.
        nav._reset_pose_origin()
        assert nav.ds.origin_reset_origin_id is None

        # Second call -- origin_id hasn't changed, guard should NOT skip.
        nav._reset_pose_origin()
        assert cli.set_origin_count == 2, "guard poisoned by timeout -- second SetOrigin never sent"


# -------------------------------------------------------------------
# _snap_marker_dock_pose -- survives the map clear
# -------------------------------------------------------------------


class TestSnapMarkerAfterReset:
    def test_marker_set_after_telemetry_arrives(self):
        """_snap_marker_dock_pose runs only after _reset_pose_origin has
        confirmed the origin-id echo in telemetry.

        On the real robot the telemetry echo triggers
        Brain._on_robot_state -> update_robot_pose, which clears the
        world map (including _aruco) on the origin change.  If the
        marker were set before that clear it would be wiped.  We verify
        the ordering by recording events in a shared list.
        """
        events: list[str] = []

        # delay_s must exceed the old 0.5 s fixed sleep to expose the
        # stale-telemetry race: with delay > 0.5 s the old code returned
        # before the echo arrived, then _snap_marker_dock_pose ran on a
        # not-yet-cleared map -- the marker was wiped later.
        nav, cli = _make_nav(yaw=0.0, origin_id=5, delay_s=0.7)
        nav.ds.origin_reset_origin_id = None

        # Wrap the telemetry echo to record the clear.
        original_apply = cli._apply_telemetry

        def tracked_telemetry():
            original_apply()
            events.append("telemetry")

        cli._apply_telemetry = tracked_telemetry

        world_map = MagicMock(spec=WorldMap)

        def record_set_aruco(**kw):
            events.append("set_aruco")

        world_map.set_aruco_world_pose = MagicMock(side_effect=record_set_aruco)
        nav.world_map = world_map

        nav._on_aruco_docked(0)

        assert "telemetry" in events, "telemetry echo never arrived"
        assert "set_aruco" in events, "marker was never snapped"
        assert events.index("telemetry") < events.index("set_aruco"), (
            "marker set BEFORE the origin-change telemetry -- it will be wiped by the delayed map clear (COZMO-72)"
        )

    def test_snap_marker_position_on_the_normal(self):
        """The snapped marker sits 65 mm behind the camera (COZMO-68)."""
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(yaw=0.0, origin_id=1, world_map=world_map)
        nav.ds.origin_reset_origin_id = None

        nav._on_aruco_docked(0)

        world_map.set_aruco_world_pose.assert_called_once()
        kw = world_map.set_aruco_world_pose.call_args.kwargs
        # Robot at origin, yaw 0 -> marker 65 mm behind on -x.
        pos = kw["position_world"]
        assert math.isclose(pos[0], -65.0, abs_tol=0.01)
        assert math.isclose(pos[1], 0.0, abs_tol=0.01)

    def test_snap_marker_yaw_rotation(self):
        """Marker position rotates with the robot's yaw."""
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(yaw=math.radians(90.0), origin_id=1, world_map=world_map)
        nav.ds.origin_reset_origin_id = None

        nav._on_aruco_docked(0)

        kw = world_map.set_aruco_world_pose.call_args.kwargs
        pos = kw["position_world"]
        # yaw=90 deg: marker is at -65*sin(90) = -65 on y, ~0 on x.
        assert math.isclose(pos[0], 0.0, abs_tol=0.01)
        assert math.isclose(pos[1], -65.0, abs_tol=0.01)

    def test_on_aruco_docked_resets_origin_then_snaps(self):
        """_on_aruco_docked calls _reset_pose_origin before snapping."""
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(yaw=math.radians(30.0), origin_id=2, world_map=world_map)
        nav.ds.origin_reset_origin_id = None
        nav._reset_pose_origin = MagicMock(return_value=True)

        nav._on_aruco_docked(0)

        nav._reset_pose_origin.assert_called_once()
        world_map.set_aruco_world_pose.assert_called_once()

    def test_snap_skipped_when_origin_reset_not_confirmed(self):
        """When _reset_pose_origin times out (returns False),
        _snap_marker_dock_pose must NOT run -- the robot's world pose is
        not at (0,0,0) so the marker would be snapped to the wrong spot."""
        nav, _cli = _make_nav(yaw=0.0, origin_id=5, delay_s=5.0)
        nav.ds.origin_reset_origin_id = None

        world_map = MagicMock(spec=WorldMap)
        nav.world_map = world_map

        nav._on_aruco_docked(0)

        world_map.set_aruco_world_pose.assert_not_called()
        assert nav.ds.origin_reset_origin_id is None

    def test_snap_marker_no_world_map_or_cli_is_safe(self):
        """Guard: no-op when world_map or cli is missing."""
        nav = Navigator(DockState(), world_map=None)
        nav._snap_marker_dock_pose(0)  # must not raise

        nav2, _ = _make_nav(origin_id=1)
        nav2.world_map = None
        nav2._snap_marker_dock_pose(0)  # must not raise


# -------------------------------------------------------------------
# Charger events -- reset + marker 0 snap
# -------------------------------------------------------------------


class TestChargerEventAnchor:
    """The charger events must anchor the map the same way an ArUco dock
    does: reset the origin *and* snap marker 0 back to its dock pose (65 mm
    behind the camera).  Before this, the events only reset the origin --
    the origin-change map clear wiped marker 0 and nothing put it back."""

    @staticmethod
    def _snap_event(world_map) -> threading.Event:
        done = threading.Event()
        world_map.set_aruco_world_pose = MagicMock(side_effect=lambda **kw: done.set())
        return done

    def test_charging_event_resets_and_snaps_marker_0(self):
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(yaw=0.0, origin_id=1, world_map=world_map)
        nav.ds.origin_reset_origin_id = None
        done = self._snap_event(world_map)

        nav.on_charging_change(None, True)

        assert done.wait(timeout=5.0), "charging event never snapped marker 0"
        kw = world_map.set_aruco_world_pose.call_args.kwargs
        assert kw["marker_id"] == 0
        # Robot docked at origin, yaw 0 -> marker 65 mm behind on -x.
        assert math.isclose(kw["position_world"][0], -65.0, abs_tol=0.01)
        assert math.isclose(kw["position_world"][1], 0.0, abs_tol=0.01)

    def test_on_charger_event_resets_and_snaps_marker_0(self):
        """The on-charger event (contacts touched) anchors too — charging
        may never start when the base is unpowered."""
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(yaw=0.0, origin_id=1, world_map=world_map)
        nav.ds.origin_reset_origin_id = None
        done = self._snap_event(world_map)

        nav.on_charger_change(None, True)

        assert done.wait(timeout=5.0), "on-charger event never snapped marker 0"
        kw = world_map.set_aruco_world_pose.call_args.kwargs
        assert kw["marker_id"] == 0
        assert math.isclose(kw["position_world"][0], -65.0, abs_tol=0.01)

    def test_charging_event_snaps_after_map_clear(self):
        """Like the ArUco path: the snap must land after the origin-change
        telemetry cleared the map, or it is wiped again (COZMO-72)."""
        events: list[str] = []

        nav, cli = _make_nav(yaw=0.0, origin_id=5, delay_s=0.7)
        nav.ds.origin_reset_origin_id = None

        original_apply = cli._apply_telemetry

        def tracked_telemetry():
            original_apply()
            events.append("telemetry")

        cli._apply_telemetry = tracked_telemetry

        world_map = MagicMock(spec=WorldMap)
        world_map.set_aruco_world_pose = MagicMock(side_effect=lambda **kw: events.append("set_aruco"))
        nav.world_map = world_map

        nav.on_charging_change(None, True)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and "set_aruco" not in events:
            time.sleep(0.05)

        assert "telemetry" in events, "telemetry echo never arrived"
        assert "set_aruco" in events, "marker was never snapped"
        assert events.index("telemetry") < events.index("set_aruco"), (
            "marker set BEFORE the origin-change telemetry — wiped by the delayed map clear (COZMO-72)"
        )

    def test_snap_skipped_when_reset_not_confirmed(self):
        """No telemetry confirmation -> no snap: the robot's world pose is
        not (0, 0, 0), so marker 0 would land at a wrong spot."""
        world_map = MagicMock(spec=WorldMap)
        nav, _cli = _make_nav(world_map=world_map)
        nav._reset_pose_origin = MagicMock(return_value=False)

        nav._anchor_on_charger()

        world_map.set_aruco_world_pose.assert_not_called()


# -------------------------------------------------------------------
# Concurrent reset + snap
# -------------------------------------------------------------------


class TestDockingResetIntegration:
    """Verify _on_aruco_docked + concurrent _reset_pose_origin produces one
    origin reset and a surviving marker (COZMO-72)."""

    def test_concurrent_reset_and_snap(self):
        events: list[str] = []

        nav, cli = _make_nav(yaw=0.0, origin_id=5, delay_s=0.7)
        nav.ds.origin_reset_origin_id = None

        original_apply = cli._apply_telemetry

        def tracked_telemetry():
            original_apply()
            events.append("telemetry")

        cli._apply_telemetry = tracked_telemetry

        world_map = MagicMock(spec=WorldMap)
        world_map.set_aruco_world_pose = MagicMock(side_effect=lambda **kw: events.append("set_aruco"))
        nav.world_map = world_map

        def from_charging_event():
            nav._reset_pose_origin()

        def from_dock_sequence():
            nav._on_aruco_docked(0)

        t1 = threading.Thread(target=from_charging_event)
        t2 = threading.Thread(target=from_dock_sequence)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert cli.set_origin_count == 1, "should send SetOrigin exactly once"
        assert "set_aruco" in events, "marker must have been snapped"
        assert events.index("telemetry") < events.index("set_aruco"), "marker wiped by delayed map clear (COZMO-72)"
