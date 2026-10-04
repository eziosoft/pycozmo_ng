"""Tests for CubeApproach — two-step standoff approach and grab alignment.

Tests the extracted class (COZMO-40) with mock CLI and callbacks, no live
hardware needed.
"""

import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from pycozmo_ng.navigation.cube_approach import CubeApproach
from pycozmo_ng.navigation.motion_controller import MotionController

# Shared controller so spawned approaches are visible to assertions.
_MOTION = MotionController()


def _make_cube_state(
    x: float = 0.0,
    y: float = 300.0,
    confidence: float = 0.95,
    updated_at: float | None = None,
):
    """Build a minimal CubeState-like object."""
    if updated_at is None:
        updated_at = time.monotonic()
    return SimpleNamespace(
        position=np.array([x, y, 22.0]),
        rotation_matrix=np.eye(3),
        confidence=confidence,
        updated_at=updated_at,
    )


def _make_mock_cli(x: float = 0.0, y: float = 0.0, yaw: float = 0.0):
    """Build a mock CLI with a controllable pose."""
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=yaw)),
    )
    cli = MagicMock()
    type(cli).pose = property(lambda self: pose)
    return cli, pose


class TestPickFaceNormal:
    """Static method — pure, already tested in test_face_pick.py.
    Here we verify delegation from CubeApproach works."""

    def test_cube_ahead_returns_front_face(self):
        result = CubeApproach.pick_face_normal(
            cube_pos=np.array([0.0, 300.0, 22.0]),
            cube_rot=np.eye(3),
            robot_pos=(0.0, 0.0),
            robot_yaw=0.0,
        )
        assert result is not None
        _nx, ny, heading = result
        assert abs(ny + 1.0) < 1e-6  # face normal outward toward robot (−Y)
        assert abs(heading - math.pi / 2) < 0.1

    def test_cube_at_robot_returns_none(self):
        result = CubeApproach.pick_face_normal(
            cube_pos=np.array([0.0, 0.0, 22.0]),
            cube_rot=np.eye(3),
            robot_pos=(0.0, 0.0),
            robot_yaw=0.0,
        )
        assert result is None


class TestStandoff:
    def test_standoff_computes_face_offset(self):
        cli, _ = _make_mock_cli()
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: None,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        state = _make_cube_state(x=100.0, y=200.0)
        sx, sy = ca._standoff(state, 90.0, (0.0, 1.0))
        assert sx == pytest.approx(100.0)
        assert sy == pytest.approx(200.0 + 90.0)


class TestVerifyGrabAlignment:
    def test_aligned_cube_returns_pass(self):
        # Cube at (0, 80), robot at origin facing +Y (heading=pi/2)
        # pick_face_normal returns heading=pi/2 for this geometry, so yaw must match
        cli, _ = _make_mock_cli(x=0.0, y=0.0, yaw=math.pi / 2)
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: _make_cube_state(x=0.0, y=80.0, confidence=0.95),
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        (passed, msgs, _facing_err, _forward, _lateral, centered, fresh) = ca._verify_grab_alignment(0)
        assert passed
        assert centered
        assert fresh
        assert len(msgs) == 3

    def test_no_cube_returns_fail(self):
        cli, _ = _make_mock_cli()
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: None,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        (passed, msgs, _, _, _, _, _) = ca._verify_grab_alignment(0)
        assert passed is False
        assert "no cube" in msgs[0]

    def test_stale_cube_returns_fail(self):
        cli, _ = _make_mock_cli()
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: _make_cube_state(x=0.0, y=80.0, updated_at=time.monotonic() - 10.0),
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        (passed, _, _, _, _, _, fresh) = ca._verify_grab_alignment(0)
        assert fresh is False
        assert passed is False

    def test_lateral_off_center_returns_fail(self):
        cli, _ = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: _make_cube_state(x=50.0, y=80.0, confidence=0.95),
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        (_passed, _, _, _, lateral, centered, _) = ca._verify_grab_alignment(0)
        assert not centered
        assert abs(lateral) > 25.0


class TestCorrectedTarget:
    def test_returns_none_when_cube_lost(self):
        cli, _ = _make_mock_cli()
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: None,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        assert ca._corrected_target(0, 90.0) is None

    def test_computes_corrected_approach(self):
        cli, _ = _make_mock_cli(x=0.0, y=0.0, yaw=math.pi / 2)
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: _make_cube_state(x=0.0, y=300.0),
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        result = ca._corrected_target(0, 90.0)
        assert result is not None
        tx, ty, _heading = result
        assert tx == pytest.approx(0.0, abs=1e-6)
        # Face normal (0,-1): standoff = 300 + (-1)*90 = 210
        assert ty == pytest.approx(210.0, abs=1e-6)


class TestApproachIntegration:
    """The approach arms/clears the carry observation around a grab."""

    @staticmethod
    def _fake_carry():
        calls = {"clear": 0, "note_grab": []}
        carry = MagicMock()
        carry.clear.side_effect = lambda: calls.__setitem__("clear", calls["clear"] + 1)
        carry.note_grab.side_effect = lambda i: calls["note_grab"].append(i)
        return carry, calls

    def test_successful_grab_arms_the_observation(self):
        # Robot parked at the step-2 standoff (0, 200) facing the cube at
        # (0, 300), so alignment passes and the grab code path runs.
        cli, _pose = _make_mock_cli(x=0.0, y=200.0, yaw=math.pi / 2)
        cube_state = _make_cube_state(x=0.0, y=300.0)
        carry, calls = self._fake_carry()

        def get_state(idx):
            cube_state.updated_at = time.monotonic()  # fresh during dwell
            return cube_state

        ca = CubeApproach(
            cli,
            get_cube_state=get_state,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
            carry=carry,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)

        # Forks down first (clears), then the grab arms the observation.
        assert calls["clear"] >= 1
        assert calls["note_grab"] == [0]

    def test_abandoned_approach_never_arms(self):
        """Cube lost at the standoff: no grab, so carry stays un-armed."""
        cli, _pose = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        cube_state = _make_cube_state(x=0.0, y=300.0)
        carry, calls = self._fake_carry()

        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: cube_state,  # updated_at never advances
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
            carry=carry,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)

        assert calls["clear"] >= 1
        assert calls["note_grab"] == []

    def test_approach_without_carry_still_works(self):
        """carry is optional — a standalone CubeApproach runs without it."""
        cli, _pose = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        cube_state = _make_cube_state(x=0.0, y=300.0)

        def get_state(idx):
            cube_state.updated_at = time.monotonic()
            return cube_state

        ca = CubeApproach(
            cli,
            get_cube_state=get_state,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)
        assert cli.go_to_pose.call_count >= 2


class TestApproachThread:
    def test_no_cube_prints_message(self):
        cli, _ = _make_mock_cli()
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: None,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)  # should not raise

    def test_full_approach_calls_cli(self):
        cli, _pose = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        cube_state = _make_cube_state(x=0.0, y=300.0)
        calls = {"drive_to": 0, "move_fwd": 0}

        def mock_drive(tx, ty, **kwargs):
            calls["drive_to"] += 1
            return True

        def mock_fwd(d):
            calls["move_fwd"] += 1
            return True

        def get_state(idx):
            # Simulate fresh camera frames being fused during the standoff
            # dwell so the approach proceeds past step 1.
            cube_state.updated_at = time.monotonic()
            return cube_state

        ca = CubeApproach(
            cli,
            get_cube_state=get_state,
            drive_to_point=mock_drive,
            move_forward=mock_fwd,
            spawn=_MOTION.spawn,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)

        # Lift and head were set
        cli.set_lift_height.assert_called()
        cli.set_head_angle.assert_called()
        # go_to_pose called (step 1 + step 2)
        assert cli.go_to_pose.call_count >= 2

    def test_approach_cancelled_when_cube_lost_at_standoff(self):
        """If the cube is never re-detected at the step-1 standoff, the
        approach is cancelled instead of using a stale pose."""
        cli, _pose = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        cube_state = _make_cube_state(x=0.0, y=300.0)
        calls = {"move_fwd": 0}

        def mock_fwd(d):
            calls["move_fwd"] += 1
            return True

        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: cube_state,  # updated_at never advances
            drive_to_point=lambda *a, **k: True,
            move_forward=mock_fwd,
            spawn=_MOTION.spawn,
        )
        with _MOTION.begin(CubeApproach.JOB_APPROACH) as job:
            ca._run(job, 0, 0.0, 300.0)

        # Step 1 only — no step 2, no grab.
        assert cli.go_to_pose.call_count == 1
        assert calls["move_fwd"] == 0
        # Cancel path stops motors.
        cli.stop_all_motors.assert_called()

    def test_second_approach_ignored_while_busy(self):
        """Calling approach() while one is in flight must not start another."""
        cli, _ = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        ca = CubeApproach(
            cli,
            get_cube_state=lambda i: _make_cube_state(x=0.0, y=300.0),
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        # Hold the approach lock to simulate an in-flight run.
        ca.approach_lock.acquire(blocking=False)
        try:
            ca.approach(0, 0.0, 300.0)
            time.sleep(0.2)  # give any (wrongly) started thread a chance
        finally:
            ca.approach_lock.release()

        # The second approach was refused — no run started, no go_to_pose.
        assert cli.go_to_pose.call_count == 0
        # And the lock is free again for the next real approach.
        assert ca.approach_lock.acquire(blocking=False)
        ca.approach_lock.release()

    def test_approach_runs_as_a_cancellable_job(self):
        """The approach is registered on the motion controller, so
        ``Navigator.stop`` can cancel it (and see it as active)."""
        cli, _pose = _make_mock_cli(x=0.0, y=0.0, yaw=0.0)
        in_lookup = threading.Event()
        release_lookup = threading.Event()

        def blocking_state(idx):
            del idx
            in_lookup.set()
            release_lookup.wait(2.0)
            # no return value -> get_cube_state reports "no cube" and the approach bails out

        ca = CubeApproach(
            cli,
            get_cube_state=blocking_state,
            drive_to_point=lambda *a, **k: True,
            move_forward=lambda d: True,
            spawn=_MOTION.spawn,
        )
        ca.approach(0, 0.0, 300.0)
        assert in_lookup.wait(2.0), "approach never started"

        assert CubeApproach.JOB_APPROACH in _MOTION.active_names
        assert _MOTION.cancel(CubeApproach.JOB_APPROACH)

        release_lookup.set()
        deadline = time.monotonic() + 2.0
        while CubeApproach.JOB_APPROACH in _MOTION.active_names and time.monotonic() < deadline:
            time.sleep(0.01)
        assert CubeApproach.JOB_APPROACH not in _MOTION.active_names
        assert ca.approach_lock.acquire(blocking=False)
        ca.approach_lock.release()
