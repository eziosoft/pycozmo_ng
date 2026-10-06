"""Tests for CubeStack — stacking the carried cube onto a second cube.

Covers the placement sequence (forks stay up, release-height drop, back
away, forks fully down, carry cleared), the entry guards (a cube must be
believed to be on the forks), the shared visit lock, and the Navigator
routing that turns a click on another cube while carrying into a stack.
"""

import math
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import pycozmo_ng
from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.navigation.cube_approach import CubeApproach
from pycozmo_ng.navigation.cube_stack import CubeStack
from pycozmo_ng.navigation.cube_visit import CubeVisit
from pycozmo_ng.navigation.dock_state import DockState
from pycozmo_ng.navigation.motion_controller import MotionController
from pycozmo_ng.navigation.navigator import Navigator

# Shared controller so spawned stacks are visible to assertions.
_MOTION = MotionController()


def _make_cube_state(
    x: float = 0.0,
    y: float = 300.0,
    confidence: float = 0.95,
    updated_at: float | None = None,
):
    if updated_at is None:
        updated_at = time.monotonic()
    return SimpleNamespace(
        position=np.array([x, y, 22.0]),
        rotation_matrix=np.eye(3),
        confidence=confidence,
        updated_at=updated_at,
    )


def _make_mock_cli(x: float = 0.0, y: float = 0.0, yaw: float = math.pi / 2):
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y),
        rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=yaw)),
    )
    cli = MagicMock()
    type(cli).pose = property(lambda self: pose)
    return cli, pose


def _fake_carry(on_forks: bool = True, cube_id: int | None = 1):
    carry = MagicMock()
    carry.on_forks = on_forks
    carry.cube_id = cube_id
    return carry


def _make_stack(
    cli=None,
    get_cube_state=None,
    carry=None,
    drive=None,
    move=None,
    spawn=_MOTION.spawn,
):
    """A CubeStack with a real CubeApproach sharing its DI seams."""
    if cli is None:
        cli, _ = _make_mock_cli()
    if get_cube_state is None:
        cube_state = _make_cube_state()

        def get_cube_state(idx):
            del idx
            cube_state.updated_at = time.monotonic()  # fresh during dwell
            return cube_state

    # One shared visit, exactly as Navigator wires it: grab and stack
    # contend on the same lock.
    visit = CubeVisit(spawn)
    approach = CubeApproach(
        cli,
        get_cube_state=get_cube_state,
        drive_to_point=drive or (lambda *a, **k: True),
        move_forward=move or (lambda d: True),
        visit=visit,
        carry=carry,
    )
    stack = CubeStack(
        cli=cli,
        drive_to_point=drive or (lambda *a, **k: True),
        move_forward=move or (lambda d: True),
        visit=visit,
        carry=carry,
        approach=approach,
    )
    return stack, approach


class TestStackPlacement:
    """The full placement sequence inside the stack job."""

    def test_release_then_back_away_then_forks_down(self):
        # Robot parked at the step-2 standoff (0, 200) facing the cube at
        # (0, 300), so drive_and_align verifies and _place runs.
        cli, _pose = _make_mock_cli(x=0.0, y=200.0)
        carry = _fake_carry()
        drives = []

        def drive(*args, **kwargs):
            drives.append(kwargs)
            return True

        stack, _ = _make_stack(cli=cli, carry=carry, drive=drive)
        with _MOTION.begin(CubeStack.JOB_STACK) as job:
            stack._run(job, 0, 0.0, 300.0)

        # Exactly two lift commands: the release height, then fully
        # down (after backing out) — never the grab's forks-down prelude.
        lift_calls = [c.args[0] for c in cli.set_lift_height.call_args_list]
        assert lift_calls == [
            CubeStack.PLACE_LIFT_MM,
            pycozmo_ng.MIN_LIFT_HEIGHT.mm,
        ]
        # One back-away drive (the other drive_to_point is the
        # approach's "too close" reverse at the start).
        assert len(drives) == 2
        # The carry observation ends only after the stack is complete,
        # and stacking never re-arms a grab.
        carry.clear.assert_called_once()
        carry.note_grab.assert_not_called()

    def test_release_height_clears_the_stack_while_backing(self):
        # Live-tuned: the prongs must stay above the 44 mm cube below
        # while the robot reverses (half up, 46 mm, wedged against it
        # and dragged the stack); the carried cube then drops the
        # remaining gap onto the stack as the prongs withdraw.
        assert CubeStack.PLACE_LIFT_MM == 72.0
        assert 44.0 < CubeStack.PLACE_LIFT_MM < pycozmo_ng.MAX_LIFT_HEIGHT.mm

    def test_unknown_cube_never_places(self):
        cli, _ = _make_mock_cli()
        carry = _fake_carry()
        stack, _ = _make_stack(cli=cli, get_cube_state=lambda i: None, carry=carry)
        with _MOTION.begin(CubeStack.JOB_STACK) as job:
            stack._run(job, 0, 0.0, 300.0)

        cli.set_lift_height.assert_not_called()
        carry.clear.assert_not_called()

    def test_failed_back_away_leaves_the_forks_at_the_release_height(self):
        """If the robot cannot reverse, the final lowering is skipped —
        dropping the forks while still over the stack would push it."""
        cli, _pose = _make_mock_cli(x=0.0, y=200.0)
        carry = _fake_carry()
        stack, _ = _make_stack(cli=cli, carry=carry, drive=lambda *a, **k: False)
        with _MOTION.begin(CubeStack.JOB_STACK) as job:
            stack._run(job, 0, 0.0, 300.0)

        lift_calls = [c.args[0] for c in cli.set_lift_height.call_args_list]
        assert lift_calls == [CubeStack.PLACE_LIFT_MM]  # no final MIN
        carry.clear.assert_not_called()

    def test_cancelled_stack_releases_lock_and_never_places(self):
        cli, _ = _make_mock_cli()
        carry = _fake_carry()
        in_lookup = threading.Event()
        release_lookup = threading.Event()

        def blocking_state(idx):
            del idx
            in_lookup.set()
            release_lookup.wait(2.0)  # returns None -> cube unknown

        stack, _ = _make_stack(cli=cli, get_cube_state=blocking_state, carry=carry)
        assert stack.stack(1, 0.0, 300.0) is True
        assert in_lookup.wait(2.0), "stack never started"
        assert CubeStack.JOB_STACK in _MOTION.active_names
        assert _MOTION.cancel(CubeStack.JOB_STACK)

        release_lookup.set()
        deadline = time.monotonic() + 2.0
        while CubeStack.JOB_STACK in _MOTION.active_names and time.monotonic() < deadline:
            time.sleep(0.01)
        assert CubeStack.JOB_STACK not in _MOTION.active_names

        cli.set_lift_height.assert_not_called()
        carry.clear.assert_not_called()
        # The shared visit lock was released -> the next visit can start.
        assert not stack.visit.busy


class TestStackEntry:
    """Guards on the public ``stack()`` entry point."""

    @staticmethod
    def _recording_spawn():
        calls = []

        def spawn(name, fn):
            calls.append((name, fn))

        return calls, spawn

    def test_refuses_without_a_carry_detector(self):
        calls, spawn = self._recording_spawn()
        stack, _ = _make_stack(carry=None, spawn=spawn)
        assert stack.stack(1, 0.0, 300.0) is False
        assert calls == []

    def test_refuses_when_no_cube_on_the_forks(self):
        calls, spawn = self._recording_spawn()
        stack, _ = _make_stack(carry=_fake_carry(on_forks=False), spawn=spawn)
        assert stack.stack(1, 0.0, 300.0) is False
        assert calls == []

    def test_refuses_the_held_cube_as_target(self):
        calls, spawn = self._recording_spawn()
        carry = _fake_carry(on_forks=True, cube_id=1)
        stack, _ = _make_stack(carry=carry, spawn=spawn)
        assert stack.stack(0, 0.0, 300.0) is False  # cube_idx 0 == cube_id 1 == held cube
        assert calls == []

    def test_refuses_while_another_visit_is_in_flight(self):
        calls, spawn = self._recording_spawn()
        stack, approach = _make_stack(carry=_fake_carry(on_forks=True, cube_id=1), spawn=spawn)
        # A grab approach is already in flight on the shared visit...
        assert approach.visit.begin(CubeApproach.JOB_APPROACH, lambda job: None) is True
        # ...so the stack is refused (not queued) and reports it.
        assert stack.stack(1, 0.0, 300.0) is False
        # Only the occupier ever started.
        assert [name for name, _ in calls] == [CubeApproach.JOB_APPROACH]

    def test_starts_as_a_stack_job_and_holds_the_visit_lock(self):
        calls, spawn = self._recording_spawn()
        stack, _approach = _make_stack(carry=_fake_carry(on_forks=True, cube_id=1), spawn=spawn)
        assert stack.stack(1, 0.0, 300.0) is True
        assert [name for name, _ in calls] == [CubeStack.JOB_STACK]
        # The visit is held for the (not yet run) job...
        assert stack.visit.busy
        # ...and released again by the job's finally when it finishes.
        name, fn = calls[0]
        with _MOTION.begin(name) as job:
            fn(job)
        assert not stack.visit.busy


class TestNavigatorRouting:
    """A click on another cube while carrying routes to the stack."""

    @staticmethod
    def _nav(carry):
        nav = Navigator(DockState(), world_map=MagicMock())
        nav.cli = MagicMock()
        nav.carry = carry
        nav._cube_stack = MagicMock()  # pre-built: no lazy creation
        nav._cube_approach = MagicMock()
        # The sub-sequences answer "did the visit start?"; routing must
        # propagate that answer instead of reporting success itself.
        nav._cube_stack.stack.return_value = True
        nav._cube_approach.approach.return_value = True
        return nav

    @staticmethod
    def _carry(on_forks: bool, cube_id: int | None = 1):
        carry = MagicMock()
        carry.on_forks = on_forks
        carry.cube_id = cube_id
        return carry

    def test_click_while_carrying_routes_to_stack(self):
        nav = self._nav(self._carry(on_forks=True, cube_id=1))
        assert nav.approach_cube_at(1, 100.0, 200.0) is True  # cube 2 (idx 1)
        nav._cube_stack.stack.assert_called_once_with(1, 100.0, 200.0)
        nav._cube_approach.approach.assert_not_called()

    def test_click_without_cube_routes_to_grab(self):
        nav = self._nav(self._carry(on_forks=False, cube_id=1))
        assert nav.approach_cube_at(1, 100.0, 200.0) is True
        nav._cube_approach.approach.assert_called_once_with(1, 100.0, 200.0)
        nav._cube_stack.stack.assert_not_called()

    def test_no_carry_detector_routes_to_grab(self):
        nav = self._nav(None)
        assert nav.approach_cube_at(1, 100.0, 200.0) is True
        nav._cube_approach.approach.assert_called_once_with(1, 100.0, 200.0)
        nav._cube_stack.stack.assert_not_called()

    def test_clicking_the_held_cube_stays_a_grab(self):
        nav = self._nav(self._carry(on_forks=True, cube_id=1))
        assert nav.approach_cube_at(0, 0.0, 300.0) is True  # cube 1 == the held cube
        nav._cube_approach.approach.assert_called_once_with(0, 0.0, 300.0)
        nav._cube_stack.stack.assert_not_called()

    def test_a_refused_visit_is_reported_not_hidden(self):
        """A busy visit must not be reported as a started one.

        This is the contract ``cozmo.stack_cube`` documents: True only
        when the job actually started.
        """
        nav = self._nav(self._carry(on_forks=True, cube_id=1))
        nav.world_map.get_cube.return_value = SimpleNamespace(position=np.array([50.0, 300.0, 0.0]))
        nav._cube_stack.stack.return_value = False  # another visit in flight
        assert nav.stack_cube(2) is False
        assert nav.approach_cube_at(1, 50.0, 300.0) is False

    def test_stack_cube_refuses_without_cube_on_forks(self):
        nav = self._nav(self._carry(on_forks=False))
        assert nav.stack_cube(2) is False
        nav._cube_approach.approach.assert_not_called()
        nav._cube_stack.stack.assert_not_called()

    def test_stack_cube_refuses_the_held_cube(self):
        nav = self._nav(self._carry(on_forks=True, cube_id=1))
        assert nav.stack_cube(1) is False
        nav._cube_stack.stack.assert_not_called()

    def test_stack_cube_resolves_and_routes(self):
        nav = self._nav(self._carry(on_forks=True, cube_id=1))
        nav.world_map.get_cube.return_value = SimpleNamespace(position=np.array([50.0, 300.0, 0.0]))
        assert nav.stack_cube(2) is True
        nav.world_map.get_cube.assert_called_once_with(1)  # world index
        nav._cube_stack.stack.assert_called_once_with(1, 50.0, 300.0)
        nav._cube_approach.approach.assert_not_called()


def test_held_cube_id_helper_matches_identity():
    # Sanity for the routing comparisons: cube ids 1-3, indices 0-2.
    assert to_cube_id(0) == 1
    assert to_cube_id(2) == 3
