"""Tests for the carry-state concept (cube on the lift, checked on demand).

Covers the post-grab visibility heuristic: wait for the fork to finish
rising, then let visibility decide — forks up + grabbed cube still in
frame = grab missed, forks up + cube gone from the world map = carrying,
forks down = not carrying. Nothing polls — every answer comes from the
moment ``state`` is read.
"""

from types import SimpleNamespace

from pycozmo.cubes.carry_state import CarryDetector, CarryState
from pycozmo.util.robot import RobotStatusFlag

LIFT_UP = 92.0
LIFT_DOWN = 32.0
FORK_SETTLED = RobotStatusFlag.LIFT_IN_POS
FORK_MOVING = 0


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _detector(
    lift: float = LIFT_UP,
    seen_age_s: float = 0.0,
    cube_index: int = 0,
    in_pos: bool = True,
):
    """Detector with controllable lift, fork flag, a cube, and a fake clock."""
    clock = FakeClock()
    lift_mm = [lift]
    in_pos_flag = [in_pos]
    cube = SimpleNamespace(updated_at=clock.now - seen_age_s)
    det = CarryDetector(
        get_cube_state=lambda i: cube if i == cube_index else None,
        get_lift_height=lambda: lift_mm[0],
        get_lift_in_pos=lambda: in_pos_flag[0],
        clock=clock,
    )
    return det, clock, lift_mm, cube, in_pos_flag


class TestOnDemandStatus:
    def test_lift_up_without_a_grab_is_unknown(self):
        det, *_ = _detector(lift=LIFT_UP)
        assert det.state is CarryState.UNKNOWN
        assert det.is_carrying is None
        assert det.cube_id is None

    def test_forks_down_is_authoritatively_not_carrying(self):
        det, *_ = _detector(lift=LIFT_DOWN)
        assert det.state is CarryState.NOT_CARRYING
        assert det.is_carrying is False

    def test_moving_lift_unarmed_is_unknown(self):
        # Mid-travel the fork is neither up nor down: no verdict.
        det, *_ = _detector(lift=LIFT_DOWN, in_pos=False)
        assert det.state is CarryState.UNKNOWN

    def test_cube_still_visible_means_grab_missed(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING
        assert det.is_carrying is False
        assert det.cube_id == 1  # public IDs are 1-3

    def test_cube_gone_from_frame_means_carrying(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING
        assert det.is_carrying is True
        assert det.cube_id == 1

    def test_short_detection_gap_is_not_yet_carrying(self):
        # Detections blink for a frame or two; only LOST_S (0.3s) without
        # an update counts as gone.
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=0.1)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING

    def test_reappearing_cube_flips_back_on_the_next_read(self):
        det, clock, _, cube, _ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING

        # A new frame sees the cube again (e.g. the robot turned around).
        clock.advance(0.1)
        cube.updated_at = clock.now
        assert det.state is CarryState.NOT_CARRYING

    def test_unknown_cube_counts_as_gone(self):
        clock = FakeClock()
        det = CarryDetector(
            get_cube_state=lambda i: None,
            get_lift_height=lambda: LIFT_UP,
            get_lift_in_pos=lambda: True,
            clock=clock,
        )
        det.note_grab(0)
        assert det.state is CarryState.CARRYING

    def test_status_is_re_evaluated_on_every_read(self):
        det, clock, _, cube, in_pos_flag = _detector(lift=LIFT_DOWN, seen_age_s=0.0)
        det.note_grab(0)
        # Command sent, fork not moving yet: wait, do not judge.
        assert det.state is CarryState.UNKNOWN
        assert det.cube_id == 1  # not disarmed while waiting

        # The lift motor starts travelling — still no verdict.
        in_pos_flag[0] = False
        assert det.state is CarryState.UNKNOWN

        # The fork settles up (height feedback still reads low) and the
        # cube has left the frame: carrying.
        in_pos_flag[0] = True
        clock.advance(CarryDetector.LOST_S + 0.1)
        assert det.state is CarryState.CARRYING

        # Cube reappears in view: immediately not carrying again.
        cube.updated_at = clock.now
        assert det.state is CarryState.NOT_CARRYING


class TestOnForks:
    """``on_forks`` — the "may I stack?" predicate for CubeStack."""

    def test_unarmed_is_false(self):
        det, *_ = _detector(lift=LIFT_DOWN)
        assert det.on_forks is False
        # Forks up without a witnessed grab: unknown, but nothing is
        # believed to be riding them, so stacking is refused too.
        det_up, *_ = _detector(lift=LIFT_UP)
        assert det_up.state is CarryState.UNKNOWN
        assert det_up.on_forks is False

    def test_armed_while_the_fork_rises_is_true(self):
        det, *_ = _detector(lift=LIFT_DOWN, in_pos=False, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.UNKNOWN  # verdict pending
        assert det.on_forks is True

    def test_carrying_is_true(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING
        assert det.on_forks is True

    def test_resolved_miss_is_false(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING
        assert det.on_forks is False

    def test_cleared_is_false(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.on_forks is True
        det.clear()
        assert det.on_forks is False

    def test_forks_settling_down_forgets_the_observation(self):
        det, clock, *_ = _detector(lift=LIFT_DOWN, in_pos=True, seen_age_s=3.0)
        det.note_grab(0)
        # Settled but within the transit grace: verdict still pending.
        assert det.state is CarryState.UNKNOWN
        assert det.on_forks is True
        # Past the grace the rise clearly never happened: auto-cleared.
        clock.advance(CarryDetector.LIFT_TRANSIT_S + 0.1)
        assert det.state is CarryState.NOT_CARRYING
        assert det.on_forks is False


class TestWaitForFork:
    """Judge only once the fork has finished rising — never mid-travel."""

    def test_no_verdict_while_the_lift_motor_travels(self):
        det, *_ = _detector(lift=LIFT_DOWN, seen_age_s=3.0, in_pos=False)
        det.note_grab(0)
        # The height may even read "up" mid-travel: still wait for settle.
        assert det.state is CarryState.UNKNOWN
        assert det.cube_id == 1

    def test_carrying_with_stale_lift_feedback_after_the_rise(self):
        # Regression: lift_height_mm can lag or read low while the fork is
        # physically up. Motion + settle is the "fork is up" signal; the
        # height reading must not disarm the observation.
        det, clock, lift_mm, _, in_pos_flag = _detector(
            lift=LIFT_DOWN, seen_age_s=0.0, in_pos=False
        )
        det.note_grab(0)
        clock.advance(0.1)
        assert det.state is CarryState.UNKNOWN  # lift travelling

        # Motor stops — fork is up — but the feedback still reads low.
        in_pos_flag[0] = True
        lift_mm[0] = LIFT_DOWN
        clock.advance(CarryDetector.LOST_S + 0.1)
        assert det.state is CarryState.CARRYING
        assert det.cube_id == 1

    def test_missed_grab_with_stale_lift_feedback_reads_not_carrying(self):
        # Same as above, but the cube stayed in view: the grab missed.
        det, clock, _, cube, in_pos_flag = _detector(
            lift=LIFT_DOWN, seen_age_s=0.0, in_pos=False
        )
        det.note_grab(0)
        assert det.state is CarryState.UNKNOWN  # lift travelling

        # Fork settles (height still reads low), cube still tracked.
        in_pos_flag[0] = True
        clock.advance(0.5)
        cube.updated_at = clock.now
        assert det.state is CarryState.NOT_CARRYING
        assert det.cube_id == 1

    def test_settled_low_fork_that_never_rises_clears_the_observation(self):
        # No motion at all and the height stays down: the rise never
        # happened — forks down is authoritative, forget the grab.
        det, clock, *_ = _detector(lift=LIFT_DOWN, seen_age_s=0.0, in_pos=True)
        det.note_grab(0)
        assert det.state is CarryState.UNKNOWN  # transit grace
        clock.advance(CarryDetector.LIFT_TRANSIT_S + 0.1)
        assert det.state is CarryState.NOT_CARRYING
        assert det.cube_id is None


class TestResolveOnSight:
    """Seeing the cube after the grab ends the check for good.

    Regression: a cube put down and seen, then left behind as the robot
    drives away (cube leaves the frame), must not flip carry back to
    true — the observation latches until the next grab.
    """

    def test_seen_cube_stays_not_carrying_when_it_leaves_view(self):
        det, clock, _, _, _ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING  # seen -> resolved

        # Robot drives away: the cube leaves the camera view entirely.
        clock.advance(CarryDetector.LOST_S + 1.0)
        assert det.state is CarryState.NOT_CARRYING  # latched, no flip

        clock.advance(30.0)
        assert det.state is CarryState.NOT_CARRYING  # and it stays
        assert det.cube_id == 1

    def test_reappearing_cube_resolves_and_stays_resolved(self):
        # carrying (never seen since the grab) -> cube reappears -> the
        # verdict resolves to NOT_CARRYING and cannot flip back either.
        det, clock, _, cube, _ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING  # not seen yet

        cube.updated_at = clock.now
        assert det.state is CarryState.NOT_CARRYING  # seen -> resolved

        clock.advance(CarryDetector.LOST_S + 1.0)
        assert det.state is CarryState.NOT_CARRYING  # latched

    def test_next_grab_rearms_the_check(self):
        det, clock, *_ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING  # resolved

        clock.advance(5.0)
        det.note_grab(0)  # a fresh grab re-arms the visibility check
        assert det.state is CarryState.CARRYING  # cube not in frame anymore

    def test_stale_grab_time_sighting_does_not_resolve_the_rise(self):
        # Live-log regression: the cube's last sighting is the grab
        # itself (it left the frame on the rising forks), and the fork
        # finishes rising ~1.3s later. That stale sighting must NOT
        # count as "still in frame" — the verdict is CARRYING.
        det, clock, *_ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)  # grab at t0, last sighting == t0

        clock.advance(1.3)  # forks finish rising
        assert det.state is CarryState.CARRYING
        assert det.is_carrying is True

        clock.advance(10.0)  # carried: never updated again -> stays
        assert det.state is CarryState.CARRYING


class TestReleaseAndDrop:
    def test_lift_down_after_transit_clears_the_observation(self):
        det, clock, lift_mm, *_ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING

        # The lift goes down for real (drop/placed) — past the transit grace.
        clock.advance(CarryDetector.LIFT_TRANSIT_S + 0.1)
        lift_mm[0] = LIFT_DOWN
        assert det.state is CarryState.NOT_CARRYING
        assert det.cube_id is None

        # Raising the lift again without a witnessed grab is unknown.
        lift_mm[0] = LIFT_UP
        assert det.state is CarryState.UNKNOWN

    def test_clear_forgets_the_observation(self):
        det, *_ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(2)
        assert det.cube_id == 3
        det.clear()
        assert det.cube_id is None
        assert det.state is CarryState.UNKNOWN


class TestBrainWiring:
    def test_cozmo_exposes_carry_status(self):
        from pycozmo import Cozmo

        cozmo = Cozmo()
        # No connection: lift reads 0.0 -> forks down -> not carrying.
        assert cozmo.carry.state is CarryState.NOT_CARRYING
        assert cozmo.carry.is_carrying is False
        assert cozmo.carry.cube_id is None

    def test_brain_carries_a_detector(self):
        from pycozmo.brain import Brain

        brain = Brain()
        assert isinstance(brain.carry, CarryDetector)
        # The lift callback follows the live client.
        brain.carry.note_grab(0)
        assert brain.carry.cube_id == 1


def _fake_cli(lift_mm: float, robot_status: int = FORK_SETTLED):
    """Minimal robot client for Brain._on_robot_state telemetry reads."""
    return SimpleNamespace(
        pose=SimpleNamespace(
            position=SimpleNamespace(x_y_z=(0.0, 0.0, 0.0), x=0.0, y=0.0),
            rotation=SimpleNamespace(angle_z=SimpleNamespace(radians=0.0)),
            origin_id=1,
        ),
        battery_voltage=3.9,
        left_wheel_speed=SimpleNamespace(mmps=0.0),
        right_wheel_speed=SimpleNamespace(mmps=0.0),
        head_angle=SimpleNamespace(degrees=0.0),
        lift_position=SimpleNamespace(height=SimpleNamespace(mm=lift_mm)),
        robot_status=robot_status,
        cliff_data=None,
    )


class TestWebTelemetry:
    """The browser HUD gets carry status through the telemetry push."""

    @staticmethod
    def _push(brain) -> dict:
        captured: dict = {}
        brain.web_viewer.update_telemetry = lambda **kw: captured.update(kw)
        brain._on_robot_state(_fake_cli(lift_mm=92.0))
        assert captured, "telemetry push skipped (incomplete fake cli)"
        return captured

    def test_robot_state_pushes_not_carrying(self):
        from pycozmo.brain import Brain

        brain = Brain()
        brain._cli = _fake_cli(lift_mm=32.0)  # forks down
        telemetry = self._push(brain)
        assert telemetry["carry"] == "not_carrying"
        assert telemetry["carry_cube"] is None

    def test_robot_state_pushes_carrying(self):
        from pycozmo.brain import Brain

        brain = Brain()
        brain._cli = _fake_cli(lift_mm=92.0)  # forks up
        brain.carry.note_grab(0)  # armed; cube never seen again
        telemetry = self._push(brain)
        assert telemetry["carry"] == "carrying"
        assert telemetry["carry_cube"] == 1

    def test_lift_up_without_grab_pushes_unknown(self):
        from pycozmo.brain import Brain

        brain = Brain()
        brain._cli = _fake_cli(lift_mm=92.0)
        telemetry = self._push(brain)
        assert telemetry["carry"] == "unknown"

    def test_robot_state_pushes_unknown_while_fork_rises(self):
        from pycozmo.brain import Brain

        brain = Brain()
        # Lift motor travelling (LIFT_IN_POS clear) right after a grab.
        brain._cli = _fake_cli(lift_mm=32.0, robot_status=FORK_MOVING)
        brain.carry.note_grab(0)
        telemetry = self._push(brain)
        assert telemetry["carry"] == "unknown"
        assert telemetry["carry_cube"] == 1
