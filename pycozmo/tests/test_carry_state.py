"""Tests for the carry-state concept (cube on the lift, checked on demand).

Covers the post-grab visibility heuristic: forks down is authoritative,
lift up + grabbed cube still in frame = grab missed, lift up + cube gone
from the world map = carrying. Nothing polls — every answer comes from the
moment ``state`` is read.
"""

from types import SimpleNamespace

from pycozmo.cubes.carry_state import CarryDetector, CarryState

LIFT_UP = 92.0
LIFT_DOWN = 32.0


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _detector(lift: float = LIFT_UP, seen_age_s: float = 0.0, cube_index: int = 0):
    """Detector with a controllable lift, a visible cube, and a fake clock."""
    clock = FakeClock()
    lift_mm = [lift]
    cube = SimpleNamespace(updated_at=clock.now - seen_age_s)
    det = CarryDetector(
        get_cube_state=lambda i: cube if i == cube_index else None,
        get_lift_height=lambda: lift_mm[0],
        clock=clock,
    )
    return det, clock, lift_mm, cube


class TestOnDemandStatus:
    def test_lift_up_without_a_grab_is_unknown(self):
        det, _, _, _ = _detector(lift=LIFT_UP)
        assert det.state is CarryState.UNKNOWN
        assert det.is_carrying is None
        assert det.cube_id is None

    def test_forks_down_is_authoritatively_not_carrying(self):
        det, _, _, _ = _detector(lift=LIFT_DOWN)
        assert det.state is CarryState.NOT_CARRYING
        assert det.is_carrying is False

    def test_cube_still_visible_means_grab_missed(self):
        det, _, _, _ = _detector(lift=LIFT_UP, seen_age_s=0.0)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING
        assert det.is_carrying is False
        assert det.cube_id == 1  # public IDs are 1-3

    def test_cube_gone_from_frame_means_carrying(self):
        det, _, _, _ = _detector(lift=LIFT_UP, seen_age_s=3.0)
        det.note_grab(0)
        assert det.state is CarryState.CARRYING
        assert det.is_carrying is True
        assert det.cube_id == 1

    def test_short_detection_gap_is_not_yet_carrying(self):
        # Detections blink; only LOST_S without an update counts as gone.
        det, _, _, _ = _detector(lift=LIFT_UP, seen_age_s=0.5)
        det.note_grab(0)
        assert det.state is CarryState.NOT_CARRYING

    def test_reappearing_cube_flips_back_on_the_next_read(self):
        det, clock, _, cube = _detector(lift=LIFT_UP, seen_age_s=3.0)
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
            clock=clock,
        )
        det.note_grab(0)
        assert det.state is CarryState.CARRYING

    def test_status_is_re_evaluated_on_every_read(self):
        det, clock, lift_mm, cube = _detector(lift=LIFT_DOWN, seen_age_s=0.0)
        det.note_grab(0)
        # Lift still travelling up (transit grace): visibility decides.
        assert det.state is CarryState.NOT_CARRYING
        assert det.cube_id == 1  # not disarmed while in transit

        # Cube leaves the frame, lift reaches the top: carrying.
        lift_mm[0] = LIFT_UP
        clock.advance(CarryDetector.LOST_S + 0.1)
        assert det.state is CarryState.CARRYING

        # Cube reappears in view: immediately not carrying again.
        cube.updated_at = clock.now
        assert det.state is CarryState.NOT_CARRYING


class TestReleaseAndDrop:
    def test_lift_down_after_transit_clears_the_observation(self):
        det, clock, lift_mm, _ = _detector(lift=LIFT_UP, seen_age_s=3.0)
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
        det, _, _, _ = _detector(lift=LIFT_UP, seen_age_s=3.0)
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
