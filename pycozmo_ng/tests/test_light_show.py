"""Cube light patterns and the show runner.

``LightPattern`` is pure — it turns a frame index into LED states and never
touches the robot — so most of this is deterministic maths.  ``CubeLightShow``
is exercised against a fake manager, which is the seam that keeps the runner
testable without a cube.
"""

import pytest

from pycozmo_ng.cubes.light_pattern import CORNERS, PATTERN_NAMES, LightPattern
from pycozmo_ng.cubes.light_show import CubeLightShow


def colors(pattern: LightPattern, index: int) -> dict[int, tuple[int, ...]]:
    """Frame as plain ints, since LightState has no __eq__."""
    return {cube_id: tuple(state.on_color for state in states) for cube_id, states in pattern.frame(index).items()}


class TestLightPattern:
    def test_every_pattern_gives_four_corners_per_cube(self) -> None:
        for name in PATTERN_NAMES:
            frame = LightPattern(name).frame(0)
            assert sorted(frame) == [1, 2, 3]
            assert all(len(states) == CORNERS for states in frame.values())

    def test_frames_are_deterministic(self) -> None:
        for name in PATTERN_NAMES:
            pattern = LightPattern(name)
            assert colors(pattern, 7) == colors(pattern, 7)

    def test_rainbow_cycles_and_neighbours_differ(self) -> None:
        pattern = LightPattern("rainbow")
        frame = colors(pattern, 0)
        assert frame[1] != frame[2] != frame[3]
        assert colors(pattern, 0) != colors(pattern, 1)

    def test_rainbow_corners_match_within_a_cube(self) -> None:
        frame = colors(LightPattern("rainbow"), 3)
        assert all(len(set(states)) == 1 for states in frame.values())

    def test_solid_never_changes(self) -> None:
        pattern = LightPattern("solid")
        assert colors(pattern, 0) == colors(pattern, 50)

    def test_chase_lights_exactly_one_corner_per_cube(self) -> None:
        pattern = LightPattern("chase")
        for index in range(6):
            for states in colors(pattern, index).values():
                lit = [c for c in states if c != 0]
                assert len(lit) == 1, states

    def test_chase_moves_between_frames(self) -> None:
        pattern = LightPattern("chase")
        lit_at = [next(i for i, c in enumerate(colors(pattern, f)[1]) if c != 0) for f in range(CORNERS)]
        assert lit_at == [0, 1, 2, 3]

    def test_pulse_varies_and_stays_in_range(self) -> None:
        pattern = LightPattern("pulse")
        frames = [colors(pattern, i)[1][0] for i in range(6)]
        assert len(set(frames)) > 1
        assert all(0 <= value <= 0x7FFF for value in frames)

    def test_cube_ids_are_respected(self) -> None:
        frame = LightPattern("rainbow", cube_ids=(2, 3)).frame(0)
        assert sorted(frame) == [2, 3]

    def test_unknown_pattern_rejected(self) -> None:
        with pytest.raises(ValueError):
            LightPattern("disco")


class FakeManager:
    """Records what the show asks for; stands in for CubeManager."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.light_refresh: bool | None = None

    def set_cube_states(self, cli, states_by_cube) -> None:
        self.sent.append(states_by_cube)

    def set_light_refresh(self, enabled: bool) -> None:
        self.light_refresh = enabled


class TestCubeLightShow:
    def test_tick_sends_one_frame_and_advances(self) -> None:
        manager = FakeManager()
        show = CubeLightShow(manager)
        show.tick()
        show.tick()
        assert len(manager.sent) == 2
        assert sorted(manager.sent[0]) == [1, 2, 3]
        assert manager.sent[0] != manager.sent[1]

    def test_pattern_can_be_swapped(self) -> None:
        manager = FakeManager()
        show = CubeLightShow(manager, LightPattern("solid"))
        show.set_pattern(LightPattern("chase"))
        assert show.pattern.name == "chase"

    def test_show_pauses_keepalive_light_refresh(self) -> None:
        """Otherwise the green keepalive refresh overwrites every frame."""
        manager = FakeManager()
        show = CubeLightShow(manager)
        assert show.running is False

        show.start(cli=None)
        try:
            assert manager.light_refresh is False
            assert show.running is True
        finally:
            show.stop()
        assert manager.light_refresh is True
        assert show.running is False

    def test_start_is_idempotent(self) -> None:
        manager = FakeManager()
        show = CubeLightShow(manager)
        show.start(cli=None)
        try:
            show.start(cli=None)
            assert show.running is True
        finally:
            show.stop()
