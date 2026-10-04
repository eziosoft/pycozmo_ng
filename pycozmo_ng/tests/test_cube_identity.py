"""Tests for cube identity — the single owner of ID ↔ index conversion.

The convention split (public cube IDs 1-3 vs internal world index 0-2)
only works if every conversion goes through this module and validates.
"""

import pytest

from pycozmo_ng.cubes.identity import (
    CUBE_COUNT,
    cube_label,
    parse_cube_label,
    to_cube_id,
    to_world_index,
)


class TestToWorldIndex:
    @pytest.mark.parametrize(
        ("cube_id", "expected"),
        [(1, 0), (2, 1), (3, 2)],
    )
    def test_valid_ids(self, cube_id, expected):
        assert to_world_index(cube_id) == expected

    @pytest.mark.parametrize("cube_id", [0, 4, -1, 10])
    def test_out_of_range_raises(self, cube_id):
        with pytest.raises(ValueError):
            to_world_index(cube_id)


class TestToCubeId:
    @pytest.mark.parametrize(
        ("cube_index", "expected"),
        [(0, 1), (1, 2), (2, 3)],
    )
    def test_valid_indices(self, cube_index, expected):
        assert to_cube_id(cube_index) == expected

    @pytest.mark.parametrize("cube_index", [-1, 3, 42])
    def test_out_of_range_raises(self, cube_index):
        with pytest.raises(ValueError):
            to_cube_id(cube_index)


class TestRoundTrip:
    def test_id_index_round_trip(self):
        for cube_id in range(1, CUBE_COUNT + 1):
            assert to_cube_id(to_world_index(cube_id)) == cube_id

    def test_label_round_trip(self):
        for cube_index in range(CUBE_COUNT):
            assert parse_cube_label(cube_label(cube_index)) == cube_index


class TestCubeLabel:
    def test_labels(self):
        assert cube_label(0) == "cube_1"
        assert cube_label(2) == "cube_3"

    @pytest.mark.parametrize("label", ["cube_0", "cube_4", "cube_99", "1", "", "cube_", "cube_x"])
    def test_bad_labels_raise(self, label):
        with pytest.raises(ValueError):
            parse_cube_label(label)

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            parse_cube_label(None)  # type: ignore[arg-type]
