"""SetPropSlot (command 0x05) — wire layout and per-cube slot allocation.

Command 0x05 changed meaning between firmware 2381 and 2458: the old
``ObjectConnect{factory_id, connect: bool}`` became ``SetPropSlot{factory_id,
slot: u8}`` — same ID, same 5 bytes.  Sending a boolean ``connect=True`` is
therefore read as *"slot 1"*, so every cube lands in one slot and each connect
evicts the previous one.  These tests pin the grammar that makes all three
cubes connect at once (verified on hardware 2026-10-15).
"""

import pytest

from pycozmo_ng.cubes.manager import CubeManager
from pycozmo_ng.protocol.protocol_encoder import PACKETS_BY_ID, SetPropSlot


class TestSetPropSlotWireLayout:
    """The robot reads the second byte as a slot index, not a flag."""

    def test_is_command_0x05(self) -> None:
        assert SetPropSlot(0, 0).id == 0x05
        assert PACKETS_BY_ID[0x05] is SetPropSlot

    def test_encodes_factory_id_then_slot(self) -> None:
        packet = SetPropSlot(factory_id=0xDEADBEEF, slot=2)
        assert packet.to_bytes() == bytes([0xEF, 0xBE, 0xAD, 0xDE, 0x02])

    def test_is_five_bytes(self) -> None:
        assert len(SetPropSlot(0, 0).to_bytes()) == 5

    def test_factory_id_zero_releases_the_slot(self) -> None:
        # The engine releases a slot with factory_id 0 (Robot::ConnectToRequestedObjects).
        assert SetPropSlot(factory_id=0, slot=3).to_bytes() == bytes([0, 0, 0, 0, 3])

    def test_round_trips(self) -> None:
        original = SetPropSlot(factory_id=0x3B3958B8, slot=4)
        decoded = SetPropSlot.from_bytes(original.to_bytes())
        assert (decoded.factory_id, decoded.slot) == (0x3B3958B8, 4)

    def test_slot_is_not_a_boolean(self) -> None:
        """A bool field would collapse every cube into one slot — the old bug."""
        assert not hasattr(SetPropSlot(0, 0), "connect")
        assert SetPropSlot(factory_id=1, slot=2).slot == 2


class TestSetPropSlotValidation:
    def test_slot_range(self) -> None:
        assert SetPropSlot(0, 255).slot == 255
        with pytest.raises(ValueError):
            SetPropSlot(0, 256)
        with pytest.raises(ValueError):
            SetPropSlot(0, -1)

    def test_factory_id_range(self) -> None:
        assert SetPropSlot(0xFFFFFFFF, 0).factory_id == 0xFFFFFFFF
        with pytest.raises(ValueError):
            SetPropSlot(-1, 0)


class TestCubeSlotAllocation:
    """Each cube needs its own slot, or they evict each other."""

    def test_cube_ids_map_to_distinct_slots(self) -> None:
        slots = [CubeManager._slot_for_cube_id(cube_id) for cube_id in (1, 2, 3)]
        assert slots == [0, 1, 2]
        assert len(set(slots)) == 3

    def test_slots_fit_the_robot_slot_count(self) -> None:
        assert all(0 <= CubeManager._slot_for_cube_id(c) < CubeManager._SLOT_COUNT for c in (1, 2, 3))
