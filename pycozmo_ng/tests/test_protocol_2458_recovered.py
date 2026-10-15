"""Payloads whose 2458 wire shape differs from the original 2381 declaration.

Two packets are declared with no fields in 2381, so every send was shorter than
the robot expects and the message was dropped.  Firmware 2458 carries 13 bytes
for ``animTurnToRecordedHeading`` (six int16 + one uint8) and 32 bytes for
``shutdownRobot``.  These tests pin that shape so a regeneration of
``protocol_encoder.py`` cannot silently reset them to zero length again.

Neither packet is used by the library; this is spec correctness only.  See
``docs/protocol.md``.
"""

import pytest

from pycozmo_ng.protocol.protocol_encoder import (
    PACKETS_BY_ID,
    ShutdownRobot,
    TurnToRecordedHeading,
)

# Fixed-length array fields take no default (as with CubeLights), so every
# construction names its payload explicitly.
SIX_INT16 = (0,) * 6
THIRTY_TWO_BYTES = (0,) * 32


class TestTurnToRecordedHeadingLayout:
    def test_is_command_0x92(self) -> None:
        assert TurnToRecordedHeading(unknown=SIX_INT16).id == 0x92
        assert PACKETS_BY_ID[0x92] is TurnToRecordedHeading

    def test_is_thirteen_bytes(self) -> None:
        assert len(TurnToRecordedHeading(unknown=SIX_INT16).to_bytes()) == 13

    def test_encodes_six_int16_then_a_uint8(self) -> None:
        packet = TurnToRecordedHeading(unknown=(1, -2, 3, 4, 5, 6), unknown2=9)
        assert packet.to_bytes() == bytes([1, 0, 0xFE, 0xFF, 3, 0, 4, 0, 5, 0, 6, 0, 9])

    def test_is_not_still_empty(self) -> None:
        """The 2381 declaration encoded nothing; 2458 reads 13 bytes."""
        assert TurnToRecordedHeading(unknown=SIX_INT16).to_bytes() != b""

    def test_round_trips(self) -> None:
        original = TurnToRecordedHeading(unknown=(0, -1, 2, -3, 4, -5), unknown2=255)
        decoded = TurnToRecordedHeading.from_bytes(original.to_bytes())
        assert tuple(decoded.unknown) == (0, -1, 2, -3, 4, -5)
        assert decoded.unknown2 == 255

    def test_ranges(self) -> None:
        assert TurnToRecordedHeading(unknown=(32767, -32768, 0, 0, 0, 0), unknown2=255).unknown2 == 255
        with pytest.raises(ValueError):
            TurnToRecordedHeading(unknown=(32768, 0, 0, 0, 0, 0))
        with pytest.raises(ValueError):
            TurnToRecordedHeading(unknown=SIX_INT16, unknown2=256)

    def test_requires_the_full_six_int16_field(self) -> None:
        with pytest.raises(ValueError):
            TurnToRecordedHeading(unknown=(1, 2, 3))


class TestShutdownRobotLayout:
    def test_is_command_0xa9(self) -> None:
        assert ShutdownRobot(unknown=THIRTY_TWO_BYTES).id == 0xA9
        assert PACKETS_BY_ID[0xA9] is ShutdownRobot

    def test_is_thirty_two_bytes(self) -> None:
        assert len(ShutdownRobot(unknown=THIRTY_TWO_BYTES).to_bytes()) == 32

    def test_is_not_still_empty(self) -> None:
        assert ShutdownRobot(unknown=THIRTY_TWO_BYTES).to_bytes() != b""

    def test_round_trips(self) -> None:
        original = ShutdownRobot(unknown=tuple(range(32)))
        decoded = ShutdownRobot.from_bytes(original.to_bytes())
        assert tuple(decoded.unknown) == tuple(range(32))

    def test_range(self) -> None:
        assert ShutdownRobot(unknown=tuple([255] + [0] * 31)).unknown[0] == 255
        with pytest.raises(ValueError):
            ShutdownRobot(unknown=tuple([256] + [0] * 31))
