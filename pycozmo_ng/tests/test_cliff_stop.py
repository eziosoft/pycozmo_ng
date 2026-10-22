"""Tests for the cliff-safety wiring (COZMO-74).

Two layers protect the robot from driving off an edge:

1. Firmware level — ``Client._initialize_robot()`` sends ``EnableStopOnCliff``
   (0x60) so the robot's ProxSensors module (body firmware) stops the wheels
   at the moment the cliff sensors read a drop. ``stop_on_cliff=False``
   disables the packet's enable flag instead (the message still goes out so
   the firmware state is deterministic rather than booted blindly).
2. Brain level — ``EvtCliffDetectedChange`` (raised by the ``CLIFF_DETECTED``
   status bit in ``RobotState``) makes the Brain cancel every motion job via
   ``Navigator.stop()``, so Python-side steering stops commanding wheels
   against a stopped robot.
"""

from unittest.mock import MagicMock

from pycozmo_ng import robot
from pycozmo_ng.client import Client
from pycozmo_ng.event import EvtCliffDetectedChange
from pycozmo_ng.protocol import protocol_encoder


def _collect_init_packets(client: Client) -> list:
    """Run _initialize_robot() against a mocked connection, return packets."""
    client.conn = MagicMock()
    client._initialize_robot()
    return [call.args[0] for call in client.conn.send.call_args_list]


def _sent_stop_on_cliff(packets: list) -> protocol_encoder.EnableStopOnCliff | None:
    sent = [p for p in packets if isinstance(p, protocol_encoder.EnableStopOnCliff)]
    assert len(sent) == 1, f"expected exactly one EnableStopOnCliff, got {len(sent)}"
    return sent[0]


class TestFirmwareStopOnCliff:
    def test_initialize_sends_enable_stop_on_cliff(self):
        cli = Client()

        sent = _sent_stop_on_cliff(_collect_init_packets(cli))

        assert sent.enable is True

    def test_initialize_honors_stop_on_cliff_false(self):
        cli = Client(stop_on_cliff=False)

        sent = _sent_stop_on_cliff(_collect_init_packets(cli))

        assert sent.enable is False


class TestCliffEventDispatch:
    def test_cliff_detected_status_bit_raises_event(self):
        """CLIFF_DETECTED going true in RobotState.status raises EvtCliffDetectedChange(True)."""
        cli = Client()
        received: list[bool] = []
        cli.add_handler(EvtCliffDetectedChange, lambda c, state: received.append(state))

        cli._on_robot_state(
            None,
            protocol_encoder.RobotState(
                cliff_data_raw=(0, 0, 0, 0),
                status=robot.RobotStatusFlag.CLIFF_DETECTED,
            ),
        )

        assert received == [True]

    def test_cliff_cleared_raises_false(self):
        cli = Client()
        received: list[bool] = []
        cli.add_handler(EvtCliffDetectedChange, lambda c, state: received.append(state))

        cli._on_robot_state(
            None,
            protocol_encoder.RobotState(
                cliff_data_raw=(0, 0, 0, 0),
                status=robot.RobotStatusFlag.CLIFF_DETECTED,
            ),
        )
        cli._on_robot_state(None, protocol_encoder.RobotState(cliff_data_raw=(0, 0, 0, 0)))

        assert received == [True, False]


class TestBrainCliffAbort:
    def test_rising_edge_stops_navigator(self):
        import pycozmo_ng

        cozmo = pycozmo_ng.Cozmo()
        brain = cozmo._brain
        brain.navigator = MagicMock()

        brain._on_cliff_detected(MagicMock(), True)

        brain.navigator.stop.assert_called_once()

    def test_falling_edge_is_ignored(self):
        import pycozmo_ng

        cozmo = pycozmo_ng.Cozmo()
        brain = cozmo._brain
        brain.navigator = MagicMock()

        brain._on_cliff_detected(MagicMock(), False)

        brain.navigator.stop.assert_not_called()
