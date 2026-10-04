"""Tests for CameraService — the single owner of the camera lifecycle.

Runs against a fake client, so the packet sequence, the settings
vocabulary, and the "echo == what the robot got" guarantee are all
verifiable without a robot.
"""

import pytest

from pycozmo_ng.camera.service import CameraService
from pycozmo_ng.protocol import protocol_encoder


class FakeCli:
    """Records camera-related calls in order."""

    def __init__(self):
        self.calls = []

    def enable_camera(self, enable=True, color=False, resolution=None):
        self.calls.append(("enable_camera", enable, color, resolution))

    def set_camera_params(self, gain=0.0, exposure_ms=0, auto_exposure_enabled=False):
        self.calls.append(
            ("set_camera_params", gain, exposure_ms, auto_exposure_enabled)
        )

    def set_head_light(self, enable):
        self.calls.append(("set_head_light", enable))


def make_service() -> CameraService:
    service = CameraService()
    service.cli = FakeCli()
    return service


class TestEnable:
    def test_sends_the_one_packet_sequence(self):
        service = make_service()
        service.enable()
        assert service.cli.calls == [
            (
                "enable_camera",
                True,
                False,
                protocol_encoder.ImageResolution.QVGA,
            ),
            ("set_camera_params", 0.0, 20, True),
            ("set_head_light", False),
        ]

    def test_records_defaults_as_in_force(self):
        service = make_service()
        service.enable()
        assert service.settings == CameraService.DEFAULTS
        assert service.active is True
        assert service.enabled_at is not None
        assert service.resolution == protocol_encoder.ImageResolution.QVGA

    def test_starts_inactive(self):
        service = make_service()
        assert service.active is False
        assert service.enabled_at is None
        assert service.settings == CameraService.DEFAULTS

    def test_overrides_layer_on_standard_config(self):
        service = make_service()
        service.enable(exposure_ms=30)
        assert ("set_camera_params", 0.0, 30, True) in service.cli.calls
        assert service.settings["exposure_ms"] == 30
        # Untouched settings stay at the standard configuration.
        assert service.settings["gain"] == CameraService.DEFAULTS["gain"]

    def test_resolution_accepts_enum(self):
        service = make_service()
        service.enable(protocol_encoder.ImageResolution.VGA)
        assert service.settings["resolution"] == "VGA"
        assert service.resolution == protocol_encoder.ImageResolution.VGA

    def test_resolution_accepts_name(self):
        service = make_service()
        service.enable("QQVGA")
        assert service.settings["resolution"] == "QQVGA"
        assert service.resolution == protocol_encoder.ImageResolution.QQVGA

    def test_unknown_resolution_falls_back_to_qvga(self):
        service = make_service()
        service.enable("HD")
        assert service.settings["resolution"] == "QVGA"

    def test_without_client_raises(self):
        service = CameraService()
        with pytest.raises(RuntimeError):
            service.enable()


class TestDisable:
    def test_turns_stream_off_and_keeps_state(self):
        service = make_service()
        service.enable("VGA")
        service.cli.calls.clear()
        service.disable()
        assert service.cli.calls == [
            (
                "enable_camera",
                False,
                False,
                protocol_encoder.ImageResolution.VGA,
            )
        ]
        assert service.active is False
        # Resolution and settings survive so a later enable matches.
        assert service.settings["resolution"] == "VGA"

    def test_without_client_raises(self):
        service = CameraService()
        with pytest.raises(RuntimeError):
            service.disable()


class TestApplySettings:
    def test_partial_update_keeps_other_settings(self):
        service = make_service()
        service.enable()
        service.apply_settings({"gain": 0.5, "color": True})
        service.apply_settings({"color": False})
        assert service.settings["gain"] == 0.5
        assert service.settings["color"] is False
        # Untouched standard config survives both updates.
        assert service.settings["exposure_ms"] == CameraService.DEFAULTS["exposure_ms"]

    def test_omitted_keys_keep_current_values(self):
        service = make_service()
        service.apply_settings({"exposure_ms": 75, "head_light": True})
        service.apply_settings({"gain": 1.0})
        assert service.settings["exposure_ms"] == 75
        assert service.settings["head_light"] is True

    def test_unknown_keys_are_dropped(self):
        service = make_service()
        service.apply_settings({"gain": 0.5, "bogus": 1, "exposure": 5})
        assert "bogus" not in service.settings
        assert "exposure" not in service.settings

    def test_types_are_coerced(self):
        service = make_service()
        service.apply_settings({"exposure_ms": 50.7, "gain": "0.25", "color": True})
        assert service.settings["exposure_ms"] == 50
        assert service.settings["gain"] == 0.25
        assert isinstance(service.settings["color"], bool)

    def test_echo_matches_what_the_robot_got(self):
        """The dict returned (and broadcast) is the applied state."""
        service = make_service()
        echoed = service.apply_settings({"gain": 1.5, "resolution": "VGA"})
        assert echoed == service.settings
        assert ("set_camera_params", 1.5, 20, True) in service.cli.calls
        assert (
            "enable_camera",
            True,
            False,
            protocol_encoder.ImageResolution.VGA,
        ) in service.cli.calls
        assert echoed["gain"] == 1.5
        assert echoed["resolution"] == "VGA"

    def test_settings_is_a_copy(self):
        service = make_service()
        settings = service.settings
        settings["gain"] = 99.0
        assert service.settings["gain"] != 99.0

    def test_without_client_raises_and_keeps_state(self):
        service = CameraService()
        with pytest.raises(RuntimeError):
            service.apply_settings({"gain": 1.0})
        assert service.settings == CameraService.DEFAULTS


class TestCalibrationFlow:
    def test_manual_exposure_then_enable_restores_standard_config(self):
        """Robot-centre calibration: manual exposure, then restore."""
        service = make_service()
        service.enable()
        service.apply_settings(
            {"gain": 8.0, "exposure_ms": 300, "auto_exposure_enabled": False}
        )
        assert service.settings["gain"] == 8.0
        assert service.settings["auto_exposure_enabled"] is False

        service.cli.calls.clear()
        service.enable()
        assert ("set_camera_params", 0.0, 20, True) in service.cli.calls
        assert service.settings == CameraService.DEFAULTS
        assert service.active is True
