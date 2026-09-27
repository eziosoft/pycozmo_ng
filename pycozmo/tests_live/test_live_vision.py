"""Live vision tests — calibration retrieval, frame flow, handler registration.

Covers the paths that only exist with a real connection: Brain.setup_vision
retrieving calibration from robot NVRAM and registering the subsystem event
handlers (candidate-1 refactor moved both here from the facade).
"""

from pycozmo.tests_live.conftest import wait_for


def test_vision_start_retrieves_calibration(vision):
    assert vision.camera_calibration is not None
    assert vision.detection_pipeline is not None


def test_camera_frames_flow(vision):
    assert wait_for(lambda: vision.latest_frame is not None, timeout=10.0), (
        "no camera frame within 10 s"
    )


def test_battery_reports_robot_state(cozmo):
    assert wait_for(lambda: cozmo.sensors.battery_voltage > 0, timeout=5.0), (
        "no robot state within 5 s"
    )
    assert 0.0 < cozmo.sensors.battery_voltage < 5.0


def test_robot_state_trail_handler_registered(vision, cozmo):
    """Brain.setup_vision registers Brain._on_robot_state."""
    if not wait_for(lambda: len(cozmo.world_map.path) > 0, timeout=5.0):
        # Robot may only send state on change — jiggle the head to force packets.
        cozmo.set_head_angle(-20)
    assert wait_for(lambda: len(cozmo.world_map.path) > 0, timeout=5.0), (
        "trail handler never recorded a pose"
    )


def test_vision_stop_turns_camera_off(vision, cozmo):
    """Must stay the LAST test in this file: stops the stream for the session.

    The ``vision`` fixture teardown stops again (idempotent).
    """
    vision.stop()
    assert cozmo._brain.frame_state.camera_active is False
