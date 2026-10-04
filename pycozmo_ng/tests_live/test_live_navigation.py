"""Live navigation tests — public seam, cancel contract, gated motion.

Unless ``PYCOZMO_NG_LIVE_MOTION=1`` is set, nothing here translates the
robot: ``approach_cube_at`` is intercepted (resolution still runs against
real world-map data) and ``drive_to`` aims at the robot's own pose
(distance below the stop tolerance → immediate arrival).
"""

import json
import math

from pycozmo_ng.navigation.navigator import Navigator
from pycozmo_ng.web.msg_router import MsgRouter


def test_stop_without_motion(cozmo):
    cozmo.navigator.stop()  # must not raise; stops already-stationary motors


def test_navigate_cancel_single_none_arg(cozmo):
    """Browser cancel contract: ``listener(None)``.

    The old 2-arg signature raised TypeError which MsgRouter swallowed,
    so cancel never stopped the robot.
    """
    cozmo._brain.navigator.navigate_to(None)  # must not raise


def test_approach_cube_resolves_id(cozmo, monkeypatch):
    nav = cozmo._brain.navigator
    calls = []
    monkeypatch.setattr(
        nav, "approach_cube_at", lambda idx, x, y: calls.append((idx, x, y))
    )

    result = cozmo.navigator.approach_cube(1)

    assert isinstance(result, bool)
    if result:
        idx, x, y = calls[0]
        assert idx == 0  # facade cube ID 1 -> world-map key 0
        assert isinstance(x, float) and isinstance(y, float)
    else:
        assert not calls  # not in the world map -> no approach started


def test_drive_to_current_pose_no_motion(cozmo):
    p = cozmo.cli.pose.position
    assert cozmo.navigator.drive_to(float(p.x), float(p.y)) is True


def test_router_cancel_reaches_real_navigator(cozmo):
    """End-to-end: navigate_cancel message reaches the real Navigator.

    Would fail if the router still swallowed a signature TypeError — the
    navigate job would stay live.
    """
    nav = cozmo._brain.navigator
    router = MsgRouter(world_map=cozmo._brain.world_map)
    router.set_actions(navigate=nav.navigate_to)

    with nav.motion.begin(Navigator.JOB_NAVIGATE) as job:
        router.route(json.dumps({"type": "navigate_cancel"}))
        assert job.cancelled


def test_drive_short_roundtrip(cozmo, motion_ok):
    """MOTION (gated): drive ~150 mm forward and back to the start pose."""
    cli = cozmo.cli
    start = cli.pose.position
    start_x, start_y = float(start.x), float(start.y)
    yaw = cli.pose.rotation.angle_z.radians
    target_x = start_x + 150.0 * math.cos(yaw)
    target_y = start_y + 150.0 * math.sin(yaw)

    assert cozmo.navigator.drive_to(target_x, target_y, timeout_s=30.0) is True
    assert cozmo.navigator.drive_to(start_x, start_y, timeout_s=30.0) is True
