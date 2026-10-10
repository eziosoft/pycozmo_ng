"""Tests for the MCP adapter (COZMO-68): jobs, safety limits, and the Toolset.

The Toolset contract is that every tool talks to the public ``Cozmo``
API and returns JSON-able data — these tests run against a real
``Cozmo()`` with no robot, like ``test_cozmo_facade``. Motion tools
are exercised through their refusal paths (not connected, extent,
confirmation) plus the job plumbing; the ``mcp`` SDK itself is only
needed for the server test, which skips without it.
"""

import json
import time
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from pycozmo_ng import Cozmo
from pycozmo_ng.mcp import JobRegistry, SafetyLimits, Toolset
from pycozmo_ng.mcp.tools import jsonable
from pycozmo_ng.vision.cube_detector.detection_pipeline import CubeDetection


@pytest.fixture
def cozmo() -> Cozmo:
    return Cozmo()


@pytest.fixture
def toolset(cozmo) -> Toolset:
    return Toolset(cozmo)


def _connected(cozmo: Cozmo):
    """Patched 'connected' Cozmo — subsystem calls still hit real objects."""
    return patch.object(Cozmo, "is_connected", new_callable=PropertyMock, return_value=True)


class TestJsonable:
    def test_numpy_round_trip(self):
        assert jsonable(np.float64(1.5)) == 1.5
        assert jsonable(np.array([1.0, 2.0])) == [1.0, 2.0]
        assert jsonable({"a": np.int64(3)}) == {"a": 3}

    def test_cube_detection_dataclass(self):
        det = CubeDetection(x=1, y=2, width=3, height=4, confidence=0.9, distance=0.4, cube_index=1)
        out = jsonable(det)
        assert out["cube_index"] == 1
        assert out["confidence"] == 0.9

    def test_unknown_object_falls_back_to_str(self):
        assert jsonable(object()).startswith("<object")


class TestSafetyLimits:
    def test_target_inside_extent_is_allowed(self):
        assert SafetyLimits(extent_mm=2000.0).target_error(1500.0, -100.0) is None

    def test_target_beyond_extent_is_rejected(self):
        reason = SafetyLimits(extent_mm=2000.0).target_error(3000.0, 0.0)
        assert reason is not None
        assert "extent limit" in reason

    def test_confirmation_gate(self):
        strict = SafetyLimits(require_confirmation=True)
        assert strict.confirmation_error(confirm=False) is not None
        assert strict.confirmation_error(confirm=True) is None
        assert SafetyLimits().confirmation_error(confirm=False) is None

    def test_clamp_speed(self):
        limits = SafetyLimits(speed_mmps=150.0)
        assert limits.clamp_speed(250.0) == 150.0
        assert limits.clamp_speed(60.0) == 60.0
        assert limits.clamp_speed(-10.0) == 0.0


class TestJobRegistry:
    def test_submit_and_wait(self):
        reg = JobRegistry()
        job_id = reg.submit("t", lambda: {"reached": True})
        entry = reg.wait(job_id, timeout_s=5.0)
        assert entry is not None
        assert entry["status"] == "done"
        assert entry["result"] == {"reached": True}
        assert entry["finished_at"] is not None

    def test_error_captured(self):
        reg = JobRegistry()
        job_id = reg.submit("boom", lambda: 1 / 0)
        entry = reg.wait(job_id, timeout_s=5.0)
        assert entry["status"] == "error"
        assert entry["error"] == "division by zero"

    def test_wait_timeout_returns_running(self):
        reg = JobRegistry()
        job_id = reg.submit("slow", lambda: time.sleep(1.0))
        entry = reg.wait(job_id, timeout_s=0.05)
        assert entry["status"] == "running"
        reg.wait(job_id, timeout_s=5.0)  # let the thread finish

    def test_unknown_job(self):
        reg = JobRegistry()
        assert reg.get("nope") is None
        assert reg.wait("nope") is None
        assert reg.cancel("nope") is False

    def test_cancel_flag(self):
        reg = JobRegistry()
        job_id = reg.submit("t", lambda: "x")
        reg.wait(job_id, timeout_s=5.0)
        assert reg.cancel(job_id) is True
        assert reg.get(job_id)["cancelled"] is True

    def test_list_newest_first(self):
        reg = JobRegistry()
        first = reg.submit("a", lambda: None)
        time.sleep(0.01)
        second = reg.submit("b", lambda: None)
        reg.wait(first, 5.0)
        reg.wait(second, 5.0)
        listed = reg.list()
        assert listed[0]["job_id"] == second
        assert {entry["name"] for entry in listed} == {"a", "b"}


class TestReads:
    def test_get_robot_state_shape(self, toolset):
        state = toolset.get_robot_state()
        assert state["ok"] is True
        assert state["connected"] is False
        for key in ("pose", "carry", "vision", "cubes_connected", "navigation", "jobs_running", "sensors", "world"):
            assert key in state
        # Orientation extras: camera freshness, BLE links, running jobs.
        assert state["vision"]["camera_active"] is False
        assert state["vision"]["frame_stale"] is True
        assert state["cubes_connected"] == []
        assert state["jobs_running"] == 0
        # Lean snapshot: trails excluded.
        assert "path_points" not in state["world"]
        json.dumps(state)  # must be JSON-serializable

    def test_get_sensors_defaults_when_disconnected(self, toolset):
        sensors = toolset.get_sensors()
        assert sensors["ok"] is True
        assert sensors["connected"] is False
        assert sensors["sensors"]["battery_voltage"] == 0.0
        json.dumps(sensors)

    def test_get_world_map_remaps_cube_ids(self, toolset, cozmo):
        wm = cozmo._brain.world_map
        wm.update_robot_pose([0.0, 0.0, 0.0], 0.0, origin_id=1, record_path=False)
        wm.update_cube_measurement(0, [100.0, 0.0, 200.0], np.eye(3), 0.9, 0.0)
        world = toolset.get_world_map()["world"]
        assert "1" in world["cubes"]  # map index 0 -> public ID 1
        assert "path_points" in world
        json.dumps(world)

    def test_get_detections_empty_without_vision(self, toolset):
        result = toolset.get_detections()
        assert result == {
            "ok": True,
            "count": 0,
            "camera_active": False,
            "frame_age_s": None,
            "frame_stale": True,
            "detections": [],
        }


class TestMotionRefusals:
    def test_not_connected(self, toolset):
        result = toolset.navigate_to(100.0, 0.0)
        assert result["ok"] is False
        assert "Not connected" in result["reason"]

    def test_extent_rejected_when_connected(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.navigate_to(5000.0, 0.0)
        assert result["ok"] is False
        assert "extent limit" in result["reason"]
        assert toolset.jobs.list() == []  # no job was started

    def test_confirmation_required(self, cozmo):
        toolset = Toolset(cozmo, limits=SafetyLimits(require_confirmation=True))
        with _connected(cozmo):
            result = toolset.navigate_to(100.0, 0.0)
            assert result["ok"] is False
            assert "confirm" in result["reason"]
            accepted = toolset.navigate_to(100.0, 0.0, confirm=True)
        assert accepted["ok"] is True
        entry = toolset.jobs.wait(accepted["job_id"], timeout_s=5.0)
        assert entry["status"] == "done"
        assert entry["result"]["target_mm"] == [100.0, 0.0]

    def test_cube_id_validation(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.approach_cube(7)
        assert result["ok"] is False
        assert "1, 2 or 3" in result["reason"]

    def test_say_rejects_empty_text(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.say("   ")
        assert result["ok"] is False


class TestStop:
    def test_stop_works_disconnected(self, toolset):
        result = toolset.stop()
        assert result == {"ok": True, "cancelled_jobs": []}


class TestRelativeMotion:
    """move_by / turn_by — the small positioning moves world-point navigate cannot express."""

    def test_move_by_not_connected(self, toolset):
        assert toolset.move_by(100.0)["ok"] is False

    def test_move_by_rejects_tiny_distance(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.move_by(0.1)
        assert result["ok"] is False
        assert "at least 1" in result["reason"]

    def test_move_by_extent_rejected(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.move_by(5000.0)
        assert result["ok"] is False
        assert "extent" in result["reason"]
        assert toolset.jobs.list() == []  # refusal starts no job

    def test_move_by_submits_job_and_clamps_speed(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.move_by(100.0, speed_mmps=999.0)
        assert result["ok"] is True
        assert result["speed_mmps"] == toolset.limits.speed_mmps  # safety cap applied
        assert result["target_mm"][0] == pytest.approx(100.0, abs=0.2)
        entry = toolset.jobs.wait(result["job_id"], timeout_s=5.0)
        assert entry["status"] == "done"
        assert entry["result"]["requested_mm"] == 100.0
        assert entry["result"]["arrived"] is False  # no robot to drive
        # ...and the failure says why instead of just "false".
        assert "reason" in entry["result"]

    def test_turn_by_not_connected(self, toolset):
        assert toolset.turn_by(90.0)["ok"] is False

    def test_turn_by_rejects_full_spins(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.turn_by(720.0)
        assert result["ok"] is False
        assert "360" in result["reason"]
        assert toolset.jobs.list() == []

    def test_turn_by_submits_job(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.turn_by(-45.0)
        assert result["ok"] is True
        entry = toolset.jobs.wait(result["job_id"], timeout_s=5.0)
        assert entry["status"] == "done"
        assert entry["result"]["requested_deg"] == -45.0
        assert entry["result"]["aligned"] is False  # no robot to turn
        assert "reason" in entry["result"]


class TestStackCube:
    def test_not_connected(self, toolset):
        assert toolset.stack_cube(1)["ok"] is False

    def test_cube_id_validation(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.stack_cube(7)
        assert result["ok"] is False
        assert "1, 2 or 3" in result["reason"]

    def test_refused_without_cube_on_forks(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.stack_cube(2)
        assert result["ok"] is False
        assert toolset.jobs.list() == []  # refusal starts no job


class TestDrivableGate:
    """Only a picked-up robot is refused — the firmware holds its wheels then.

    A robot on the charger is *not* blocked: it simply drives off, so
    the charger never gates a drive tool.
    """

    def test_picked_up_blocks_turn(self, cozmo, toolset):
        cozmo._brain.cli = MagicMock()  # sensors.picked_up reads the client
        with _connected(cozmo):
            result = toolset.turn_by(90.0)
        assert result["ok"] is False
        assert "picked up" in result["reason"]
        assert toolset.jobs.list() == []  # refused before any job started

    def test_picked_up_blocks_move(self, cozmo, toolset):
        cozmo._brain.cli = MagicMock()
        with _connected(cozmo):
            result = toolset.move_by(100.0)
        assert result["ok"] is False
        assert "picked up" in result["reason"]

    def test_picked_up_blocks_cube_visits_and_dock(self, cozmo, toolset):
        cozmo._brain.cli = MagicMock()
        with _connected(cozmo):
            assert toolset.approach_cube(1)["ok"] is False
            assert toolset.stack_cube(2)["ok"] is False
            assert toolset.dock()["ok"] is False

    def test_on_charger_does_not_block_driving(self, cozmo, toolset):
        """The robot drives off the charger — only pickup holds the wheels."""
        cozmo._brain.cli = None  # sensors fall back to DockState
        cozmo._brain.dock_state.on_charger = True
        cozmo._brain.dock_state.charging = True
        with _connected(cozmo):
            assert toolset._drivable_refusal() is None
            assert toolset.turn_by(90.0)["ok"] is True
            assert toolset.move_by(100.0)["ok"] is True

    def test_say_is_not_gated_by_pickup(self, cozmo, toolset):
        """Head/lift/say have no wheels — only drive tools consult the gate."""
        cozmo._brain.cli = MagicMock()
        cozmo._cli = MagicMock()
        with _connected(cozmo):
            result = toolset.say("hello")
        assert result["ok"] is True

    def test_failure_reason_without_wheel_response(self, toolset):
        assert "did not respond" in toolset._motion_failure_reason(made_progress=False)

    def test_failure_reason_when_it_moved_but_missed(self, toolset):
        assert "retry" in toolset._motion_failure_reason(made_progress=True)

    def test_failure_reason_names_pickup(self, cozmo, toolset):
        cozmo._brain.cli = MagicMock()
        assert "picked up" in toolset._motion_failure_reason(made_progress=True)


class TestCameraToggle:
    def test_not_connected(self, toolset):
        assert toolset.camera_toggle(False)["ok"] is False

    def test_disable_when_already_off_is_a_noop(self, cozmo, toolset):
        with _connected(cozmo):
            result = toolset.camera_toggle(False)
        assert result == {"ok": True, "camera_active": False}

    def test_enable_without_camera_client_is_refused(self, cozmo, toolset):
        # is_connected says yes but the camera service has no client to
        # send the enable to — must surface as a refusal, never a raise.
        with _connected(cozmo):
            result = toolset.camera_toggle(True)
        assert result["ok"] is False
        assert result["reason"]


class TestWait:
    def test_sleeps_up_to_the_request(self, toolset):
        result = toolset.wait(0.05)
        assert result["ok"] is True
        assert result["slept_s"] >= 0.04

    def test_negative_request_returns_immediately(self, toolset):
        result = toolset.wait(-5.0)
        assert result["ok"] is True
        assert result["slept_s"] < 1.0

    def test_sleep_cap_stays_under_client_timeouts(self):
        from pycozmo_ng.mcp.tools import MAX_SLEEP_S

        assert MAX_SLEEP_S == 10.0


class TestJobTools:
    def test_wait_for_job_unknown_id(self, toolset):
        result = toolset.wait_for_job("missing")
        assert result["ok"] is False

    def test_wait_for_job_finished(self, toolset):
        job_id = toolset.jobs.submit("quick", lambda: 42)
        result = toolset.wait_for_job(job_id, timeout_s=5.0)
        assert result["finished"] is True
        assert result["job"]["result"] == 42


class TestServer:
    def test_registration_and_lifecycle(self, cozmo):
        pytest.importorskip("mcp")
        from pycozmo_ng.mcp.server import McPServer

        server = McPServer(cozmo, port=18791)
        assert server.is_running is False
        assert server.url.endswith("/mcp")

        import asyncio

        from mcp.server.mcpserver import MCPServer as SdkServer

        sdk = SdkServer(name="test")
        server._register_tools(sdk)
        tools = asyncio.run(sdk.list_tools())
        names = {tool.name for tool in tools}
        assert "navigate_to" in names
        assert "get_robot_state" in names
        assert "stop" in names
        for added in ("move_by", "turn_by", "stack_cube", "camera_toggle", "wait"):
            assert added in names
        # Docstrings double as tool descriptions.
        assert all(tool.description for tool in tools)
        # Reads advertise themselves read-only (clients may auto-approve);
        # motion does not.
        hints = {tool.name: tool.annotations.read_only_hint for tool in tools}
        assert hints["get_sensors"] is True
        assert hints["navigate_to"] is False
        assert hints["move_by"] is False


class TestGetFrameImageContent:
    """get_frame is repackaged by the server wrapper: a populated frame must
    leave the MCP endpoint as real image content, not a base64 JSON string."""

    def _server(self, cozmo):
        from mcp.server.mcpserver import MCPServer as SdkServer

        from pycozmo_ng.mcp.server import McPServer

        sdk = SdkServer(name="test")
        McPServer(cozmo)._register_tools(sdk)
        return sdk

    def test_frame_returns_image_content_plus_summary(self, cozmo):
        pytest.importorskip("mcp")
        import asyncio
        import base64
        import io

        from mcp_types import ImageContent, TextContent
        from PIL import Image as PILImage

        cozmo._brain.frame_state.latest_raw_frame = PILImage.new("RGB", (64, 48), (255, 0, 0))
        sdk = self._server(cozmo)
        result = asyncio.run(sdk.call_tool("get_frame", {}))
        assert result.is_error is False
        blocks = result.content
        images = [b for b in blocks if isinstance(b, ImageContent)]
        texts = [b for b in blocks if isinstance(b, TextContent)]
        assert len(images) == 1
        assert images[0].data  # non-empty base64 payload
        assert PILImage.open(io.BytesIO(base64.b64decode(images[0].data))).size == (64, 48)
        assert len(texts) == 1
        assert "64x48" in texts[0].text
        assert "timestamp" in texts[0].text

    def test_no_frame_returns_text_snapshot(self, cozmo):
        pytest.importorskip("mcp")
        import asyncio

        from mcp_types import ImageContent, TextContent

        sdk = self._server(cozmo)
        result = asyncio.run(sdk.call_tool("get_frame", {}))
        assert result.is_error is False
        assert all(isinstance(b, TextContent) for b in result.content)
        assert not any(isinstance(b, ImageContent) for b in result.content)

    def test_frame_reports_capture_age_and_staleness(self, cozmo):
        """A stopped camera must not present an old frame as current."""
        import time

        from PIL import Image as PILImage

        frame_state = cozmo._brain.frame_state
        frame_state.latest_raw_frame = PILImage.new("RGB", (8, 8))
        cozmo._brain.camera._active = True  # stream on — the fresh case
        toolset = Toolset(cozmo)

        frame_state.last_frame_time = time.time()
        fresh = toolset.get_frame()["frame"]
        assert fresh["stale"] is False
        assert fresh["age_s"] < 2.0

        frame_state.last_frame_time = time.time() - 30.0
        stale = toolset.get_frame()["frame"]
        assert stale["stale"] is True
        assert stale["age_s"] >= 30.0

    def test_stopped_camera_marks_even_the_newest_frame_stale(self, cozmo):
        """camera_toggle(false) freezes the last frame — it is history, not live."""
        import time

        from PIL import Image as PILImage

        frame_state = cozmo._brain.frame_state
        frame_state.latest_raw_frame = PILImage.new("RGB", (8, 8))
        frame_state.last_frame_time = time.time()  # captured a moment ago...
        cozmo._brain.camera._active = False  # ...but the stream is off
        toolset = Toolset(cozmo)

        assert toolset.get_frame()["frame"]["stale"] is True
        assert toolset.get_detections()["frame_stale"] is True

    def test_stale_frame_is_called_out_in_the_text_block(self, cozmo):
        pytest.importorskip("mcp")
        import asyncio
        import time

        from mcp_types import TextContent
        from PIL import Image as PILImage

        frame_state = cozmo._brain.frame_state
        frame_state.latest_raw_frame = PILImage.new("RGB", (8, 8))
        frame_state.last_frame_time = time.time() - 30.0
        sdk = self._server(cozmo)
        result = asyncio.run(sdk.call_tool("get_frame", {}))
        texts = [b for b in result.content if isinstance(b, TextContent)]
        assert any("STALE" in block.text for block in texts)
        assert any("camera_toggle" in block.text for block in texts)
