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
from unittest.mock import PropertyMock, patch

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
        for key in ("pose", "carry", "navigation", "sensors", "world"):
            assert key in state
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
        assert result == {"ok": True, "count": 0, "detections": []}


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
        # Docstrings double as tool descriptions.
        assert all(tool.description for tool in tools)
