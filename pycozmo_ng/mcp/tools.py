"""Toolset — MCP tool implementations over the public ``Cozmo`` API (COZMO-68).

One method per MCP tool; ``McPServer`` registers them by name. The
toolset is a pure adapter: every tool talks to the public
:class:`~pycozmo_ng.Cozmo` facade (subsystem properties and methods
documented in ``cozmo.py``), never to ``Brain`` or ``Client``
internals — extending the public surface was agreed before building
(Q9b), so ``SensorsAPI``, ``WorldMapAPI``, ``NavigatorAPI`` and
``VisionAPI`` gained the read-only members this layer needs.

Conventions (also stated in the MCP server instructions):

* **Frames/units**: world frame, millimetres, radians; cube IDs are
  1-3 everywhere (the world-map 0-2 index never leaks out).
* **Reads are synchronous** dicts, JSON-able by construction.
* **Motion tools return a job id** — the command keeps running in the
  background; poll ``get_job``/``wait_for_job``. Refusals (safety,
  not connected, confirmation required) come back as
  ``{"ok": false, "reason": ...}`` *without* starting a job.
* **``stop`` is always available** and cancels both motion and jobs.

This module deliberately does not import the ``mcp`` SDK — it stays
testable with no robot and no optional dependencies.
"""

from __future__ import annotations

import base64
import io
import math
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.logger import logger_mcp
from pycozmo_ng.mcp.jobs import JobRegistry
from pycozmo_ng.mcp.safety import SafetyLimits
from pycozmo_ng.util.robot import MAX_HEAD_ANGLE, MAX_LIFT_HEIGHT, MIN_HEAD_ANGLE, MIN_LIFT_HEIGHT

if TYPE_CHECKING:
    from pycozmo_ng.cozmo import Cozmo

#: How long ``wait_for_job`` may block by default (seconds). Kept under
#: typical MCP client tool timeouts so the call itself never looks hung.
DEFAULT_WAIT_S = 30.0

#: Hard cap for a requested wait (seconds).
MAX_WAIT_S = 300.0


def jsonable(value: Any) -> Any:
    """Best-effort conversion of library values to JSON-able data.

    Handles numpy scalars/arrays, dataclasses (``CubeDetection``),
    enums (``CarryState``), ``Vector3``/``Angle``/``Distance`` and
    tuples; anything else falls back to ``str()`` so a tool call can
    never fail on serialization.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if isinstance(value, Enum):
        return jsonable(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    # Vector3 / Angle / Distance and friends expose typed accessors.
    for attr in ("x_y_z", "radians", "mm"):
        if hasattr(value, attr):
            member = getattr(value, attr)
            return jsonable(member() if callable(member) else member)
    return str(value)


class Toolset:
    """The MCP tool surface — one public method per tool.

    Args:
        cozmo: The public API instance the tools drive.
        limits: Safety caps on motion (defaults to ``SafetyLimits()``).
        jobs: Job registry for background commands (a private one is
            created when omitted — share one to inspect jobs from tests).
    """

    def __init__(
        self,
        cozmo: Cozmo,
        limits: SafetyLimits | None = None,
        jobs: JobRegistry | None = None,
    ) -> None:
        self._cozmo = cozmo
        self.limits = limits or SafetyLimits()
        self.jobs = jobs or JobRegistry()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refusal(self, reason: str) -> dict[str, Any]:
        logger_mcp.info("MCP tool refused: %s", reason)
        return {"ok": False, "reason": reason}

    def _require_connected(self) -> dict[str, Any] | None:
        if not self._cozmo.is_connected:
            return self._refusal("Not connected to the robot")
        return None

    def _motion_precheck(self, confirm: bool) -> dict[str, Any] | None:
        """Common gate for every motion tool: connection + confirmation."""
        refusal = self._require_connected()
        if refusal is not None:
            return refusal
        confirm_err = self.limits.confirmation_error(confirm)
        if confirm_err is not None:
            return self._refusal(confirm_err)
        return None

    def _submit_visit(self, name: str, started: bool, job_name: str) -> dict[str, Any]:
        """Shared body for approach/dock: report a refusal, or run a job.

        ``started`` comes from the navigator call already made by the
        caller: a refused visit returns immediately; an accepted one
        becomes a background job that waits for the underlying motion
        job to drain.
        """
        if not started:
            return self._refusal("Visit refused - the target is unknown or another visit is in flight")
        job_id = self.jobs.submit(name, lambda: self._await_motion(job_name))
        return {"ok": True, "job_id": job_id}

    def _await_motion(self, job_name: str) -> dict[str, Any]:
        """Poll until no motion job named *job_name* (or any) remains.

        Cube visits spawn their own motion tokens; we observe completion
        through ``navigator.active_motions`` and give up after the visit
        timeout so an orphaned job cannot spin forever.
        """
        deadline = time.monotonic() + self.limits.visit_timeout_s
        # The spawned motion token may appear a tick after start() returns.
        time.sleep(0.3)
        while time.monotonic() < deadline:
            active = set(self._cozmo.navigator.active_motions)
            if job_name not in active:
                return {"finished": True}
            time.sleep(0.2)
        return {"finished": False, "reason": f"timed out after {self.limits.visit_timeout_s:.0f}s"}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_robot_state(self) -> dict[str, Any]:
        """Merged snapshot: connection status, robot pose, telemetry, carry state, navigation status, and the fused world map (path trails omitted).

        The one call to orient a new turn. Positions are world-frame
        millimetres relative to the last set_origin(); headings radians.
        For the full map including path trails use get_world_map; for
        telemetry only use get_sensors.
        """
        cozmo = self._cozmo
        world = self._world_snapshot(include_trails=False)
        position, yaw = cozmo.world_map.robot_pose
        return {
            "ok": True,
            "connected": cozmo.is_connected,
            "timestamp": time.time(),
            "pose": {"position_mm": jsonable(position), "yaw_rad": jsonable(yaw)},
            "carry": {
                "state": jsonable(cozmo.carry.state),
                "is_carrying": cozmo.carry.is_carrying,
                "cube_id": cozmo.carry.cube_id,
                "on_forks": cozmo.carry.on_forks,
            },
            "navigation": {
                "astar_enabled": cozmo.navigator.is_astar_enabled(),
                "active_motions": list(cozmo.navigator.active_motions),
            },
            "sensors": self.get_sensors().get("sensors", {}),
            "world": world,
        }

    def get_world_map(self) -> dict[str, Any]:
        """Full world map: robot pose, cubes (IDs 1-3), faces, charger, AruCo markers, path trail, A* path.

        World frame, millimetres, rotation matrices row-major. Includes
        the path trails that get_robot_state omits.
        """
        return {"ok": True, "world": self._world_snapshot(include_trails=True)}

    def _world_snapshot(self, include_trails: bool) -> dict[str, Any]:
        """``get_world_state`` remapped to the public conventions.

        Cube keys become public IDs 1-3 (the 0-2 map index never
        escapes); trails are dropped from lean snapshots — they can be
        thousands of points and are one command away.
        """
        world = self._cozmo.world_map.get_world_state()
        cubes = world.get("cubes") or {}
        world["cubes"] = {str(to_cube_id(int(idx))): state for idx, state in cubes.items()}
        if not include_trails:
            world.pop("path_points", None)
            world.pop("astar_path", None)
        return jsonable(world)

    def get_sensors(self) -> dict[str, Any]:
        """Telemetry only: battery voltage, head angle, lift height, wheel speeds, IMU, cliff, charging/on-charger, picked-up/moving.

        Returns zeros/defaults when not connected.
        """
        cozmo = self._cozmo
        return {
            "ok": True,
            "connected": cozmo.is_connected,
            "sensors": jsonable(
                {
                    "battery_voltage": cozmo.sensors.battery_voltage,
                    "head_angle_deg": cozmo.sensors.head_angle,
                    "lift_height_mm": cozmo.sensors.lift_height,
                    "wheel_speeds_mmps": cozmo.sensors.wheel_speeds,
                    "imu": cozmo.sensors.imu,
                    "cliff": cozmo.sensors.cliff,
                    "is_charging": cozmo.sensors.is_charging,
                    "on_charger": cozmo.sensors.on_charger,
                    "picked_up": cozmo.sensors.picked_up,
                    "moving": cozmo.sensors.moving,
                }
            ),
        }

    def get_detections(self) -> dict[str, Any]:
        """Raw detections from the latest processed camera frame: cubes with pixel boxes, confidence, distance, and cube ID (1-3).

        Empty when vision is not running. The fused, world-frame view
        lives in get_world_map.
        """
        detections = self._cozmo.vision.latest_detections
        out = []
        for det in detections:
            item = jsonable(det)
            if isinstance(item, dict) and "cube_index" in item:
                item["cube_id"] = to_cube_id(int(item.pop("cube_index")))
            out.append(item)
        return {"ok": True, "count": len(out), "detections": out}

    def get_frame(
        self,
        annotated: bool = False,
        max_width: int = 0,
        jpeg_quality: int = 70,
    ) -> dict[str, Any]:
        """Latest camera frame as a base64-encoded JPEG, optimised for LLM consumption.

        Args:
            annotated: When true, return the annotated frame (detection boxes,
                labels, confidence, distance drawn on the image) instead of the
                raw camera frame.
            max_width: Downscale the image so its width is at most this many
                pixels (aspect ratio preserved). 0 = original resolution. Use
                320 or 160 for smaller payloads.
            jpeg_quality: JPEG encoding quality (1-95, default 70). Lower
                values produce smaller base64 strings at the cost of fidelity.

        Returns ``{"ok": true, "frame": ...}`` with fields:
        ``jpeg_base64``, ``width``, ``height``, ``timestamp``,
        ``detections_summary``. When the camera is stopped or no frame has
        arrived yet returns ``{"ok": true, "frame": null}``.
        """
        frame = self._cozmo.vision.latest_annotated_frame if annotated else self._cozmo.vision.latest_frame
        if frame is None:
            return {"ok": True, "frame": None}

        # Downscale when max_width is set (preserve aspect ratio).
        img = frame
        if max_width > 0 and frame.width > max_width:
            ratio = max_width / frame.width
            new_size = (max_width, max(1, int(frame.height * ratio)))
            img = frame.resize(new_size, Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        jpeg_q = max(1, min(95, int(jpeg_quality)))
        img.save(buf, format="JPEG", quality=jpeg_q)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        # Build a text summary of what the vision pipeline sees in this frame.
        summary = self._frame_summary()

        return {
            "ok": True,
            "frame": {
                "jpeg_base64": b64,
                "width": img.width,
                "height": img.height,
                "timestamp": time.time(),
                "detections_summary": summary,
            },
        }

    def _frame_summary(self) -> str:
        """Plain-English summary of detections visible in the current frame."""
        parts: list[str] = []

        # Cube detections
        cubes = self._cozmo.vision.latest_detections
        if cubes:
            cube_lines: list[str] = []
            for det in cubes:
                cx = det.x + det.width // 2
                cy = det.y + det.height // 2
                dist_str = f"{det.distance:.1f}m" if det.distance else "unknown"
                cube_lines.append(
                    f"  Cube {det.cube_index + 1} at pixel ({cx}, {cy}), "
                    f"confidence {det.confidence:.2f}, distance {dist_str}"
                )
            parts.append(f"{len(cubes)} cube(s) detected:")
            parts.extend(cube_lines)
        else:
            parts.append("No cubes detected in frame.")

        # Faces, charger, ArUco from the world-map snapshot
        try:
            world = self._cozmo.world_map.get_world_state()
        except RuntimeError:
            world = {}

        faces = world.get("faces") or {}
        if faces:
            face_names = ", ".join(f"Face {fid}" for fid in sorted(faces.keys()))
            parts.append(f"Face(s) visible: {face_names}.")

        charger = world.get("charger")
        if charger and isinstance(charger, dict) and charger.get("position"):
            parts.append("Charger marker visible.")

        arucos = world.get("aruco_markers") or {}
        if arucos:
            aruco_desc = ", ".join(f"ArUco #{mid}" for mid in sorted(arucos.keys()))
            parts.append(f"ArUco marker(s): {aruco_desc}.")

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Motion (job-returning)
    # ------------------------------------------------------------------

    def navigate_to(self, x: float, y: float, confirm: bool = False) -> dict[str, Any]:
        """Drive to a world-frame point (mm), following the A* path when enabled. Returns a job_id.

        Poll get_job/wait_for_job for {arrived, distance_mm, timed_out}.
        Rejected beyond the safety extent from the origin. Pass
        confirm=true when confirmation is required.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        reason = self.limits.target_error(x, y)
        if reason is not None:
            return self._refusal(reason)
        job_id = self.jobs.submit("navigate_to", lambda: self._run_navigate(x, y))
        return {"ok": True, "job_id": job_id}

    def _run_navigate(self, x: float, y: float) -> dict[str, Any]:
        """Start ``navigator.navigate_to`` and watch the navigate job drain.

        Arrival is measured against the world-map pose (the navigator's
        own result is only a log line); a straggler past the motion
        timeout is cancelled via navigate-cancel, not ``stop()`` — an
        unrelated approach or dock must survive its sibling timing out.
        """
        cozmo = self._cozmo
        cozmo.navigator.navigate_to(x, y)
        deadline = time.monotonic() + self.limits.motion_timeout_s
        timed_out = False
        while time.monotonic() < deadline:
            if "navigate" not in cozmo.navigator.active_motions:
                break
            time.sleep(0.2)
        else:
            timed_out = True
            cozmo.navigator.cancel_navigation()
        position, _yaw = cozmo.world_map.robot_pose
        distance = math.hypot(x - float(position[0]), y - float(position[1]))
        return {
            "arrived": distance <= 25.0 and not timed_out,
            "distance_mm": round(distance, 1),
            "timed_out": timed_out,
            "target_mm": [x, y],
        }

    def approach_cube(self, cube_id: int, confirm: bool = False) -> dict[str, Any]:
        """Drive to cube 1-3, align, and grab it (stacks the held cube onto it when one is on the forks).

        Returns a job_id when the visit starts; refused (ok=false) when
        the cube is unknown or another visit is in flight.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        if cube_id not in (1, 2, 3):
            return self._refusal(f"cube_id must be 1, 2 or 3 (got {cube_id})")
        return self._submit_visit(
            "approach_cube",
            self._cozmo.navigator.approach_cube(cube_id),
            "approach",
        )

    def dock(self, confirm: bool = False) -> dict[str, Any]:
        """Dock on the charger AruCo marker 0: drive to it, refresh pose, reverse into it until charging.

        Returns a job_id when the attempt starts; ok=false when the
        marker is not in the world map.
        """
        return self.dock_on_aruco(marker_id=0, confirm=confirm)

    def dock_on_aruco(self, marker_id: int = 0, confirm: bool = False) -> dict[str, Any]:
        """Dock on a specific AruCo marker (0 = the charger marker).

        Same contract as dock; returns a job_id when the attempt starts.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        return self._submit_visit(
            "dock_on_aruco",
            self._cozmo.navigator.dock_on_aruco(marker_id),
            "aruco_dock",
        )

    def stop(self) -> dict[str, Any]:
        """Emergency stop: cancel all motion jobs, stop the motors, and flag all background jobs cancelled.

        Always available (works even when confirmation is required and
        when disconnected). Call this whenever motion looks wrong.
        """
        self._cozmo.navigator.stop()
        cancelled: list[str] = []
        for entry in self.jobs.list():
            if entry["status"] == "running" and self.jobs.cancel(entry["job_id"]):
                cancelled.append(entry["job_id"])
        return {"ok": True, "cancelled_jobs": cancelled}

    # ------------------------------------------------------------------
    # Actuation (synchronous, short)
    # ------------------------------------------------------------------

    def say(self, text: str, confirm: bool = False) -> dict[str, Any]:
        """Speak text through the robot's offline TTS (a few seconds; results are cached).

        Not motion, but gated by confirmation like other actuators when
        confirmation is on (it interrupts whatever the robot is doing).
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        if not text or not text.strip():
            return self._refusal("text must be non-empty")
        self._cozmo.say(text.strip())
        return {"ok": True}

    def set_head_angle(self, angle_deg: float, confirm: bool = False) -> dict[str, Any]:
        """Point the head camera at an angle in degrees (clamped to the mechanical range, about -25 to 44.5).

        Tilt down to see the floor/cubes close to the robot.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        clamped = max(MIN_HEAD_ANGLE.degrees, min(MAX_HEAD_ANGLE.degrees, angle_deg))
        self._cozmo.set_head_angle(clamped)
        return {"ok": True, "angle_deg": clamped}

    def set_lift_height(self, height_mm: float, confirm: bool = False) -> dict[str, Any]:
        """Move the lift to a height in mm (clamped to the mechanical range, about 32 to 92).

        92 is forks-up (carrying height); 32 is down.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        clamped = max(MIN_LIFT_HEIGHT.mm, min(MAX_LIFT_HEIGHT.mm, height_mm))
        self._cozmo.set_lift_height(clamped)
        return {"ok": True, "height_mm": clamped}

    def set_origin(self, confirm: bool = False) -> dict[str, Any]:
        """Set the localization origin to the robot's current pose - world coordinates restart here.

        Use after manually placing the robot; existing world-map poses
        keep their frame, the robot pose resets relative to it.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        self._cozmo.set_origin()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def camera_settings(
        self,
        auto_exposure_enabled: bool | None = None,
        gain: float | None = None,
        exposure_ms: float | None = None,
        jpeg_quality: int | None = None,
        color: bool | None = None,
        head_light: bool | None = None,
    ) -> dict[str, Any]:
        """Update camera settings (any subset; omitted keys keep their current values) and echo back the applied state.

        Unknown keys are ignored by the camera service.
        """
        refusal = self._require_connected()
        if refusal is not None:
            return refusal
        settings = {
            key: value
            for key, value in {
                "auto_exposure_enabled": auto_exposure_enabled,
                "gain": gain,
                "exposure_ms": exposure_ms,
                "jpeg_quality": jpeg_quality,
                "color": color,
                "head_light": head_light,
            }.items()
            if value is not None
        }
        if not settings:
            return {"ok": True, "settings": jsonable(self._cozmo.vision.camera_settings)}
        return {"ok": True, "settings": jsonable(self._cozmo.vision.apply_settings(settings))}

    def astar_toggle(self, enabled: bool) -> dict[str, Any]:
        """Enable or disable A* obstacle-avoiding pathfinding for navigate commands.

        Needs an occupancy map (vision floor pipeline enabled).
        """
        self._cozmo.navigator.set_astar_enabled(bool(enabled))
        return {"ok": True, "astar_enabled": self._cozmo.navigator.is_astar_enabled()}

    # ------------------------------------------------------------------
    # Job tools
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return one background job's status: running | done | error, plus result/error when finished.

        Unknown ids return ok=false.
        """
        entry = self.jobs.get(job_id)
        if entry is None:
            return self._refusal(f"Unknown job_id {job_id!r}")
        return {"ok": True, "job": entry}

    def wait_for_job(self, job_id: str, timeout_s: float = DEFAULT_WAIT_S) -> dict[str, Any]:
        """Block up to timeout_s (max 300) for a job to finish, then return it like get_job.

        A still-running job comes back with status running - retry later
        or use get_job to poll.
        """
        entry = self.jobs.wait(job_id, timeout_s=min(max(0.0, timeout_s), MAX_WAIT_S))
        if entry is None:
            return self._refusal(f"Unknown job_id {job_id!r}")
        return {"ok": True, "job": entry, "finished": entry["status"] != "running"}

    def list_jobs(self) -> dict[str, Any]:
        """List all background jobs (newest first) with status, result, and error."""
        return {"ok": True, "jobs": self.jobs.list()}
