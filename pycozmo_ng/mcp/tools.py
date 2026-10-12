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
* **Frame-derived reads report freshness** — ``frame_age_s`` /
  ``stale`` (and ``camera_active``) distinguish "nothing in view" from
  "no frames"; the recovery for the latter is ``camera_toggle``.
  ``get_occupancy_map`` carries the same freshness: the grid only grows
  while frames arrive, and it renders the explored world for the agent
  as an ASCII top-down map ('#' obstacle, '.' free, '?' fog, 'R' robot)
  plus a colour PNG.
* **Drive tools refuse only while the robot is picked up** — the
  firmware holds its wheels then, and only the operator can put it
  back down. A robot on the charger just drives off, so the charger is
  reported by get_sensors but never a refusal; a wheel command that
  goes unanswered is caught by stall detection and the job's ``reason``
  instead of a timeout that explains nothing.
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
from pycozmo_ng.sound import TTSGenerationError
from pycozmo_ng.util.robot import MAX_HEAD_ANGLE, MAX_LIFT_HEIGHT, MIN_HEAD_ANGLE, MIN_LIFT_HEIGHT
from pycozmo_ng.vision.occupancy_map import FREE, OCCUPIED

if TYPE_CHECKING:
    from pycozmo_ng.cozmo import Cozmo

#: How long ``wait_for_job`` may block by default (seconds). Kept under
#: typical MCP client tool timeouts so the call itself never looks hung.
DEFAULT_WAIT_S = 30.0

#: Hard cap for a requested wait (seconds).
MAX_WAIT_S = 300.0

#: Longest ``wait`` may sleep inside one tool call (seconds). Short
#: enough to stay under any MCP client's tool timeout.
MAX_SLEEP_S = 10.0

#: A frame older than this is reported as ``stale`` — the camera has
#: stopped producing (or the pipeline is not running) and the image the
#: caller is looking at is history, not the current scene.
STALE_FRAME_S = 2.0

#: Longest side, in cells, of the ASCII grid ``get_occupancy_map``
#: returns. Larger maps are block-downsampled to fit — an LLM reads a
#: grid it can hold in attention, and an occupied cell survives its
#: block, so coarsening never hides an obstacle.
MAX_ASCII_CELLS = 40

#: Longest side, in pixels, of the colour PNG ``get_occupancy_map``
#: attaches (nearest-neighbour upscale — the grid is flat colour).
MAX_OCCUPANCY_PNG_PX = 320


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

    def _current_pose(self) -> tuple[float, float, float]:
        """World pose ``(x, y, yaw)`` from the world map (mm, radians)."""
        position, yaw = self._cozmo.world_map.robot_pose
        return float(position[0]), float(position[1]), float(yaw)

    def _frame_freshness(self) -> tuple[float | None, bool]:
        """``(age_s, stale)`` of the newest decoded frame.

        ``age_s`` is ``None`` before the first frame ever arrived. A
        frame is stale when it is older than :data:`STALE_FRAME_S` *or*
        the camera has been stopped — a stopped camera produces nothing
        new, so whatever is on hand is history however recent it looks.
        """
        age = self._cozmo.vision.frame_age_s
        if age is None:
            return None, True
        stale = age > STALE_FRAME_S or not self._cozmo.vision.active
        return round(age, 2), stale

    def _drivable_refusal(self) -> dict[str, Any] | None:
        """Gate for every tool that turns the wheels, or ``None`` when drivable.

        Only a picked-up robot is refused: the firmware holds its wheels
        then, and only the operator can put it back down. A robot on the
        charger is *not* blocked — it simply drives off — so the charger
        is reported by get_sensors but never a refusal; a wheel command
        that genuinely goes unanswered is caught by the stall detection
        and reported through the job's ``reason``.
        """
        if self._cozmo.sensors.picked_up:
            return self._refusal("Robot is picked up - set it down before driving")
        return None

    def _motion_failure_reason(self, made_progress: bool) -> str:
        """Why a motion job came back unsuccessful, in terms the caller can act on.

        ``made_progress`` distinguishes "moved, but not far enough"
        (localization/obstruction — re-read state and retry) from "the
        wheels never turned" (the robot is being held, or something
        physically blocks it).
        """
        if self._cozmo.sensors.picked_up:
            return "Robot is picked up - set it down before driving"
        if not made_progress:
            return "Wheels did not respond - is the robot picked up or obstructed?"
        return "Did not complete in time - re-read get_robot_state and retry"

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_robot_state(self) -> dict[str, Any]:
        """Merged snapshot: connection, pose, telemetry, carry, vision freshness, navigation, running jobs, and the fused world map (path trails omitted).

        The one call to orient a new turn. Positions are world-frame
        millimetres relative to the last set_origin(); headings radians.
        ``vision.frame_age_s``/``camera_active`` say whether the camera is
        actually producing; ``cubes_connected`` is the BLE link (distinct
        from the cubes the world map sees); ``jobs_running`` is how many
        background commands are still going. For the full map including
        path trails use get_world_map; for telemetry only get_sensors.
        """
        cozmo = self._cozmo
        world = self._world_snapshot(include_trails=False)
        position, yaw = cozmo.world_map.robot_pose
        frame_age_s, stale = self._frame_freshness()
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
            "vision": {
                "camera_active": cozmo.vision.active,
                "frame_age_s": frame_age_s,
                "frame_stale": stale,
            },
            "cubes_connected": cozmo.cubes.connected,
            "navigation": {
                "astar_enabled": cozmo.navigator.is_astar_enabled(),
                "active_motions": list(cozmo.navigator.active_motions),
            },
            "jobs_running": sum(1 for entry in self.jobs.list() if entry["status"] == "running"),
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

    def get_occupancy_map(self) -> dict[str, Any]:
        """Top-down occupancy map of the world you have explored: '#' is an obstacle, '.' proven floor, '?' fog you have never seen, 'R' is you.

        Read it as a map laid in front of you: the top row is ahead (+X
        forward), the left column is to your left (+Y left) — the same
        world frame, millimetres, as navigate_to and get_world_map. Each
        character is one cell of ``render_cell_size_mm``.

        '#' is the inflated base of something standing on the floor: do
        not plan a path through it. '?' is not empty space — it is
        unexplored: look (get_frame) or creep closer before navigating
        there, and prefer A*-guided navigate (astar_toggle) which plans
        on this very grid.

        The grid accumulates only while camera frames arrive, so
        ``camera_active`` / ``frame_stale`` say whether this is a live
        map or a snapshot frozen when the camera stopped. A colour PNG
        of the same grid (red = obstacle, green = free, grey = fog) is
        attached by the MCP server as image content.

        Refused when the occupancy map is disabled — the vision floor
        pipeline is off.
        """
        occ = self._cozmo.vision.occupancy_map
        if occ is None:
            return self._refusal(
                "Occupancy map is not enabled - the vision floor pipeline is off "
                "(setup_vision(enable_occupancy_map=True))"
            )

        grid = occ.grid
        cells = int(grid.shape[0])
        # Block-downsample grids larger than the ASCII cap; an occupied
        # cell always wins its block — the drawing may coarsen, an
        # obstacle may never disappear into one.
        factor = max(1, -(-cells // MAX_ASCII_CELLS))
        rows = cells // factor

        lines: list[str] = []
        counts = {"occupied": 0, "free": 0, "unknown": 0}
        for i in range(rows):
            x_block = rows - 1 - i  # display rows run from +X (ahead) down
            chars: list[str] = []
            for j in range(rows):
                y_block = rows - 1 - j  # ...columns from +Y (left) right
                block = grid[y_block * factor : (y_block + 1) * factor, x_block * factor : (x_block + 1) * factor]
                if (block == OCCUPIED).any():
                    chars.append("#")
                    counts["occupied"] += 1
                elif (block == FREE).any():
                    chars.append(".")
                    counts["free"] += 1
                else:
                    chars.append("?")
                    counts["unknown"] += 1
            lines.append("".join(chars))

        # 'R' marks the robot — which cell *I* stand on matters more than
        # that cell's class underneath. An occupied block keeps its '#':
        # coarsening may never hide an obstacle — robot.cell says where
        # I am instead.
        rx, ry, ryaw = self._current_pose()
        origin_x, origin_y = occ.origin
        half = occ.grid_size_mm / 2.0
        x_idx = (rx - origin_x + half) / occ.cell_size_mm
        y_idx = (ry - origin_y + half) / occ.cell_size_mm
        robot_cell = None
        if 0.0 <= x_idx < cells and 0.0 <= y_idx < cells:
            i = rows - 1 - int(x_idx) // factor
            j = rows - 1 - int(y_idx) // factor
            if lines[i][j] != "#":
                lines[i] = lines[i][:j] + "R" + lines[i][j + 1 :]
            robot_cell = {"row": i, "col": j}

        # Colour render for the image content block (to_color_image is
        # already in the ASCII orientation: +X forward, +Y left).
        img = occ.to_color_image()
        png = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
        scale = max(1, MAX_OCCUPANCY_PNG_PX // cells)
        if scale > 1:
            png = png.resize((cells * scale, cells * scale), Image.Resampling.NEAREST)
        buf = io.BytesIO()
        png.save(buf, format="PNG")

        age_s, stale = self._frame_freshness()
        total = rows * rows
        explored = counts["occupied"] + counts["free"]
        return {
            "ok": True,
            "occupancy": {
                "grid_size_mm": float(occ.grid_size_mm),
                "cell_size_mm": float(occ.cell_size_mm),
                "render_cell_size_mm": float(occ.cell_size_mm) * factor,
                "origin_mm": [float(origin_x), float(origin_y)],
                "view": "top-down: top = +X forward, left = +Y left (world frame, mm); one char per cell",
                "legend": {
                    "#": "occupied - the base of an obstacle, do not drive into",
                    ".": "free - floor that has been seen",
                    "?": "unknown - fog, never observed; look before driving there",
                    "R": "robot - you are here (a cell that is also occupied stays '#'; robot.cell locates you)",
                },
                "map": lines,
                "robot": {"x_mm": rx, "y_mm": ry, "yaw_rad": ryaw, "cell": robot_cell},
                "counts": {**counts, "explored_percent": round(100.0 * explored / total, 1) if total else 0.0},
                "png_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                "camera_active": self._cozmo.vision.active,
                "frame_age_s": age_s,
                "frame_stale": stale,
            },
        }

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

        Empty when vision is not running — check ``camera_active`` and
        ``frame_age_s`` to tell "nothing in view" from "no frames" (an
        empty list with camera_active false or a large frame_age_s means
        the camera is off/stale, not that the scene is empty; recover
        with camera_toggle). The fused, world-frame view lives in
        get_world_map.
        """
        detections = self._cozmo.vision.latest_detections
        out = []
        for det in detections:
            item = jsonable(det)
            if isinstance(item, dict) and "cube_index" in item:
                item["cube_id"] = to_cube_id(int(item.pop("cube_index")))
            out.append(item)
        frame_age_s, stale = self._frame_freshness()
        return {
            "ok": True,
            "count": len(out),
            "camera_active": self._cozmo.vision.active,
            "frame_age_s": frame_age_s,
            "frame_stale": stale,
            "detections": out,
        }

    def get_frame(
        self,
        annotated: bool = False,
        max_width: int = 0,
        jpeg_quality: int = 70,
    ) -> dict[str, Any]:
        """Latest camera frame as a base64-encoded JPEG, optimised for LLM consumption.

        Args:
            annotated: When true, return the annotated frame (detection boxes,
                labels, confidence, distance drawn on the image, plus a
                translucent orange ground-distance tint — strongest from the
                image bottom to 250 mm, lighter from 250 to 500 mm — for
                judging range by eye)
                instead of the raw camera frame.
            max_width: Downscale the image so its width is at most this many
                pixels (aspect ratio preserved). 0 = original resolution. Use
                320 or 160 for smaller payloads.
            jpeg_quality: JPEG encoding quality (1-95, default 70). Lower
                values produce smaller base64 strings at the cost of fidelity.

        Returns ``{"ok": true, "frame": ...}`` with fields:
        ``jpeg_base64``, ``width``, ``height``, ``timestamp`` (capture
        time), ``age_s``, ``stale``, ``detections_summary``. When the
        camera is stopped the newest frame is still returned but marked
        ``stale`` — history, not a live view; when no frame has ever
        arrived ``frame`` is ``null``. Recover the stream with
        camera_toggle.
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
        age_s, stale = self._frame_freshness()

        return {
            "ok": True,
            "frame": {
                "jpeg_base64": b64,
                "width": img.width,
                "height": img.height,
                # Capture time, not serialization time — the point of age_s.
                "timestamp": time.time() - (age_s or 0.0),
                "age_s": age_s,
                "stale": stale,
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

        # Translucent ground-distance zones (if enabled)
        dr = self._cozmo.vision.distance_rings
        if dr is not None and dr.radii_mm:
            inner = 0.0
            zone_desc: list[str] = []
            for radius in dr.radii_mm:
                zone_desc.append(f"{inner:g}-{radius:g} mm")
                inner = radius
            parts.append(
                "Translucent orange ground zones for judging range — "
                + ", ".join(zone_desc)
                + " (nearer zone more saturated; tint never fully opaque)"
            )

        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Motion (job-returning)
    # ------------------------------------------------------------------

    def navigate_to(self, x: float, y: float, confirm: bool = False) -> dict[str, Any]:
        """Drive to a world-frame point (mm). Returns a job_id.

        This is the way to cross the room: with A* enabled
        (astar_toggle(true), the recommended default — turn it on once
        and leave it on) the route is planned around the obstacles in
        the occupancy map and re-planned live if the way ahead fills
        in; without it the robot drives straight at the target. Poll
        get_job/wait_for_job for {arrived, distance_mm, timed_out}
        (plus reason when not arrived). Refused beyond the safety
        extent from the origin, while the robot is picked up, and
        (when configured) without confirmation - pass confirm=true
        then. For short in-place steps use move_by/turn_by instead.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        refusal = self._drivable_refusal()
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
        start_x, start_y, _ = self._current_pose()
        start_distance = math.hypot(x - start_x, y - start_y)
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
        arrived = distance <= 25.0 and not timed_out
        result = {
            "arrived": arrived,
            "distance_mm": round(distance, 1),
            "timed_out": timed_out,
            "target_mm": [x, y],
        }
        if not arrived:
            result["reason"] = self._motion_failure_reason(start_distance - distance > 1.0)
        return result

    def move_by(self, distance_mm: float, speed_mmps: float = 60.0, confirm: bool = False) -> dict[str, Any]:
        """Drive straight forward (positive mm) or reverse (negative) from the current pose, keeping the heading. Returns a job_id.

        The projected target is checked against the safety extent like
        navigate_to, and speed_mmps is clamped to the safety speed cap.
        Poll get_job/wait_for_job for {arrived, requested_mm, actual_mm}
        (plus reason when not arrived). Refused while the robot is picked
        up, and (when configured) without confirm=true.
        Use this for small positioning moves that need no world point.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        refusal = self._drivable_refusal()
        if refusal is not None:
            return refusal
        if abs(distance_mm) < 1.0:
            return self._refusal("distance_mm must be at least 1 mm")
        speed = self.limits.clamp_speed(speed_mmps)
        start_x, start_y, yaw = self._current_pose()
        target = (
            start_x + distance_mm * math.cos(yaw),
            start_y + distance_mm * math.sin(yaw),
        )
        reason = self.limits.target_error(*target)
        if reason is not None:
            return self._refusal(reason)
        job_id = self.jobs.submit("move_by", lambda: self._run_move_by(distance_mm, speed))
        return {"ok": True, "job_id": job_id, "target_mm": [round(v, 1) for v in target], "speed_mmps": speed}

    def _run_move_by(self, distance_mm: float, speed: float) -> dict[str, Any]:
        """Run ``navigator.move_by`` and report the displacement actually covered."""
        cozmo = self._cozmo
        start_x, start_y, _ = self._current_pose()
        arrived = cozmo.navigator.move_by(distance_mm, speed_mmps=speed)
        end_x, end_y, _ = self._current_pose()
        covered = math.hypot(end_x - start_x, end_y - start_y)
        result: dict[str, Any] = {
            "arrived": bool(arrived),
            "requested_mm": distance_mm,
            "actual_mm": round(covered if distance_mm >= 0 else -covered, 1),
        }
        if not arrived:
            result["reason"] = self._motion_failure_reason(covered >= 1.0)
        return result

    def turn_by(self, angle_deg: float, confirm: bool = False) -> dict[str, Any]:
        """Turn in place by a relative angle (degrees, positive = CCW/left). Returns a job_id.

        Angles past ±180 take the short way around (the target heading
        wraps); beyond ±360 is refused as meaningless. Poll
        get_job/wait_for_job for {aligned, requested_deg, turned_deg}
        (plus reason when not aligned). Refused while the robot is picked
        up, and (when configured) without confirm=true.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        refusal = self._drivable_refusal()
        if refusal is not None:
            return refusal
        if abs(angle_deg) > 360.0:
            return self._refusal(f"angle_deg must be within ±360 (got {angle_deg:.0f})")
        job_id = self.jobs.submit("turn_by", lambda: self._run_turn(angle_deg))
        return {"ok": True, "job_id": job_id}

    def _run_turn(self, angle_deg: float) -> dict[str, Any]:
        """Run ``navigator.turn_by`` and report the heading actually swept."""
        cozmo = self._cozmo
        _, start_yaw = cozmo.world_map.robot_pose
        aligned = cozmo.navigator.turn_by(angle_deg)
        _, end_yaw = cozmo.world_map.robot_pose
        turned = (math.degrees(float(end_yaw) - float(start_yaw)) + 180.0) % 360.0 - 180.0
        result: dict[str, Any] = {
            "aligned": bool(aligned),
            "requested_deg": angle_deg,
            "turned_deg": round(turned, 1),
        }
        if not aligned:
            result["reason"] = self._motion_failure_reason(abs(turned) >= 1.0)
        return result

    def approach_cube(self, cube_id: int, confirm: bool = False) -> dict[str, Any]:
        """Drive to cube 1-3, align, and grab it (stacks the held cube onto it when one is on the forks).

        Returns a job_id when the visit starts; refused (ok=false) when
        the cube is unknown or another visit is in flight.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        refusal = self._drivable_refusal()
        if refusal is not None:
            return refusal
        if cube_id not in (1, 2, 3):
            return self._refusal(f"cube_id must be 1, 2 or 3 (got {cube_id})")
        return self._submit_visit(
            "approach_cube",
            self._cozmo.navigator.approach_cube(cube_id),
            "approach",
        )

    def stack_cube(self, cube_id: int, confirm: bool = False) -> dict[str, Any]:
        """Carry the cube on the raised forks to cube 1-3 and stack it on top.

        Requires a cube already on the forks (get_robot_state's carry
        block). Returns a job_id when the stack starts; refused
        (ok=false) when nothing is on the forks, the target is unknown,
        or another visit is in flight.
        """
        refusal = self._motion_precheck(confirm)
        if refusal is not None:
            return refusal
        refusal = self._drivable_refusal()
        if refusal is not None:
            return refusal
        if cube_id not in (1, 2, 3):
            return self._refusal(f"cube_id must be 1, 2 or 3 (got {cube_id})")
        return self._submit_visit(
            "stack_cube",
            self._cozmo.navigator.stack_cube(cube_id),
            "stack",
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
        refusal = self._drivable_refusal()
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
        try:
            self._cozmo.say(text.strip())
        except TTSGenerationError as exc:
            return self._refusal(str(exc))
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

    def camera_toggle(self, enabled: bool) -> dict[str, Any]:
        """Enable or disable the camera stream, echoing the state after (idempotent).

        Enabling restores the settings already in force (resolution,
        exposure, ...), so a toggle off/on loses no configuration.
        This is the recovery path when get_frame returns null or
        get_detections stays empty with camera_active false; disabling
        frees the robot's Wi-Fi bandwidth while you work from the map.
        """
        refusal = self._require_connected()
        if refusal is not None:
            return refusal
        vision = self._cozmo.vision
        try:
            if enabled and not vision.active:
                settings = dict(vision.camera_settings)
                self._cozmo.enable_camera(resolution=settings.pop("resolution", None), **settings)
            elif not enabled and vision.active:
                vision.stop()
        except RuntimeError as exc:
            return self._refusal(str(exc))
        return {"ok": True, "camera_active": vision.active}

    def astar_toggle(self, enabled: bool) -> dict[str, Any]:
        """Turn A* obstacle-avoiding pathfinding for navigate_to on or off.

        Recommended: enable it once at the start of a session and leave
        it on. It needs an occupancy map (vision floor pipeline
        enabled); without one - or when no route exists through the
        map - navigate_to quietly falls back to driving straight at the
        target, so a stubborn arrival usually means the target is
        walled off in get_occupancy_map, not that navigation is broken.
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def wait(self, seconds: float) -> dict[str, Any]:
        """Sleep up to 10 s inside the tool call, then report how long it actually slept.

        Use it to let the robot settle — a turn to finish, fresh camera
        frames to arrive, cube lights to refresh — instead of hammering
        the read tools back to back. For a motion job's outcome prefer
        wait_for_job, which returns as soon as the job ends.
        """
        started = time.monotonic()
        time.sleep(min(max(0.0, float(seconds)), MAX_SLEEP_S))
        slept = time.monotonic() - started
        return {"ok": True, "slept_s": round(slept, 2), "requested_s": seconds}
