# Changelog

All notable changes to **PyCozmo-NG** are recorded here.

History up to v0.8.0 belongs to upstream [pycozmo](https://github.com/zayfod/pycozmo)
and is intentionally not carried over — this changelog starts fresh with the first
PyCozmo-NG release. Keep it up to date with every release (see `AGENTS.md`).

## Unreleased

- Standard 4x4 ArUco markers are detected as first-class objects with
  **per-ID physical sizes** and rendered in the 3D viewer with their real
  bitmap texture. Sizes live in a JSON config (`aruco_markers.json` in the
  project root — committed to the repo; search order: explicit path →
  `PYCOZMO_NG_ARUCO_CONFIG` → project root/cwd → `~/.pycozmo_ng/`):
  `{"dictionary": "4X4_50",
  "default_size_mm": 50.0, "markers": {"0": 22.5}}` — the per-ID size sets the
  absolute PnP scale for that marker (unlisted IDs fall back to the default
  with a one-time warning; a missing file disables detection, an invalid file
  fails loudly rather than reporting wrong distances). New
  `vision/aruco_detector/` package: `ArucoDetector` (modern
  `cv2.aruco.ArucoDetector` API, `SOLVEPNP_IPPE_SQUARE`, border-clip guard),
  `ArucoDetection`, `MarkerSizeRegistry`. Wiring mirrors the charger path:
  `DetectionPipeline.detect_aruco()`, `VisionSetup.enable_aruco_detection`,
  events `EvtArucoDetected/Observed/Lost` (per-ID set diff), annotation
  overlay (magenta quad + `ID | size | distance`), and
  `WorldMap.update_aruco_measurement` — static landmarks keyed by marker ID,
  never forgotten on visibility loss, cleared by `reset_map`/origin change.
  Viewer: `state_builder` sends `aruco_markers` (scene pose + `size_mm`),
  `GET /aruco-texture?id=N` serves each marker's PNG (rendered with
  `generateImageMarker`, cached), the browser renders a textured plane at the
  fused pose with an ID/size label, plus an "ArUco" HUD counter.
- The charger's rectangular marker (23 × 15 mm battery drawing) is detected as
  a first-class object instead of being misread as a cube whose ID flapped
  every second (COZMO-65). New `vision/charger_detector/` package:
  `ChargerDetector` (cube-detector chassis with rectangular object points and
  `SOLVEPNP_IPPE`, marker aspect gate 1.1–2.0, best-quad-wins), a template
  loader matching the cube preprocessing (32×32), and `ChargerDetection`.
  Both detector families now carry a symmetric `reject_markers` guard — a quad
  is skipped when the *other* family's template out-scores its own — so the
  marker is never a cube and a cube face is never the charger (validated:
  phantom cube detections 5 → 0 across the real marker frames; charger found
  in 22/23, scores 0.71–0.84). Wiring: `DetectionPipeline.detect_charger()`,
  `VisionSetup.enable_charger_detection`, events
  `EvtChargerDetected/Observed/Lost`, `WorldMap.update_charger_measurement`
  (single stationary slot — no forgetting on visibility loss, cleared on
  `reset_map` or localization-origin change), and a green "Charger" overlay on
  the annotated stream. Cube ID space stays strictly 0–2; charger PnP uses the
  measured 23 × 15 mm — the image quads (1.32–1.37) are narrower than the
  physical 1.53 aspect, flagged as a calibration follow-up.
- The charger marker shows up in the 3D web viewer: `build_state_json` now
  carries a top-level `charger` key (`position`/`confidence`/`width`/`height`
  in scene frame, `null` until the first fused sighting), and the browser
  renders it as a marker-sized green (0x00ff40) tablet with a "Charger |
  NN%" label — the same green as the annotated-stream overlay — plus a
  "Charger" HUD row showing the fused confidence. It lerp/slerp-smooths
  like cubes and is removed when the world map forgets the charger
  (`reset_map` / localization-origin change).
- The charger pose now carries orientation, so the robot can align to the
  dock (COZMO-67): the PnP rotation flows `ChargerDetection.rotation_matrix`
  → `WorldMap.update_charger_measurement(position, rotation, …)` (camera →
  world transform, then confidence-weighted SLERP — the same fusion rules as
  cubes) → `get_charger()` / `get_world_state()["charger"]["rotation"]`.
  `build_state_json` sends the scene-frame rotation and the browser orients
  the marker tablet from it. The marker normal's world yaw is the dock's
  facing direction — the reference pose for docking alignment (navigation
  integration still to come).
- The frame stage reaches the display through one declared interface: new
  `vision/frame_sink.py` (`FrameSink`). `FrameStage` used to be typed against the
  concrete `WebViewerServer` and reach into its internals
  (`web_viewer.occupancy.push(...)`); `vision/` no longer imports `web/`, the
  two always-co-called floor-grid setters collapsed into `set_floor_grid(size,
  center)`, and `tests/test_frame_stage.py` asserts the per-frame sequence on a
  recording adapter instead of a `Mock()` that accepted anything (a fake that
  drifts from the interface now fails mypy).
- Occupancy delivery has one owner: new `web/occupancy_delivery.py`
  (`OccupancyDelivery.push` throttles the encode, `take()` answers "changed?
  → send → mark sent" in a single call that `build_state_json` consumes as one
  payload argument). The same decision used to be written twice — in
  `state_builder.py` and again in `viewer_server.py` — where the copies could
  drift apart. Deleted along the way: the no-op `WebViewerServer.render()`,
  the write-only `WebViewerServer.occupancy_map`, and the never-subscribed
  active-client listener (`add_active_clients_listener` /
  `active_client_count`); frame stage now calls `server.occupancy.push(...)`
  instead of `server.push_occupancy_grid(...)`.
- Vision setup options are declared once, in the new typed container
  `vision/vision_setup.py` (`VisionSetup`): `Brain.setup_vision(setup)` takes
  that object and `Cozmo.setup_vision(...)` / `Cozmo.vision.start(...)` accept
  its fields as keyword overrides. The facade used to re-list a shorter
  parameter map, so `cozmo.setup_vision(enable_occupancy_map=False)` raised
  `TypeError` while `cozmo.vision.start(...)` accepted it. Also: a camera
  calibration that the robot never delivers now raises a clear `RuntimeError`
  instead of failing later inside pipeline construction.
- Robot-center calibration now has one owner: new
  `vision/calibration/robot_center_calibration.py` holds the session offset,
  the arm/observe/freeze run protocol, and the collected readings.
  `DetectionPipeline` no longer latches a private `_robot_center_measured` flag
  and no longer exposes a settable `camera_lateral_offset_mm`; frames are handed
  to `RobotCenterCalibration.observe()` only while a run is armed, and
  `Cozmo.calibrate_robot_center()` averages the readings a run collected instead
  of polling a stored result every 0.1 s (which recounted the same measurement
  — the "average of 10 frames" was effectively one frame).
- Removed `vision/processor.py` backward-compat shim (first no-shim deletion). Nothing imported it.
- Removed `CubeMoveDetector` (`cubes/cube_imu.py`) and the whole cube-move
  suppression feature: the accel sample source was disabled in COZMO-28, so
  nothing fed the detector, and the downstream path it drove was unreachable.
  Gone with it: `CubeTrackingState.move_detectors`/`suppress`, the suppression
  block in `FrameStage._feed_world_map` (and `FrameStage`'s now-unused
  `cube_tracking` dependency), and the no-op/uncalled `forget_cube` on
  `DetectionPipeline` and `WorldMap`. To be rebuilt properly when the cube IMU
  is understood.
- New `CubeVisit` seam (`navigation/cube_visit.py`): one object now owns the
  "one cube visit at a time" lock, the background-job spawn, and the truthful
  "did it start?" answer that `approach_cube` / `approach_cube_at` /
  `stack_cube` (and so `cozmo.stack_cube`) return — they used to report
  `True` even when the visit was silently refused as busy. Replaces the bare
  `CubeApproach.approach_lock` that `CubeStack` reached through its sibling.

## v0.9.0 (2026-10-04)

First release of PyCozmo-NG — a standalone fork of pycozmo.

- Renamed project: distribution `pycozmo-ng`, package `pycozmo_ng`, display name PyCozmo-NG.
- Public `Cozmo` API; `Brain` is an internal orchestrator behind it.
- Vision: detection pipeline (cube + face), annotation engine, event dispatcher,
  robot-center calibration, floor projection with world-frame occupancy map.
- 3D web viewer: Three.js scene, WebSocket state broadcast, MJPEG camera/floor
  streams, procedural face streaming, browser keyboard driving, click-to-navigate,
  cube approach and dock requests.
- Navigation: A* planning and following, firmware path executor, cube approach and
  stacking, docking, motion-job registry with cancellation tokens.
- Cubes: BLE cycler, manager, IMU-based move detection, on-demand carry detection.
- Display: procedural face rendering, expressions, animations.
- Offline TTS via pyttsx3 + espeak-ng with WAV caching (`Cozmo.say()`).
- Architecture: typed state containers (`FrameState`, `CubeTrackingState`,
  `DockState`), deep-module refactor of vision, navigation, and web viewer.
