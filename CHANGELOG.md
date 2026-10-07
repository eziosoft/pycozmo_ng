# Changelog

All notable changes to **PyCozmo-NG** are recorded here.

History up to v0.8.0 belongs to upstream [pycozmo](https://github.com/zayfod/pycozmo)
and is intentionally not carried over — this changelog starts fresh with the first
PyCozmo-NG release. Keep it up to date with every release (see `AGENTS.md`).

## Unreleased

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
