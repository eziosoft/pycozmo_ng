# Changelog

All notable changes to **PyCozmo-NG** are recorded here.

History up to v0.8.0 belongs to upstream [pycozmo](https://github.com/zayfod/pycozmo)
and is intentionally not carried over — this changelog starts fresh with the first
PyCozmo-NG release. Keep it up to date with every release (see `AGENTS.md`).

## Unreleased

- Removed `vision/processor.py` backward-compat shim (first no-shim deletion). Nothing imported it.

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
