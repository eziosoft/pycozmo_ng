<!--
Maintainer runbook (private): read tracker page COZMO-DOC-22 before starting work.
-->

## Project overview

PyCozmo-NG is a pure-Python library for direct Cozmo robot control over Wi-Fi — a standalone replacement for the Cozmo app, with computer vision, navigation, and a 3D web viewer.

**Origin.** Forked from [zayfod/pycozmo](https://github.com/zayfod/pycozmo), which is credited as the origin of this codebase — not a development target.

**Standalone means, for every change:**

- **This repository is home.** Nothing is contributed back to zayfod/pycozmo.
- **Remove, never wrap.** The fork starts fresh at **v0.9.0**: when an old `pycozmo` surface gets in the way, delete it instead of keeping a compatibility shim. Import the package as `pycozmo_ng` (distribution `pycozmo-ng`, display "PyCozmo-NG" — hyphens are legal in package names but illegal in Python identifiers). `import pycozmo` always means the upstream package.
- **The agent is responsible for this project.** The user reviews results and sets direction.

## Setup

```bash
# Always inside a virtual environment — packages never install system-wide
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install pytest ruff mypy
```

Running the demo or the live tests needs a Cozmo on Wi-Fi: copy `.env.example` to `.env` and fill in the robot's Wi-Fi credentials (`demo.py` reads it via python-dotenv; `start.sh` bootstraps venv + credentials + reachability, then runs the demo). Try `python demo.py --help` for the options.

## Pre-commit gates

Every gate must pass before committing. Run them from the repo root **with the venv activated** — the gates must use the venv's toolchain, since a ruff outside the venv can be a different version and report a different error set:

```bash
ruff check pycozmo_ng/            # lint
ruff format --check pycozmo_ng/   # formatting
mypy pycozmo_ng/                  # types
pytest pycozmo_ng/tests/ -v       # tests — mocked, no robot needed
```

There is no working CI: `.github/workflows/pythonpackage.yml` is a stale upstream leftover (dead branches, missing `requirements-dev.txt`), so these local gates are the only check. JS: no linter is configured — the viewer's JavaScript lives inline in `pycozmo_ng/web/static/index.html`.

## Read CONTEXT.md first

`CONTEXT.md` defines the canonical frames (world / body / camera / scene) and coordinate transforms the whole localization stack depends on. Read it before touching `vision/`, `navigation/`, `util/coords.py`, the occupancy map, or anything that moves, stores, or renders a pose.

## How the code is organized

- **Library-first.** New functionality lives in `pycozmo_ng/` with a clean public API. The test: *"could someone use this without demo.py?"* `demo.py` is only an entrypoint — argparse + wiring; it owns no reusable behavior, state, or workflows.
- **One class per file.** A feature that doesn't fit an existing class's stated job gets a new file; files group into folders by function (`vision/`, `navigation/`, `web/`, `cubes/`); a genuinely new function gets a new folder.
- **`Cozmo` is the public API** (`from pycozmo_ng import Cozmo`), with subsystems behind properties: `.vision`, `.cubes`, `.carry`, `.navigator`, `.world_map`, `.sensors`, `.web_viewer`. `Brain` is the internal composition root: it builds the typed state containers (`FrameState`, `CubeTrackingState`, `DockState`) and injects them into domain classes.
- **The web viewer is a pure display.** All localization math lives in `WorldMap`; the browser receives scene-frame coordinates from `state_builder` and sends actions back through `MsgRouter` (vocabulary in CONTEXT.md "Browser action").
- **Detectors are pluggable.** `DetectionPipeline` takes any `CubeDetector`, `CubeFusionTracker`, `FaceDetector` implementation via constructor injection.

## Layout

```
CONTEXT.md              # Frames & coordinate transforms — read before any pose/frame work
demo.py                 # Demo entry point — argparse + wiring only; no reusable logic
start.sh                # Bootstrap: venv, .env Wi-Fi credentials, robot reachability → demo.py
pyproject.toml          # Packaging (distribution pycozmo-ng) + ruff config
requirements.txt        # Runtime dependencies (mirrors pyproject)
mypy.ini                # mypy per-module config
aruco_markers.json      # Committed ArUco marker sizes — data, not code (see Domain notes)
.env.example            # Template for the robot's Wi-Fi credentials → copy to .env
README.md               # User-facing overview; its install section still names upstream `pycozmo`
docs/                   # Reference documentation (tracked)
└── protocol.md         # Full wire-protocol reference + the 2458 map (read before any protocol work)
not_used/               # Deprecated code kept locally for reference — gitignored, not installed
pycozmo_ng/             # Main library (import name: pycozmo_ng)
├── cozmo.py            # Cozmo — the public API (from pycozmo_ng import Cozmo)
├── brain.py            # Brain — internal composition root; not public API
├── client.py           # Client — low-level robot connection (paths → PathExecutor)
├── event.py            # Event dispatcher
├── exception.py        # Exception classes
├── logger.py           # Per-channel loggers + setup_basic_logging()
├── wifi.py             # Wi-Fi auto-connect helper
├── protocol/           # Cozmo wire protocol: declarations → encoder, BinaryReader/Writer
├── transport/          # conn (socket + threads), frame (encode/decode), window (reliability)
├── camera/service.py   # CameraService — enable/disable, settings vocabulary, state in force
├── display/            # procedural_face.py + expressions/ (Happiness, …)
├── vision/             # Per-frame sequence: frame_stage → detect → annotate → publish → world map
│   ├── frame_stage.py      # FrameStage — the per-frame sequence
│   ├── frame_state.py      # FrameState — typed frame/detection container (injected)
│   ├── frame_sink.py       # FrameSink — the display-side seam FrameStage publishes through
│   ├── vision_setup.py     # VisionSetup — the one declaration of the vision options
│   ├── annotation.py       # AnnotationEngine — drawing/detection overlays
│   ├── distance_rings.py   # DistanceRings — ground range circles (250…1000 mm) in perspective
│   ├── events.py           # DetectionEventDispatcher — cube/face events
│   ├── camera.py           # Camera image decoding
│   ├── image_encoder.py    # Image encoding utilities
│   ├── face_detector.py    # FaceDetector (Haar cascade)
│   ├── face_identity.py    # FaceIdentity/FaceFrame — face presence epochs (ID per run)
│   ├── interfaces.py       # Detector/tracker protocols (CubeDetector, …)
│   ├── calibration/        # camera_calibration.py (from robot NVRAM), robot_center* (two-ring offset)
│   ├── cube_detector/      # CozmoCubeDetector + CubeFusionTracker, detection_pipeline, cube_faces/
│   ├── aruco_detector/     # Standard 4x4 ArUco + per-ID sizes (aruco_config → aruco_markers.json)
│   ├── charger_detector/   # ChargerDetector — rectangular dock marker (template + marker/charger.png)
│   └── occupancy_map/      # floor_pipeline, floor_projector (bird's-eye), occupancy_map (world grid)
├── navigation/
│   ├── navigator.py        # Navigator — navigate, approach, dock, stop
│   ├── world_map.py        # WorldMap — fused robot/cube poses, path trail; all localization math
│   ├── astar.py, astar_follower.py, path_executor.py, steering.py
│   │                        # A* plan/follow, firmware path state machine + lock, differential drive
│   ├── docking.py, aruco_docking.py, dock_state.py
│   │                        # Charger dock (contact + retries), ArUco-marker dock, dock pose tracking
│   ├── cube_approach.py, cube_stack.py, cube_visit.py
│   │                        # Two-step standoff approach, stack the carried cube, one-cube visit lock
│   ├── motion_controller.py, motion_token.py   # Running motion jobs + per-job cancellation handle
│   └── remote_control.py   # RemoteControl — browser keyboard driving
├── cubes/              # BLE: manager (mapping/connection + keepalive), carry_state, identity (ID↔index ±1)
├── web/                # WebViewerServer — WS + HTTP, MJPEG streams, MsgRouter, state_builder
│   └── static/          # Three.js viewer: index.html (JS is inline), style.css
├── animation/          # AnimController + cozmo_anim.fbs schema
├── sound/              # audio.py (μ-law encoding), tts.py (offline TTS, WAV cache)
├── util/               # coords.py (frame transforms), util.py (Pose/Vector3/Angle), robot.py, filter.py, object.py
├── lights/             # Color class + predefined light states
├── mcp/                # MCP adapter for LLM clients (tools, jobs, safety, server)
├── tests/              # Unit tests — mocked, run without a robot
└── tests_live/         # Opt-in tests against a connected Cozmo
```

## Code style

- **Python**: ruff, max line length 120, PEP 8. Every `.py` file stays green under both `ruff check` and `ruff format --check` (see gates).
- **All output goes through the loggers** in `pycozmo_ng/logger.py` (`logger`, `logger_cubes`, `logger_protocol`, `logger_robot`, …): info = progress/status, warning = recoverable problems, `logger.exception` inside `except` blocks, debug = per-frame/per-packet detail. Never embed severity in the message text (emoji, `WARNING:`, banners) — the level and format already carry it. `print()` is allowed only in docstring usage examples, `not_used/`, and throwaway test scripts.
- Type hints and docstrings throughout.

## Domain notes

**Camera calibration.** Cozmo's stored distortion coefficients are unreliable → zero distortion by default; undistortion is off by default (it can warp). Calibration is retrieved from robot NVRAM (`vision/calibration/camera_calibration.py`).

**Cubes.** Three light cubes, IDs 1–3 (`cubes/identity.py` owns the only ID↔index ±1). Marker templates: `vision/cube_detector/cube_faces/` (6 faces × 3 cubes). `CubeFusionTracker` fuses per-frame face detections into coherent 3D poses; head-tilt compensation is required for accuracy. `DetectionPipeline` returns camera-frame measurements — `WorldMap` does the world-frame fusion.

**Faces.** Haar cascade (`FaceDetector`); on detection the procedural face switches to `Happiness`. `face_detect_every_n_frames` (in `VisionSetup`, default 1 = every frame) trades CPU for latency.

**Floor & occupancy.** Bird's-eye floor projection feeds a world-frame `OccupancyMap` grid accumulated over time; floor center and size are configurable in mm.

**Markers (ArUco & charger).** Marker sizes are data, not code: `aruco_markers.json` at the project root (override with `PYCOZMO_NG_ARUCO_CONFIG`). A wrong size sets the reported distance wrong by the same factor. Dock on the charger marker via `navigator.dock_on_aruco()` (browser: `Z`).

**Web viewer.** One server on port **8765**: `/` serves the viewer, `/annotated-video` and `/floor-video` are MJPEG streams. It renders robot, cubes, faces, charger marker, path trail, occupancy grid; streams the procedural face over the WebSocket; browser actions (drive, navigate, approach, dock, camera settings) arrive through `MsgRouter`. Carried cube: `state_builder` marks it `carried: true` and the browser pins it to the robot's rendered pose at a fixed forks-up height — the height is fixed by design, never read from lift telemetry, so it must not lerp on its own. The HUD "Carry" row reads the same on-demand `CarryDetector` as the public `.carry`.

**TTS.** `Cozmo.say()` / `Client.say()` uses pyttsx3 offline with WAV caching; needs the system package `espeak-ng` on Linux (`sudo apt-get install espeak-ng`).

**Environment variables.** `PYCOZMO_NG_LOG_LEVEL`, `PYCOZMO_NG_PROTOCOL_LOG_LEVEL`, `PYCOZMO_NG_ROBOT_LOG_LEVEL` (defaults INFO / INFO / WARNING); `PYCOZMO_NG_ARUCO_CONFIG` (marker-size JSON path); `PYCOZMO_NG_LIVE_MOTION=1` (permits live tests that move the robot).

## Testing

- `pycozmo_ng/tests/` — mocked unit tests, run without a robot; part of the gates above.
- `pycozmo_ng/tests_live/` — opt-in integration tests against a connected Cozmo (own `conftest.py`): run on the robot network, the whole suite skips when the robot is unreachable, and tests that physically move the robot additionally require `PYCOZMO_NG_LIVE_MOTION=1` — clear the floor space first.
- No automated coverage for the real camera/vision pipeline beyond what the mocked unit tests provide.

## Branches

Active development happens on **`develop`** — the main line, where everything described here lives. `master` mirrors upstream's default branch.
