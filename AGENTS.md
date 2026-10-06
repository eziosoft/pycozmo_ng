# AGENTS.md

<!--
Maintainer runbook (private): read tracker page COZMO-DOC-22 before starting work.
-->

## Project Overview

PyCozmo-NG is a pure-Python library for direct Cozmo robot control over Wi-Fi — a standalone replacement for the Cozmo app, with full computer vision, navigation, and a 3D web viewer.

**Origin.** Forked from [zayfod/pycozmo](https://github.com/zayfod/pycozmo), which is credited as the origin of this codebase — not a development target.

**Goal.** Extend PyCozmo-NG into a full **standalone robot-control stack**. The standalone decision has three consequences for day-to-day work:

- **No upstream work.** Nothing is contributed back to zayfod/pycozmo; this repository is the home of the code.
- **No backward-compatibility shims.** The fork starts fresh at **v0.9.0** (see `CHANGELOG.md`); compatibility with old `pycozmo` import paths is intentionally not kept. When an old surface gets in the way, remove it instead of wrapping it.
- **Naming.** Distribution name `pycozmo-ng`, import name `pycozmo_ng`, display name "PyCozmo-NG". Hyphens are legal in package names but illegal in Python identifiers — that is why the import name differs. Never `import pycozmo`: that is the upstream package.

**Agent is responsible for this project.** The user reviews results and sets direction.

## Quick Start

```bash
# Always use a virtual environment — never install packages system-wide
python3 -m venv venv
source venv/bin/activate

# Install for development
pip install -e .
pip install pytest ruff mypy   # dev tools

# Run linting (every Python file must pass ruff before committing)
ruff check pycozmo_ng/
ruff format --check pycozmo_ng/

# Run type checking
mypy pycozmo_ng/

# Run tests (no robot required)
pytest pycozmo_ng/tests/ -v

# Run JS linting (for web viewer JS — no eslint config yet)
npx eslint pycozmo_ng/web/static/
```

## Key Commands

| Task | Command |
|------|---------|
| Lint | `ruff check pycozmo_ng/` |
| Format check | `ruff format --check pycozmo_ng/` |
| Type check | `mypy pycozmo_ng/` |
| Test | `pytest pycozmo_ng/tests/ -v` |
| Install | `pip install -e .` |
| Demo | `python demo.py` (see `--help`; needs a live robot) |

## Project Structure

```
CHANGELOG.md          # Release changelog (fresh from v0.9.0 — keep up to date)
CONTEXT.md            # Canonical frame conventions & coordinate transforms — read before vision/nav work
demo.py               # Demo entry point — argparse + wiring only; no reusable logic here
pyproject.toml        # Packaging (distribution `pycozmo-ng`) + ruff config
requirements.txt      # Runtime dependencies
mypy.ini              # mypy config
docs/                 # Protocol / hardware / reverse-engineering documentation
not_used/             # Deprecated & moved code (kept for reference, not installed)
pycozmo_ng/           # Main library (import name: pycozmo_ng)
├── cozmo.py          # Cozmo — public API for library users
├── brain.py          # Brain — internal orchestrator (NOT public API)
├── client.py         # Client — low-level robot connection (paths → PathExecutor)
├── event.py          # Event dispatcher
├── exception.py      # Exception classes
├── logger.py         # Logging setup, configure_basic_logging()
├── wifi.py           # WiFi auto-connect helper
├── protocol/         # Cozmo wire protocol
│   ├── protocol_ast.py        # Packet type enums, abstract syntax
│   ├── protocol_base.py       # Struct, Packet base classes
│   ├── protocol_declaration.py # Packet declarations, constants
│   ├── protocol_encoder.py    # Packet encode/decode
│   ├── protocol_generator.py  # Code generation from declarations
│   └── protocol_utils.py      # BinaryReader, BinaryWriter
├── transport/        # Connection and protocol transport
│   ├── conn.py       # Robot connection (socket, threading; robot address constant)
│   ├── frame.py      # Frame encode/decode
│   └── window.py     # Send/receive windowing
├── camera/
│   └── service.py    # CameraService — enable/disable, settings vocabulary, state in force
├── display/
│   ├── procedural_face.py  # Procedural face rendering
│   └── expressions/        # Facial expression definitions
├── vision/
│   ├── annotation.py        # AnnotationEngine — drawing/detection overlays
│   ├── camera.py            # Camera image decoding
│   ├── events.py            # DetectionEventDispatcher — cube/face events
│   ├── face_detector.py     # FaceDetector (Haar cascade)
│   ├── face_identity.py     # FaceIdentity/FaceFrame — face presence epochs (ID per run)
│   ├── frame_stage.py       # FrameStage — per-frame sequence (detect → annotate → publish → world map)
│   ├── frame_state.py       # FrameState — typed container for frame/detection state
│   ├── image_encoder.py     # Image encoding utilities
│   ├── interfaces.py        # Detector/tracker protocols (CubeDetector, etc.)
│   ├── calibration/         # camera_calibration.py (from robot NVRAM), robot_center.py (two-ring detection),
│   │                        # robot_center_calibration.py (single owner of the measured session offset)
│   ├── cube_detector/       # cube_detector.py (CozmoCubeDetector + CubeFusionTracker),
│   │                        # detection_pipeline.py (core detection), cube_faces/ (marker templates)
│   └── occupancy_map/       # floor_pipeline.py, floor_projector.py (bird's-eye view),
│                            # occupancy_map.py (world-frame occupancy grid)
├── navigation/
│   ├── astar.py             # AStarPlanner, PathPoint — A* pathfinding
│   ├── astar_follower.py    # AStarFollower — plan, publish, follow A* path
│   ├── cube_approach.py     # CubeApproach — two-step standoff approach
│   ├── cube_stack.py        # CubeStack — stack the carried cube on another cube
│   ├── cube_visit.py        # CubeVisit — one cube visit at a time (lock + job + start answer)
│   ├── dock_state.py        # DockState — charger/dock pose tracking
│   ├── docking.py           # Docking — dock approach, contact, retries
│   ├── motion_controller.py # MotionController — registry of running motion jobs
│   ├── motion_token.py      # MotionToken — per-job cancellation handle
│   ├── navigator.py         # Navigator — navigate, approach, dock, stop
│   ├── path_executor.py     # PathExecutor — firmware path state machine + lock
│   ├── remote_control.py    # RemoteControl — browser keyboard driving
│   ├── steering.py          # steer_velocities, wrap_pi — differential drive
│   └── world_map.py         # WorldMap — fused robot/cube poses, path trail
├── animation/
│   ├── anim_controller.py   # AnimController — animation scheduling
│   └── cozmo_anim.fbs       # FlatBuffer schema for animation data
├── cubes/                   # Cube BLE management
│   ├── carry_state.py       # CarryDetector/CarryState — cube on the lift, checked on demand
│   ├── cycler.py            # CubeCycler — rapid BLE cycling
│   ├── cube_tracking_state.py # CubeTrackingState — factory mapping, BLE cycler
│   ├── identity.py          # Cube ID 1-3 ↔ cube index 0-2 conversion (the only ±1)
│   └── manager.py           # CubeManager — mapping, connection state
├── web/                     # Web viewer display adapter
│   ├── __init__.py          # Exports: WebViewerServer
│   ├── file_server.py       # FileServer — static file serving
│   ├── mjpeg_streamer.py    # MjpegEndpoint — MJPEG streaming
│   ├── msg_router.py        # MsgRouter — browser WebSocket message dispatch
│   ├── state_builder.py     # build_state_json — world→scene serialization
│   ├── viewer_server.py     # WebViewerServer — thin composition root
│   ├── ws_server.py         # WSServer — WebSocket + HTTP lifecycle
│   └── static/              # Three.js browser viewer (index.html, style.css)
├── sound/                   # Audio encoding + TTS
│   ├── audio.py             # Audio encoding (μ-law)
│   ├── exception.py         # AudioKinetic and TTS exceptions
│   └── tts.py               # TTS engine using pyttsx3 (espeak-ng backend)
├── util/                    # Utility modules
│   ├── util.py              # Angle, Distance, Pose, Vector3, hex_dump, etc.
│   ├── coords.py            # Coordinate transforms (camera→world, world→scene)
│   ├── filter.py            # Signal filtering
│   ├── robot.py             # Robot, Head, LightCube classes
│   ├── robot_debug.py       # Debug message tables
│   └── object.py            # Object tracking
├── lights/
│   └── lights.py            # Color class, predefined light states
├── tests/                   # Unit tests (mocked — run without a robot)
└── tests_live/              # Live-robot tests (need a connected Cozmo)
```

## Branches

Active development happens on **`develop`** — the main line, where everything described here lives. `master` mirrors upstream's default branch.

## Features

- **Public API**: `Cozmo` class (`pycozmo_ng/cozmo.py`) — high-level interface to all robot capabilities
- **Vision pipeline**: DetectionPipeline with pluggable detectors, annotation engine, event dispatcher
- **Floor projection**: Bird's-eye floor view with occupancy mapping
- **Robot center calibration**: Two-ring marker detection for precise lateral offset measurement
- **Face detection**: Haar cascade face detection with expression control
- **3D web viewer**: Three.js browser viewer with WebSocket state broadcast, MJPEG camera streams, occupancy grid overlay, procedural face streaming
- **Navigation**: Navigator with cube approach, cube stacking, docking, point-to-point steering
- **Cube BLE management**: CubeCycler and CubeManager — factory/object mapping, connection state
- **Offline TTS**: `Cozmo.say()` / `Client.say()` using pyttsx3 + espeak-ng, with WAV caching

## Important Notes

### Read CONTEXT.md first
- Always read `CONTEXT.md` before working on vision, navigation, or world-map code — it defines the canonical frame conventions and coordinate transforms that the entire localization stack depends on.

### Public API
- **Use `Cozmo` class** (`from pycozmo_ng import Cozmo`) as the library entry point
- `Brain` is an internal orchestrator — do not import or use directly
- Subsystem APIs are accessed through `Cozmo` properties: `.vision`, `.cubes`, `.carry`, `.navigator`, `.world_map`, `.sensors`, `.web_viewer`

### Camera Calibration
- Cozmo's stored distortion coefficients are unreliable; zero distortion is used by default
- Undistortion is disabled by default (may cause warping)
- Calibration is retrieved from robot NVRAM via `CameraCalibrationRetriever`

### Cube Detection
- Marker templates are in `pycozmo_ng/vision/cube_detector/cube_faces/*.jpg` (6 faces × 3 cubes)
- Cozmo has 3 light cubes (IDs 1-3)
- `CubeFusionTracker` fuses multiple face detections into coherent 3D poses
- Head tilt compensation required for accurate tracking
- `DetectionPipeline` returns camera-frame measurements; world-frame fusion happens in `WorldMap`

### Face Detection
- Face detector uses Haar cascade (`FaceDetector`)
- Face expression changes to `Happiness` when a face is detected
- `face_detect_every_n_frames` setting reduces CPU cost (default: 1 = every frame)

### Floor & Occupancy
- Floor projection creates a bird's-eye view of the area
- OccupancyMap accumulates a world-frame grid over time
- Floor center and size are configurable in mm

### Web Viewer
- Single server on port **8765** (HTTP + WebSocket): `/` serves the viewer page, `/annotated-video` and `/floor-video` are MJPEG streams
- Three.js browser viewer renders robot, cubes, faces, path trail, occupancy grid
- Carried cube in the 3D view: `state_builder` marks the cube `carried: true`
  (plus a robot-pose placeholder if the map lost it); the browser locks that
  cube to the robot's rendered pose — it rides the forks, following heading
  at a fixed forks-up height (never lift telemetry — the render height is
  fixed by design), no lerp of its own
- Procedural face image streamed via WebSocket
- Browser sends keyboard controls, click-to-navigate, cube approach, dock requests
- Camera settings (gain, exposure, resolution, head light) configurable from browser
- Telemetry HUD row "Carry" (cube N / none / unknown) — read from the same
  on-demand CarryDetector the public API uses (`.carry`)

### Dependencies
- Runtime (from `pyproject.toml`): dpkt, numpy, Pillow, flatbuffers, opencv-contrib-python, scipy, aiohttp, websockets, python-dotenv, pyttsx3, setuptools
- System (Linux): `espeak-ng` (TTS backend, `sudo apt-get install espeak-ng`)
- Dev: pytest, ruff, mypy
- JS: eslint (for `web/static`; config not set up yet)
- No CI/CD workflows configured

## Code Style

- **Python**: ruff (max line length 120, PEP 8). Every `.py` file must pass `ruff check` and `ruff format --check` before committing.
- **No `print()`** in library or entry-script code — use the loggers from `pycozmo_ng/logger.py` (`logger`, `logger_cubes`, ...). Levels: info = progress/status, warning = recoverable problems, `logger.exception` in `except` blocks, debug = per-frame/per-packet detail. Do not embed severity (emoji, `WARNING:`, banners) in message text. `print` is allowed only in docstring usage examples, `not_used/`, and throwaway test scripts.
- **JS**: eslint (recommended: `@eslint/js` + `eslint-plugin-unicorn`). Web viewer files live in `pycozmo_ng/web/static/`.
- Type hints encouraged
- Docstrings encouraged

## Changelog

`CHANGELOG.md` records notable changes to PyCozmo-NG, starting fresh at v0.9.0 (upstream pycozmo history is not carried over).

- **Keep it up to date with every release** — when a version ships (or `pyproject.toml` `version` bumps), add a matching `## vX.Y.Z (YYYY-MM-DD)` section to `CHANGELOG.md` in the same change. Never let the changelog lag behind releases.
- For work-in-progress changes, add bullets under a `## Unreleased` section at the top and fold them into the version section at release time.
- Log user-visible changes: features, bug fixes, breaking changes, dependency/config changes. Skip pure refactors unless they change behavior.

## Library-First Approach

PyCozmo-NG is a **library**, not a script collection. New functionality must be added inside `pycozmo_ng/` with a clean public API so others can `import pycozmo_ng` and use it independently. `demo.py` is only a test/demo program that orchestrates and exercises library functions; it must not own reusable behavior, state, hardware integration, or workflows. When adding a feature, ask: "could someone use this without demo.py?" If yes, the API is right.

## Architecture & Organization

The application layer — everything `demo.py` drives — lives in **`pycozmo_ng/`** subpackages. `demo.py` is only an entrypoint (argparse + wiring); no logic lives in `demo.py`.

- **One class per file.** If a feature doesn't fit an existing class's stated job, create a *new file* — never grow an existing class.
- **Files are grouped in folders by function.** `vision/`, `navigation/`, `web/`, `cubes/`. New functionality goes in the matching folder.
- A genuinely new function gets a new folder.
- **Brain is internal.** Use `Cozmo` as the public API. Brain creates typed state containers and wires them to domain classes.
- **Typed state containers** (`FrameState`, `CubeTrackingState`, `DockState`) are passed via DI to domain classes — not through Brain itself.
- **Web viewer is a pure display.** All localization math lives in `WorldMap`; the browser receives scene-frame coordinates only.
- **Pluggable detectors.** `DetectionPipeline` accepts any `CubeDetector`, `CubeFusionTracker`, `FaceDetector` implementation via constructor injection.
- **No backward-compat shims.** Legacy modules are removed, not kept as wrappers (see Project Overview).

## Testing

- `pycozmo_ng/tests/` — unit tests, mocked, run without a robot: `pytest pycozmo_ng/tests/ -v`
- `pycozmo_ng/tests_live/` — integration tests against a live robot (has its own `conftest.py`); requires a connected Cozmo on the robot's Wi-Fi AP
- No automated coverage for the real camera/vision pipeline beyond what the mocked unit tests provide

## Constraints

- Requires Python 3.10+
- Live robot required only for `tests_live/` and manual demos
- No CI/CD configured — lint, type check, and tests run locally before committing
