# AGENTS.md

## Knowledge Base (LIFIC)

Maintain a running knowledge base on the LIFIC issue tracker (project `COZMO`).

- **Always keep it up to date.** Whenever the user learns or decides something worth remembering (architecture findings, design decisions, gotchas, conventions, debugging tips, deployment notes, Pi/Mac/robot quirks, useful commands), capture it as a LIFIC page or issue comment immediately — do not batch it for later.
- **Fill new info as it surfaces.** After exploring code, solving a problem, or finishing a task, write the takeaways to LIFIC (issue comments, pages, or a dedicated `architecture` / `notes` page) before moving on.
- Treat `AGENTS.md` + the LIFIC pages as a pair: short, project-stable rules live in `AGENTS.md`; deeper, evolving notes (with citations, links, and structure) live on LIFIC.
- Quick reference: `list_resources(type='project')` to find `COZMO`; `lific_search` to find existing notes before adding duplicates.

## Project Overview

PyCozmo is a pure-Python library for direct Cozmo robot control over Wi-Fi, replacing the Cozmo app. This is the `cube_detection` branch implementing computer vision features.

**Agent is responsible for this project.** User reviews results and sets direction. Goal: extend pycozmo with new features, maintain good code quality, and upstream working features as PRs to the real pycozmo repository.

## Quick Start

```bash
# Always use a virtual environment — never install packages system-wide
python3 -m venv venv
source venv/bin/activate

# Install for development
pip install -e .
pip install -r requirements-dev.txt

# Run linting (every Python file must pass ruff before committing)
ruff check pycozmo/
ruff format --check pycozmo/

# Run type checking
mypy pycozmo/

# Run JS linting (for web_viewer JS)
npx eslint pycozmo/web/static/

# Run tests
pytest pycozmo/tests/ -v
```

## Key Commands

| Task | Command |
|------|---------|
| Lint | `ruff check pycozmo/` |
| Format check | `ruff format --check pycozmo/` |
| Type check | `mypy pycozmo/` |
| Test | `pytest pycozmo/tests/ -v` |
| Install | `pip install -e .` |

## Project Structure

```
pycozmo/           # Main library
├── brain.py       # Brain — top-level orchestrator (public API)
├── client.py      # Client class (main entrypoint)
├── event.py       # Event dispatcher
├── exception.py   # Exception classes
├── logger.py      # Logging setup
├── run.py         # connect(), setup_basic_logging()
├── image_encoder.py  # Image encoding utilities
├── remote_control.py # Keyboard remote control
├── wifi.py        # WiFi auto-connect helper
├── protocol/      # Cozmo wire protocol
│   ├── protocol_ast.py       # Packet type enums, abstract syntax
│   ├── protocol_base.py      # Struct, Packet base classes
│   ├── protocol_declaration.py  # Packet declarations, constants
│   ├── protocol_encoder.py   # Packet encode/decode
│   ├── protocol_generator.py # Code generation from declarations
│   └── protocol_utils.py     # BinaryReader, BinaryWriter
├── transport/     # Connection and protocol transport
│   ├── conn.py      # Client connection (socket, threading)
│   ├── frame.py     # Frame encode/decode
│   └── window.py    # Send/receive windowing
├── display/       # Display subsystem
│   ├── procedural_face.py  # Procedural face rendering
│   ├── emotions.py         # Emotion definitions
│   └── expressions/        # Facial expression definitions
├── vision/        # Vision subsystem
│   ├── camera.py          # Camera image decoding
│   ├── processor.py       # AnnotatedVisionProcessor (cube detection pipeline)
│   ├── display.py         # VisionDisplay — frame handlers, display, overlay
│   └── camera_calibration.py  # Camera calibration from robot NVRAM
├── navigation/    # Navigation subsystem
│   ├── navigator.py  # Navigator — navigate, approach, dock
│   └── world_map.py  # WorldMap — fused robot/cube poses, path trail
├── anim/          # Animation and behavior system
│   ├── anim.py          # Anim — animation playback
│   ├── anim_controller.py  # AnimController — animation scheduling
│   ├── anim_encoder.py  # AnimEncoder — animation file encoding
│   ├── activity.py      # Activity system
│   └── behavior.py      # Behavior system
├── util/          # Utility modules
│   ├── util.py      # Angle, Distance, Pose, Vector3, hex_dump, etc.
│   ├── coords.py    # Coordinate transforms (camera→world)
│   ├── filter.py    # Signal filtering
│   ├── robot.py     # Robot, Head, LightCube classes
│   ├── robot_debug.py  # Debug message tables
│   └── object.py    # Object tracking
├── lights/        # LED color and light state
│   └── lights.py    # Color class, predefined light states
├── cubes/         # Cube BLE management
│   ├── cycler.py      # CubeCycler — rapid BLE cycling
│   ├── manager.py     # CubeManager — mapping, connection state
│   └── cube_imu.py    # CubeMoveDetector — accelerometer movement detection
├── web/           # Web viewer bridge
│   ├── viewer.py      # WebBridge — streaming, telemetry
│   ├── viewer_server.py  # WebSocket server for browser viewer
│   └── static/        # Three.js browser viewer (HTML/CSS/JS)
├── cubeDetector/  # Cube detection module
│   ├── cubeDetector.py  # CozmoCubeDetector + CubeFusionTracker
│   └── cube_faces/      # Marker templates (JPGs)
├── audiokinetic/  # SoundBank + audio encoding
├── CozmoAnim/     # Animation data classes (flatbuffers)
└── tests/         # Unit tests (pytest)
```

## Branch: cube_detection

This branch adds:
- Cube detection using marker-based vision (`pycozmo/vision/processor.py`)
- Camera calibration retrieval (`pycozmo/vision/camera_calibration.py`)
- 3D cube tracking and visualization
- Head tilt compensation for accurate positioning

## Important Notes

### Camera Calibration
- Cozmo's stored distortion coefficients are unreliable; zero distortion is used by default
- Undistortion is disabled by default (may cause warping)
- Calibration is retrieved from robot NVRAM via `CameraCalibrationRetriever`

### Cube Detection
- Marker templates are in `pycozmo/cubeDetector/cube_faces/*.jpg`
- Cozmo has 3 light cubes (IDs 1-3)
- `CubeFusionTracker` fuses multiple face detections into coherent 3D poses
- Head tilt compensation required for accurate tracking

### Dependencies
- Runtime: opencv-python, numpy, Pillow, scipy, PyOpenGL, pygame, aiohttp, websockets
- Dev: pytest, ruff, mypy
- JS: eslint (for web/static)
- No CI/CD workflows configured

## Code Style

- **Python**: ruff (max line length 120, PEP 8). Every `.py` file must pass `ruff check` and `ruff format --check` before committing.
- **JS**: eslint (recommended: `@eslint/js` + `eslint-plugin-unicorn`). Web viewer files live in `pycozmo/web/static/`.
- Type hints encouraged
- Docstrings encouraged

## Library-First Approach

PyCozmo is a **library**, not a script collection. New functionality must be added inside `pycozmo/` with a clean public API so others can `import pycozmo` and use it independently. `main.py` is only a test/demo program that orchestrates and exercises library functions; it must not own reusable behavior, state, hardware integration, or workflows. When adding a feature, ask: "could someone use this without main.py?" If yes, the API is right.

## Architecture & Organization

The application layer — everything `main.py` drives — lives in **`pycozmo/`** subpackages. `main.py` is only an entrypoint (argparse + wiring); no logic lives in `main.py`.

- **One class per file.** If a feature doesn't fit an existing class's stated job, create a *new file* — never grow an existing class.
- **Files are grouped in folders by function.** `vision/`, `navigation/`, `web/`, `cubes/`. New functionality goes in the matching folder.
- A genuinely new function gets a new folder.

## Testing

Tests are in `pycozmo/tests/` and `pycozmo/audiokinetic/tests/`. No conftest.py or fixtures found.

## Constraints

- Requires Python 3.10+
- Requires live Cozmo robot connection for vision tests
- No automated tests for vision/cube detection modules

## Deployment Topology (Remote Pi)

Live robot code runs on a Raspberry Pi 5, NOT on the dev Mac.

- **Mac** = dev workstation (`/Users/bartosz.szczygiel/Downloads/pycozmo`), SSH + rsync to Pi.
- **Pi** = `192.168.10.100`, hostname `Pi`, Debian 12 (aarch64), Python 3.13.5.
  - Login: user `e`, password `e`. SSH key from Mac installed (`~/.ssh/id_rsa.pub`) — use key auth, fallback sshpass.
  - **Always use a venv on Pi** — `python3 -m venv ~/pycozmo/venv && source ~/pycozmo/venv/bin/activate`. Never install packages system-wide.
- **Network split**:
  - `eth0` = wired LAN `192.168.10.0/24` — management path (SSH/rsync). Default route stays here (`ipv4.never-default` unset).
  - `wlan0` = Cozmo AP only. SSID **`Cozmo_7144B4`**, password **`973-37134-7253402`**. Set `ipv4.never-default yes` so robot AP never becomes gateway.
- **Robot** = `172.31.1.1:5551` (hardcoded in `pycozmo/transport/conn.py`). Pi wlan0 usually gets `172.31.1.57/24` via DHCP.

### Syncing code to the Pi

**Recommended: use git** — the cleanest way to keep Mac and Pi in sync without touching venvs/caches:

```bash
# On Mac: commit and push
git add -A && git commit -m "..." && git push

# On Pi: pull (and reinstall only if requirements changed)
ssh e@192.168.10.100
cd ~/pycozmo && git pull
source venv/bin/activate && pip install -e . --quiet   # only when deps change
```

**Alternative: rsync** — fine for quick one-off pushes, but **must use `--exclude-from`** so caches, the local `.venv/`, `.env`, and build artifacts don't get copied or (with `--delete`) wiped on the Pi:

```bash
rsync -avz --delete \
  --exclude-from='.gitignore' \
  --exclude='.git/' --exclude='.env' --exclude='.venv/' --exclude='venv/' \
  --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.ruff_cache/' --exclude='.mypy_cache/' --exclude='.pytest_cache/' \
  --exclude='pycozmo.egg-info/' \
  ./ e@192.168.10.100:~/pycozmo/
```

Never run a bare `rsync -avz --delete ./` — it will wipe the Pi's `venv/` and other state.

- Raspberry Pi will serve the Three.js web viewer (project in progress).

### Reconnect Pi → Cozmo (if robot rebooted)

On Pi (via SSH):
```
sudo nmcli dev wifi connect Cozmo_7144B4 password '973-37134-7253402'
```
Verify: `ip route` must show `default via 192.168.10.1 dev eth0`. Ping robot: `ping 172.31.1.1`.

Note: wifi connection saved with `autoconnect yes`; NetworkManager auto-reconnects when AP visible.
