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
npx eslint pycozmo/web_viewer/

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
├── client.py      # Client class (main entrypoint)
├── vision.py      # Vision processor for cube detection
├── camera_calibration.py  # Camera calibration from robot NVRAM
├── cubeDetector/  # Cube detection module
│   ├── cubeDetector.py  # CozmoCubeDetector + CubeFusionTracker
│   ├── cube_faces/      # Marker templates (JPGs)
│   └── world_3d_viewer.py  # 3D visualization
├── web_viewer_server.py  # WebSocket server for browser viewer
├── web_viewer/     # Three.js browser viewer
│   ├── index.html
│   └── style.css
├── tests/         # Unit tests (pytest)
└── audiokinetic/  # SoundBank module (has own tests)

examples/          # Demo scripts (run with python examples/X.py)
tools/             # CLI utilities (installed as scripts)
main.py            # Vision system test (connects to Cozmo)
```

## Branch: cube_detection

This branch adds:
- Cube detection using marker-based vision (`pycozmo/vision.py`)
- Camera calibration retrieval (`pycozmo/camera_calibration.py`)
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
- JS: eslint (for web_viewer)
- No CI/CD workflows configured

## Code Style

- **Python**: ruff (max line length 120, PEP 8). Every `.py` file must pass `ruff check` and `ruff format --check` before committing.
- **JS**: eslint (recommended: `@eslint/js` + `eslint-plugin-unicorn`). Web viewer files live in `pycozmo/web_viewer/`.
- Type hints encouraged
- Docstrings encouraged

## Library-First Approach

PyCozmo is a **library**, not a script collection. New functionality must be added inside `pycozmo/` with a clean public API so others can `import pycozmo` and use it independently. `main.py` is a simple test/demo script that exercises the library — never put core logic there. When adding a feature, ask: "could someone use this without main.py?" If yes, the API is right.

## Testing

Tests are in `pycozmo/tests/` and `pycozmo/audiokinetic/tests/`. No conftest.py or fixtures found.

## Constraints

- Requires Python 3.6.0+
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
- **Robot** = `172.31.1.1:5551` (hardcoded in `pycozmo/conn.py`). Pi wlan0 usually gets `172.31.1.57/24` via DHCP.

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
