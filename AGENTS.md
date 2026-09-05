# AGENTS.md

## Project Overview

PyCozmo is a pure-Python library for direct Cozmo robot control over Wi-Fi, replacing the Cozmo app. This is the `cube_detection` branch implementing computer vision features.

**Agent is responsible for this project.** User reviews results and sets direction. Goal: extend pycozmo with new features, maintain good code quality, and upstream working features as PRs to the real pycozmo repository.

## Quick Start

```bash
# Install for development
pip install -e .
pip install -r requirements-dev.txt

# Run linting
flake8 pycozmo/

# Run type checking
mypy pycozmo/

# Run tests
pytest pycozmo/tests/ -v
```

## Key Commands

| Task | Command |
|------|---------|
| Lint | `flake8 pycozmo/` |
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
- Runtime: opencv-python, numpy, Pillow, scipy, PyOpenGL, pygame
- Dev: pytest, flake8, mypy
- No CI/CD workflows configured

## Code Style

- Max line length: 120 (flake8)
- PEP 8 compliant
- Type hints encouraged
- Docstrings encouraged

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
- **Network split**:
  - `eth0` = wired LAN `192.168.10.0/24` — management path (SSH/rsync). Default route stays here (`ipv4.never-default` unset).
  - `wlan0` = Cozmo AP only. SSID **`Cozmo_7144B4`**, password **`973-37134-7253402`**. Set `ipv4.never-default yes` so robot AP never becomes gateway.
- **Robot** = `172.31.1.1:5551` (hardcoded in `pycozmo/conn.py`). Pi wlan0 usually gets `172.31.1.57/24` via DHCP.
- **File sync**: `rsync -avz --delete ./ e@192.168.10.100:~/pycozmo/` (key auth, no password). Run from repo root.
- Raspberry Pi will serve the Three.js web viewer (project in progress).

### Reconnect Pi → Cozmo (if robot rebooted)

On Pi (via SSH):
```
sudo nmcli dev wifi connect Cozmo_7144B4 password '973-37134-7253402'
```
Verify: `ip route` must show `default via 192.168.10.1 dev eth0`. Ping robot: `ping 172.31.1.1`.

Note: wifi connection saved with `autoconnect yes`; NetworkManager auto-reconnects when AP visible.
