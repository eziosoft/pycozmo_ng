# PyCozmo-NG

`PyCozmo-NG` is a pure-Python library for controlling a [Cozmo robot](https://www.digitaldreamlabs.com/pages/cozmo)
directly over Wi-Fi. Your program replaces the official Cozmo app: no phone, no cloud — PyCozmo-NG speaks the robot's
own UDP protocol and adds the whole off-board stack on top.

- **Robot control** — wheels, head, lift, LEDs, camera, speaker; low-level `Client` for raw protocol packets.
- **Vision** — light-cube marker detection, standard 4x4 ArUco markers with per-ID physical sizes, charger-dock
  marker detection, face detection, camera calibration read from the robot's NVRAM.
- **Navigation** — fused world map of robot and cube poses, closed-loop point driving, optional A\* obstacle
  avoidance, cube approach & stacking, docking by contact or by ArUco marker.
- **3D web viewer** — live Three.js scene, camera streams, browser driving, click-to-navigate, telemetry HUD.
- **Speech** — offline text-to-speech (pyttsx3 + espeak-ng), from code or from the browser.

Current version **0.9.0** — pre-1.0, expect API changes.

> The distribution is `pycozmo-ng`, the import is `pycozmo_ng` (module names cannot contain hyphens).
> `pycozmo` is the [upstream project](https://github.com/zayfod/pycozmo) this repository was forked from —
> importing it gets you the upstream package, not this one.

## How it differs from the Cozmo SDK

A Cozmo SDK application ("game") talks to the official Cozmo app ("engine") on a phone, which in turn talks to the
robot. With PyCozmo-NG your program *is* the engine:

```
With the official SDK:          With PyCozmo-NG:

  SDK app ("game")                your program
        │ USB                           │
        ▼                               │ UDP / Wi-Fi
  Cozmo app ("engine")                  │
        │ UDP / Wi-Fi                   │
        ▼                               ▼
  Cozmo robot                      Cozmo robot
```

## Requirements

- Python **3.10+**
- A Cozmo robot on its own Wi-Fi AP (factory-default address `172.31.1.1`)
- Optional, for spoken output: `espeak-ng`
  (`sudo apt-get install espeak-ng` on Linux, `brew install espeak-ng` on macOS)

Python dependencies (numpy, OpenCV, Pillow, SciPy, aiohttp, websockets, python-dotenv, pyttsx3, …) are declared in
`pyproject.toml` and install automatically with the package.

## Installation

Not published on PyPI — install from a clone:

```bash
git clone <repository-url>
cd pycozmo

python3 -m venv venv
source venv/bin/activate
pip install -e .
```

For development, add the quality tools:

```bash
pip install pytest ruff mypy
```

## Connecting to the robot

A Wi-Fi connection to the robot must exist before any PyCozmo-NG program runs:

1. Wake Cozmo by placing it on its charging platform.
2. Raise and lower the lift — Cozmo shows its Wi-Fi PSK on its screen.
3. Join the `Cozmo_XXXXXXXX` network with that PSK (per-OS).
   [This video](https://www.youtube.com/watch?v=-k_oiQhBa5o) shows the process.

To let the demo auto-join on Linux (NetworkManager/`nmcli`), put the credentials in `.env` — the same file the
launcher and the demo read:

```bash
cp .env.example .env    # then fill in COZMO_WIFI_SSID / COZMO_WIFI_PASSWORD
```

## Quick start

Minimal script — connect, move, speak:

```python
from pycozmo_ng import Cozmo

cozmo = Cozmo()
with cozmo.connect():                          # robot AP: 172.31.1.1:5551
    cozmo.set_head_angle(30.0)                 # degrees
    cozmo.drive_wheels(left_mmps=50.0, right_mmps=50.0, duration=2.0)
    cozmo.say("Hello!")
```

`connect()` defaults to the robot's own AP (`172.31.1.1:5551`); pass `cozmo.connect(host=...)` when the robot sits
elsewhere on your network.

Full stack — vision, cubes, browser remote control, web viewer:

```python
from pycozmo_ng import Cozmo

cozmo = Cozmo()
with cozmo.connect():
    cozmo.web_viewer.start()                   # → http://<machine-ip>:8765
    cozmo.vision.start(floor_size_mm=(600.0, 600.0))
    cozmo.start_cube_lights()                  # BLE keepalive for the light cubes
    cozmo.discover_cubes()
    cozmo.enable_remote_control()              # keyboard/click control from the browser
    cozmo.set_origin()                         # world origin = current pose
    cozmo.say("Hello, I am Cozmo")
    cozmo.run()                                # blocks until Ctrl-C
```

`demo.py` is the full, annotated version of this startup (calibration, annotation options, tuning flags —
`python demo.py --help`).

### API at a glance

| Access             | What it does                                                             |
| ------------------ | ------------------------------------------------------------------------ |
| `cozmo.vision`     | Start/stop the pipeline, latest frame and detections                     |
| `cozmo.navigator`  | `drive_to`, `approach_cube`, `stack_cube`, `dock`, `dock_on_aruco`, `stop` |
| `cozmo.cubes`      | Cube discovery and BLE keepalive lights                                  |
| `cozmo.carry`      | `is_carrying`, `cube_id` — is a cube riding the forks?                   |
| `cozmo.world_map`  | Fused robot/cube poses and the path trail (world frame, mm)              |
| `cozmo.sensors`    | Battery voltage, IMU, cliff sensor, wheel speeds, head/lift angles       |
| `cozmo.web_viewer` | Start/stop the browser viewer                                            |
| `cozmo.cli`        | The underlying low-level `Client`, for raw protocol access               |

## Running the demo

```bash
./start.sh              # bootstraps .env + venv, waits for the robot, runs demo.py
# or, with the venv already active:
python demo.py
```

Then open **http://\<machine-ip\>:8765** in a browser.

## The 3D web viewer

One server (port **8765**) serves the page, the streams, and the WebSocket:

- **Left** — live camera feeds (`/annotated-video`, `/floor-video`, MJPEG)
- **Center** — Three.js scene: robot, cubes, faces, charger marker, planned path, occupancy grid
- **Right** — telemetry HUD: battery, poses, cube states, carry status

Controls in the browser:

| Input                        | Action                                             |
| ---------------------------- | -------------------------------------------------- |
| `W`/`A`/`S`/`D`              | Drive                                              |
| `R`/`F`                      | Head up/down                                       |
| `T`/`G`                      | Lift up/down                                       |
| Click the floor              | Drive there (`Esc` cancels)                        |
| Click a cube                 | Approach and grab it                               |
| Click another cube while carrying | Stack the held cube on it                     |
| `Z`                          | Dock on the charger (ArUco marker 0)               |
| A\* panel                    | Toggle obstacle-avoiding pathfinding               |
| Camera panel                 | Gain, exposure, resolution, head light             |
| Say box                      | Speak text through the robot (offline TTS)         |

Endpoints: `/` (viewer), `/annotated-video`, `/floor-video` (MJPEG), `/aruco-texture?id=N` (marker textures).

## Configuration

| Variable                                  | Purpose                                                                    |
| ----------------------------------------- | -------------------------------------------------------------------------- |
| `COZMO_WIFI_SSID`, `COZMO_WIFI_PASSWORD`  | Robot AP credentials in `.env`, used for the automatic Wi-Fi join          |
| `COZMO_WIFI_HOST`                         | Robot address used by `start.sh` reachability checks (default `172.31.1.1`) |
| `PYCOZMO_NG_LOG_LEVEL`                    | Log level of the general channel (default `INFO`)                          |
| `PYCOZMO_NG_PROTOCOL_LOG_LEVEL`           | Log level of the protocol channel (default `INFO`)                         |
| `PYCOZMO_NG_ROBOT_LOG_LEVEL`              | Log level of the robot channels (default `WARNING`)                        |
| `PYCOZMO_NG_ARUCO_CONFIG`                 | Path to an ArUco marker-size JSON (see `aruco_markers.json` in the repo)   |
| `PYCOZMO_NG_DIR`                          | Data directory for assets (default `~/.pycozmo_ng`)                        |
| `PYCOZMO_NG_LIVE_MOTION`                  | Set to `1` to let the live test suite physically move the robot            |

## Development

All commands run from the repository root, inside the virtual environment:

```bash
pytest pycozmo_ng/tests/ -v      # unit tests — mocked, no robot needed
ruff check pycozmo_ng/           # lint
ruff format --check pycozmo_ng/  # formatting
mypy pycozmo_ng/                 # types
```

Against a real robot (join the robot's Wi-Fi first):

```bash
pytest pycozmo_ng/tests_live/ -v                      # skips everything if unreachable
PYCOZMO_NG_LIVE_MOTION=1 pytest pycozmo_ng/tests_live/ -v   # also allows motion tests
```

## Documentation

- `CONTEXT.md` — the canonical coordinate frames and transforms used by the vision/navigation stack; read it before
  changing that code.
- [pycozmo.readthedocs.io](https://pycozmo.readthedocs.io/) — wire protocol, hardware, and reverse-engineering
  documentation inherited from the origin project.

## Robot support

Everything the library drives on the robot itself:

- **Sensors** — camera, cliff sensor, accelerometers, gyro, battery voltage, cube battery & accelerometers,
  backpack button (v1.5 hardware and newer)
- **Actuators** — wheel motors, head motor, lift motor, OLED face, speaker, backpack LEDs, IR LED, cube LEDs,
  platform LEDs (when available)
- **On-board** — Wi-Fi AP, Bluetooth LE, localization, path tracking, NV RAM storage, OTA firmware updates

Off-board features implemented here:

- [x] Procedural face generation and expressions
- [x] Cube detection and marker recognition
- [x] ArUco and charger-marker recognition
- [x] Face detection
- [x] Camera calibration (robot NVRAM + robot-center markers)
- [x] Navigation map building (world map + occupancy grid)
- [x] Floor projection
- [x] Text-to-speech
- [x] 3D web viewer with remote control
- [ ] Personality engine
- [ ] Behaviors and reactions
- [ ] Face recognition and expression estimation
- [ ] Pet detection
- [ ] Voice commands
- [ ] Songs

## Related projects

Built on this codebase or its origin, [zayfod/pycozmo](https://github.com/zayfod/pycozmo):

- [Expressive Eyes](https://git.brl.ac.uk/ca2-chambers/expressive-eyes) — facial expressions via the
  `procedural_face` module
- [ROS2 driver](https://github.com/solosito/cozmo_ros2_ws/)
- [Another ROS2 driver](https://github.com/brean/cozmo_ros)

## Origin and license

PyCozmo-NG is a fork of [zayfod/pycozmo](https://github.com/zayfod/pycozmo), credited as the origin of this
codebase, and is loosely based on the [Anki Cozmo Python SDK](https://github.com/anki/cozmo-python-sdk) and
[cozmoclad](https://pypi.org/project/cozmoclad/). It starts fresh at v0.9.0 and does not keep the upstream
`pycozmo` import path.

Licensed under the MIT License — see `LICENSE.md`. Code derived from the Anki Cozmo Python SDK and cozmoclad is
covered by the Apache License 2.0 — see `NOTICE`.

This project is not affiliated with [Digital Dream Labs](https://www.digitaldreamlabs.com/) or [Anki](https://anki.com/).
