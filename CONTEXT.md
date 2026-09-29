# PyCozmo World Map Context

The world-localization subsystem that fuses Cozmo's onboard pose, camera-based
cube detection, and the driven path into one coherent map, rendered in a 3D viewer.

## Language

**World frame**:
The single canonical coordinate frame for the whole map: **X forward, Y left,
Z up**, in millimetres (**Anki convention** — verified from `pycozmo/util/coords.py`
`WORLD_TO_SCENE_MATRIX`, forward on the X row). Yaw is rotation about Z (CCW positive).
Origin is the current Cozmo localization origin (re-anchored on pick-up /
put-down). All entities (Cozmo, cubes, path) are stored in this frame.
_Avoid_: "X right, Y forward, Z up" (a common but wrong statement of the Anki frame)

**Body frame**:
A robot-relative frame, **X forward, Y left, Z up**, with origin at the front-centre
ground point directly under the camera. The intermediate frame between camera and
world.
_Avoid_: robot frame (ambiguous)

**Camera frame**:
The OpenCV camera coordinate frame (X right, Y down, Z forward) used by the marker
pose estimator (solvePnP). Raw cube measurements start here.
_Avoid_: pixel space, image space

**Scene frame**:
The Three.js render coordinate frame (right-handed, X right, Y up, Z forward). The
web viewer draws in this frame; the browser is a pure display and does no frame math.
_Avoid_: world frame (when describing what the browser consumes)

**Camera-to-body offset**:
The fixed translation from the camera origin to the body origin, `[30, 0, 38]` mm
(camera sits 30 mm back from front edge, centred laterally, 38 mm above the ground).
Applied in `camera_to_body()` along with the head-tilt rotation.
_Avoid_: camera height, `+40` hack

**World map**:
The single source of truth holding Cozmo's pose, cube poses, and the path, all in
world frame. Owned by the `WorldMap` class (`pycozmo/navigation/world_map.py`),
constructed by `Brain` (the composition root) and injected into every reader and
writer (`Navigator`, `FrameStage`, `MsgRouter`, `WebViewerServer`,
`Cozmo.WorldMapAPI`). The web server and viewer are thin adapters over it and
expose no world-map writes.
_Avoid_: scene state, viewer state; reaching the world map through the web viewer

**Scene yaw**:
The world-frame yaw maps to scene `rotation.y` **unchanged** (`scene_yaw = world_yaw`).
The world→scene basis change is a proper rotation (det = +1) taking world forward
(+X) to scene forward (+Z). Chirality is preserved: a clockwise drive-out in real
life still renders clockwise in the top-down 3D view; a right turn renders as a
right turn.
_Avoid_: negating the yaw, or using a reflection (det = -1) world→scene basis —
either one mirrors every turn/circle (real right turn appears as a left turn).

**Browser action**:
A named listener on the message router (`MsgRouter.ACTIONS`: control, navigate,
approach, dock, camera_settings, say, test_mode, astar_toggle) through which the
web viewer commands the library. Registered in one call —
`web_viewer.router.set_actions(...)` from `Brain.install_remote_control`;
unknown names raise at registration.
_Avoid_: callback, handler (when naming the registered seam); per-action
`set_*_listener` methods (deleted — they duplicated the router's interface).

**Frame stage**:
The per-frame vision sequence, owned by the `FrameStage` class
(`pycozmo/vision/frame_stage.py`): convert → detect → annotate → publish →
dispatch events → feed the world map (plus floor projection and face
expression). One small interface — `configure(...)` (once per vision setup)
and `on_camera_image(...)` (once per frame); `Brain` only constructs and
wires it. Stable deps (state containers, world map, `WebViewerServer`) come
from the constructor, detectors from `configure()`.
_Avoid_: FrameHandler, WebStreamer, WebBridge (deleted); calling Brain the
place where frames are processed.

**Camera service**:
The single owner of the camera lifecycle, in the `CameraService` class
(`pycozmo/camera/service.py`): enable/disable, the settings vocabulary
(`DEFAULTS` — the one defaults table), the resolution name↔enum map, and
the one packet sequence (`enable_camera` → `set_camera_params` →
`set_head_light`) behind `enable(...)`, `disable()`, `apply_settings(...)`.
It records what is actually in force (`settings`, `active`,
`enabled_at`, `resolution`), so the browser `camera_settings` action is
registered as `camera.apply_settings` directly and the echo the viewer
broadcasts is the applied state by construction. `Brain` constructs it
and injects it into `WebViewerServer`; the `cli` setter propagates the
client to it.
_Avoid_: calling `cli.enable_camera` / `cli.set_camera_params` /
`cli.set_head_light` from anywhere else (facade, demo, Brain);
storing camera state in `FrameState` or `MsgRouter` (deleted); a second
settings defaults table.

**Fusion**:
The process of combining noisy, repeated cube measurements into one stable
world-frame estimate using confidence-gated, confidence-weighted exponential
smoothing (position) and SLERP (rotation).
_Avoid_: averaging, filtering (when it erases world-frame stability)

**Coordinate transforms** (`pycozmo/util/coords.py`):
- `camera_to_body()` — applies head-tilt rotation + camera-to-body offset
- `body_to_world()` — applies robot yaw + translation
- `camera_to_world()` — convenience: camera → body → world in one step
- `world_to_scene_point()` — maps `(x_fwd, y_left, z_up)` → `(y_left, z_up, x_fwd)`
- `world_to_scene_rotation()` — change-of-basis rotation for 3x3 matrices
- `world_yaw_to_scene()` — yaw passes through unchanged (det = +1)
- `scene_to_world_point()` — inverse of `world_to_scene_point`

**Cube ID vs cube index**:
Two numbering conventions exist for the same three cubes and must not be
mixed inside one call. **Cube ID 1-3** (`cube_id`) is user-facing: the
physical cubes, the `CubeManager` factory/connection maps, browser wire
labels (`cube_1`), public navigation (`Navigator.approach_cube`), detection
events (`EvtCubeLost`), and the whole `Cozmo` facade
(`cozmo.world_map.cubes`, `cozmo.navigator.approach_cube`, `cozmo.cubes.*`).
The **cube index 0-2** (`cube_index` / `cube_idx`) is internal: detection
output (`marker_id // 6`), `WorldMap` keys, `CubeTrackingState`
suppression, and navigation's `cube_idx`, always `index = cube_id - 1`.
All conversion lives in **one module** — `pycozmo/cubes/identity.py`
(`to_world_index`, `to_cube_id`, `cube_label`, `parse_cube_label`) — and
is validated at the boundary (`cube_99` and `cube_id=0` raise). Every
boundary (facade, `MsgRouter`, `state_builder`, event dispatch) converts
through it.
_Avoid_: raw `cube_id ± 1` anywhere except `cubes/identity.py`; calling
the cube index "cube ID" in public API; naming an index variable
`cube_id`.

**Face index mapping** (cube marker faces, 0-5):
| face_idx | face_type | rotation | Marker file |
|----------|-----------|----------|-------------|
| 0 | wall | 0° | cube{N}_face_0.jpg |
| 1 | wall | 90° | cube{N}_face_1.jpg |
| 2 | wall | 180° | cube{N}_face_2.jpg |
| 3 | wall | 270° | cube{N}_face_3.jpg |
| 4 | bottom | 0° | cube{N}_face_bottom.jpg |
| 5 | top | 0° | cube{N}_face_top.jpg |

**Face identity** (not the marker-face table above — that is cube surface
geometry): `pycozmo/vision/face_identity.py`, class `FaceIdentity`.
Assigns an ID to each **presence run** of a human face: one uninterrupted
stretch of measured frames with a face in view. IDs start at 1 and are
never reused within a session; the pipeline cannot re-identify people, so
a face that leaves and returns gets a **new** ID. `FrameStage` calls
`FaceIdentity.update(detections, measured=...)` once per frame, before
annotation reads `detection.face_id`; a skipped face frame
(`face_detect_every_n_frames`) passes `measured=False` and holds state —
"not measured" is never "gone". The returned `FaceFrame`
(`face_id` / `started` / `ended`) drives `EvtFaceDetected`/`EvtFaceLost`,
is the `WorldMap` face key, and `ended` triggers `WorldMap.forget_face`.
_Avoid_: `getattr(det, "_face_id")` probing (deleted — read the field);
hardcoding `face_id: 0`; treating a frame skip as face loss.

**Robot center calibration**:
Two-ring marker detection on the robot chassis. Head tilted to max-down
(-25°) to reveal two black rings with white centres. The 20 mm baseline
between ring centres gives the lateral offset of the camera from robot centre.
Measured once during startup calibration and frozen for the session.

**Motion job**:
One cancellable unit of background motion (navigate/A* follow, drive, align,
dock, cube approach, test mode). Started either as a scope on the current
thread (`with motion.begin(...)`) or on a daemon thread
(`motion.spawn(name, fn)`), and polled by the job through its token
(`job.cancelled`). Job names are constants on the owning class
(`Navigator.JOB_NAVIGATE`, `Docking.JOB_DOCK`, `CubeApproach.JOB_APPROACH`, …).
Jobs nest: a job
started inside another is its child and dies with it.
_Avoid_: an ad-hoc `bool` flag (`_path_following`, `_stop_requested`,
`_test_mode`) for "should this keep running".

**Motion controller** (`pycozmo/navigation/motion_controller.py`):
The single registry that knows which motion jobs are running
(`active_names`), and the only cancellation authority: `cancel(name)` kills
jobs by name, `cancel_all()` bumps a generation counter so *everything live*
stops — and anything begun afterwards starts un-cancelled, which is what
re-arms the robot after `Navigator.stop()` (replacing
`_stop_requested.clear()`). `Navigator` owns one instance (`.motion`) and hands
the same object to `Docking`; `CubeApproach` receives its `spawn` callable by
injection.
_Avoid_: per-feature cancellation flags/locks, spawning bare
`threading.Thread`s for motion work.

**Motion token** (`pycozmo/navigation/motion_token.py`):
The cancellation handle a single job polls — `token.cancelled`,
`token.cancel()`, usable as a `with` scope. Cancellation is the token's own
flag, a generation comparison, and its parent's answer; nesting outranks the
generation, so a dying parent takes children begun after a `cancel_all()`.
_Avoid_: reading the controller's internals from a job.

**Navigator** (`pycozmo/navigation/navigator.py`):
The navigation subsystem's one entry point — `navigate_to`, `drive_to_point`,
`approach_cube*`, `dock`, `stop`, and the A* toggle — so callers address
navigation as a single thing. It owns the shared `MotionController` and the
`DockState`, and composes three sequences that hold their own tuning and job
names: `AStarFollower` (`.astar` — plan, publish, and follow an A* path),
`Docking` (`.docking` — dock approach, reverse into contact, retries), and the
lazily built `CubeApproach` (`_cube_approach`). `Navigator.cli` is a property:
assigning it (Brain wiring, or a reconnect) propagates the client to all three.
_False friend_: "delegate" here does not mean callers reach past Navigator —
they keep using `navigator.dock()` / `navigator.set_astar_enabled()`;
sub-classes are reached only through `navigator.astar` and `navigator.docking`.

## Example dialogue

**Dev**: The cube drifts when Cozmo turns. Why?
**Expert**: Cubes were stored in camera frame, so they moved with the camera. The
world map must store them in world frame — transform each camera-frame measurement
through body frame (add the camera-to-body offset, apply head-tilt rotation) then
through the robot's yaw and position into world frame, and fuse there. Then a
stationary cube stays put as Cozmo drives.

**Dev**: And the box misalignment?
**Expert**: The box yaw was derived and applied without accounting for the
world-to-scene reflection. Use Cozmo's true yaw and one shared world-to-scene
transform; the browser just renders the scene-frame coordinates the library sends.

**Dev**: Why is the camera-to-body offset [30, 0, 38] and not [0, 0, 40]?
**Expert**: The camera sits 30 mm back from the front edge (not at the front edge)
and 38 mm above the ground (not 40 mm). The lateral offset (Y component) is 0
because the camera is centred, but a measured lateral offset is added during
startup calibration via `robot_center.py`.
