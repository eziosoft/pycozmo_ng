# PyCozmo-NG World Map Context

The world-localization subsystem that fuses Cozmo's onboard pose, camera-based
cube detection, and the driven path into one coherent map, rendered in a 3D viewer.

## Language

**World frame**:
The single canonical coordinate frame for the whole map: **X forward, Y left,
Z up**, in millimetres (**Anki convention** — verified from `pycozmo_ng/util/coords.py`
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
world frame. Owned by the `WorldMap` class (`pycozmo_ng/navigation/world_map.py`),
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
(`pycozmo_ng/vision/frame_stage.py`): convert → detect → annotate → publish →
dispatch events → feed the world map (plus floor projection and face
expression). One small interface — `configure(...)` (once per vision setup)
and `on_camera_image(...)` (once per frame); `Brain` only constructs and
wires it. Stable deps (state containers, world map, the frame sink below)
come from the constructor, detectors from `configure()`.
_Avoid_: FrameHandler, WebStreamer, WebBridge (deleted); calling Brain the
place where frames are processed; typing the constructor against
`WebViewerServer` (its display calls leaked across the seam — deleted).

**Frame sink** (`pycozmo_ng/vision/frame_sink.py`, protocol `FrameSink`):
The display-side seam behind `FrameStage`: everything the frame stage
publishes to the browser crosses it — `push_annotated_frame`,
`push_floor_frame`, `push_face_image`, `push_occupancy_frame`,
`set_floor_grid` (size + center in one call), `broadcast`. Two adapters
make the seam real: `WebViewerServer` in production, the recording
`RecordingFrameSink` in `tests/test_frame_stage.py`, which is where the
per-frame sequence is asserted (the interface is the test surface).
`vision/` therefore never imports `web/`.
_Avoid_: importing `WebViewerServer` from `vision/`; reaching through
the sink into its parts (`sink.occupancy.push` — deleted); one setter per
floor-grid field.

**Camera service**:
The single owner of the camera lifecycle, in the `CameraService` class
(`pycozmo_ng/camera/service.py`): enable/disable, the settings vocabulary
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

**Occupancy map** (`vision/occupancy_map/occupancy_map.py`, class `OccupancyMap`):
The world-frame free/obstacle grid A* plans on. Classification is
**geometry+appearance**, not brightness/edges: an **adaptive floor model**
(the Lab median of the trusted near-robot strip, EMA-smoothed — any floor
works, no fixed brightness threshold), **shadow tolerance** (distance
down-weights lightness, leans on chroma), the **contact line** (only the near
edge of a non-floor region — where it touches the floor — votes occupied, the
one part of the z=0-flattened projection at the true ground position), and
**log-odds** accumulation (`+logodds_hit` per contact cell, `-logodds_miss`
per free cell, one step per cell per frame; sign classifies, 0 = unknown).
`logodds_hit > logodds_miss` is the damped clearing: a new obstacle appears
in one frame, an established one survives several contradictory frames, then
clears. Decay is off by default (a decaying map fades to UNKNOWN outside the
FOV — rejected live, COZMO-DOC-9). Grid contract for consumers: int8 with
`FREE`/`OCCUPIED`/`UNKNOWN`; knobs are the `occupancy_*` fields of
`VisionSetup`.
_Avoid_: brightness thresholds / Canny edges for occupancy (deleted —
misfired on shadows and texture in both directions); voting the whole
projected silhouette (smears occupied cells past the obstacle base); decaying
votes by default.

**Coordinate transforms** (`pycozmo_ng/util/coords.py`):
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
output (`marker_id // 6`), `WorldMap` keys, and navigation's `cube_idx`,
always `index = cube_id - 1`.
All conversion lives in **one module** — `pycozmo_ng/cubes/identity.py`
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
geometry): `pycozmo_ng/vision/face_identity.py`, class `FaceIdentity`.
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

**Robot center calibration** (`vision/calibration/robot_center_calibration.py`):
Two-ring marker detection on the robot chassis. Head tilted to max-down
(-25°) to reveal two black rings with white centres. The 20 mm baseline
between ring centres gives the lateral offset of the camera from robot centre.
Measured once during startup calibration and frozen for the session.
`RobotCenterCalibration` is the single owner of that measurement: a run calls
`arm()`, each processed frame calls `observe()` (one fresh reading per frame
that sees the rings), and only `freeze(avg)` writes the session value —
`offset_mm` is the only read. Outside a run the detector does no work, so a
stray head-down frame can never latch the value.
_Avoid_: poking a private latch on the detection pipeline; polling a stored
result on a timer (the same measurement gets counted twice).

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

**Motion controller** (`pycozmo_ng/navigation/motion_controller.py`):
The single registry that knows which motion jobs are running
(`active_names`), the only cancellation authority — `cancel(name)` kills
jobs by name, `cancel_all()` bumps a generation counter so *everything live*
stops — and the owner of the **driver lease** (below). Anything begun
afterwards starts un-cancelled, which is what re-arms the robot after
`Navigator.stop()` (replacing `_stop_requested.clear()`). `Brain` creates
the one instance and injects it into `Navigator` (`.motion`) and
`RemoteControl`; `Navigator` hands the same object to `Docking`;
`CubeApproach` receives its `spawn` callable by injection.
_Avoid_: per-feature cancellation flags/locks, spawning bare
`threading.Thread`s for motion work, a second place that decides who may
drive.

**Driver lease**:
The wheel-command authority held by exactly one driver at a time, recorded
on the `MotionController` (`lease_owner`: a job name, `"remote"`, or
`None`). A *root* motion job (no enclosing job on its thread) takes it in
`begin()` — cancelling the previous holder, last requester wins — while
nested jobs ride their parent's (`approach → drive → align` never
contends with itself). Non-job drivers take/release it explicitly
(`acquire_lease` / `release_lease`): the browser `RemoteControl` takes it
on the *edge* of fresh drive-key input (w/a/s/d), so a key merely held
never steals control back from a programmatic sequence, and goes silent
the moment it loses the lease. Only the holder sends `drive_wheels` /
`go_to_pose` / `execute_polyline`; `stop_all_motors` is always free, and
head/lift keys are ungated. `cancel_all()` drops the lease, so
`Navigator.stop()` silences held keys until the next key edge or root job
asks again. `Client` additionally serializes each path build-and-wait on
one lock (the firmware path buffer is a single resource).
_Avoid_: a `drive_wheels` writer outside the lease; per-feature "may
drive" flags; re-triggering the lease from a held (unchanged) key set.

**Motion token** (`pycozmo_ng/navigation/motion_token.py`):
The cancellation handle a single job polls — `token.cancelled`,
`token.cancel()`, usable as a `with` scope. Cancellation is the token's own
flag, a generation comparison, and its parent's answer; nesting outranks the
generation, so a dying parent takes children begun after a `cancel_all()`.
_Avoid_: reading the controller's internals from a job.

**Path executor** (`pycozmo_ng/navigation/path_executor.py`):
The firmware path-following state machine, owned by the `PathExecutor`
class: settle → allocate an event id → `ClearPath` → `AppendPathSeg*` →
`ExecutePath` → wait for a `PathFollowingEvent` **or** pose proximity
(COZMO-53), plus the chunked-polyline loop and the single lock that
serializes it all (the firmware buffer holds one path at a time).
Interface: `go_to_pose(...)` and `execute_polyline(...)` — everything
else is behind them. It reaches the robot only through the `PathRobot`
port (`send`, `stop_all_motors`, `pose`, wheel speeds, `add_handler` /
`del_handler`), which `Client` satisfies; `cli.paths` is the instance and
`Client.go_to_pose()` / `Client.execute_polyline()` are one-line
delegations, so navigation callers (`Navigator`, `AStarFollower`,
`Docking`, `CubeApproach`) never touch it directly. The pure helpers
(`chunk_polyline_points`, `build_polyline_packets`) live beside it.
_Avoid_: reaching path logic through `client.py` or importing the helpers
from `pycozmo_ng.client` (moved); a second path lock or event-id counter
outside `PathExecutor`; sending path packets without that lock.

**Navigator** (`pycozmo_ng/navigation/navigator.py`):
The navigation subsystem's one entry point — `navigate_to`, `drive_to_point`,
`approach_cube*`, `stack_cube`, `dock`, `stop`, and the A* toggle — so callers
address navigation as a single thing. It receives `Brain`'s shared
`MotionController` (`.motion`) and owns the `DockState`, and composes four
sequences that hold their own tuning and job names: `AStarFollower` (`.astar` — plan, publish, and follow an A* path),
`Docking` (`.docking` — dock approach, reverse into contact, retries), the
lazily built `CubeApproach` (`_cube_approach`) and `CubeStack`
(`_cube_stack`) — both wired to the one `CubeVisit` (below), whose
"did it start?" answer `approach_cube*` / `stack_cube` return.
`Navigator.cli` is a property:
assigning it (Brain wiring, or a reconnect) propagates the client to all of
them.
_False friend_: "delegate" here does not mean callers reach past Navigator —
they keep using `navigator.dock()` / `navigator.set_astar_enabled()`;
sub-classes are reached only through `navigator.astar` and `navigator.docking`.

**Carry state** (`pycozmo_ng/cubes/carry_state.py`):
Whether a cube is riding the lift — read it whenever you want via
`cozmo.carry.state` / `cozmo.carry.is_carrying` (`True`/`False`/`None` for
unknown) / `cozmo.carry.cube_id`. Nothing polls: `CarryDetector.state` is
evaluated on each read, using the **post-grab visibility heuristic** — a
grabbed cube sits on the raised forks in front of/above the camera. Two
phases: after `note_grab` the detector *waits for the fork to finish
rising* (lift motion observed through the `LIFT_IN_POS` status flag, or
height ≥ `MIN_CARRY_LIFT_MM`) — no verdict mid-travel, and a lagging
`lift_height_mm` reading never disarms the observation (known issue:
live feedback can read low right after a grab). Once the fork is up,
visibility decides: "cube re-detected since the grab" = grab missed
(`NOT_CARRYING`), "no update within `LOST_S` (0.3 s — detections run at
~30 Hz, so the stale *pre-grab* sighting never counts as still-in-frame
right after the rise)" = it is on the forks (`CARRYING`). Seeing the
cube **resolves** the check: the `NOT_CARRYING` verdict latches until
the next `note_grab`, so driving away from a left-behind cube (it leaves
the frame) can never flip the status back to `CARRYING`. A fork settled
*down* that never rose within
`LIFT_TRANSIT_S` of the grab is authoritative `NOT_CARRYING` and clears
the observation; lift up with no witnessed grab is `UNKNOWN`.
`on_forks` is the derived "may I stack?" predicate: armed observation and
not resolved-miss → `True` (carrying, or the fork is still rising),
otherwise `False`.
`CubeApproach` owns the hooks: it calls `clear()` when the
forks go down at approach start and `note_grab(cube_idx)` after the grab
(lift-up) — Brain wires the detector (`brain.carry`) into Navigator →
CubeApproach, and `Cozmo.carry` is a pure reader. State transitions log
with the raw reading (`Carry: carrying - cube 2 out of frame ... (lift
92.0mm)`), so live logs show the verdict and the lift value that fed it.
The 3D viewer renders a carrying cube **locked to the robot**:
`build_state_json` flags it `carried: true` (appending a robot-pose
placeholder if the world map lost the cube out of frame) and `index.html`
re-positions that cube every frame on the forks — robot heading at a
fixed forks-up height (`CARRY_H_MM`, fixed by design: the render locks
to the robot, not to lift telemetry), no lerp of its own, so it
never drifts off the robot.
_Avoid_: background polling or a cached boolean (the state is decided at
read time); treating "lift up" alone as "carrying" (it is `UNKNOWN` without
a witnessed grab); judging while the lift motor is travelling; assuming
the heuristic survives a cube driven back into
frame — re-`note_grab` after the next grab to resynchronise.

**Lift feedback** (`pycozmo_ng/util/robot.py`):
`RobotState.lift_height_mm` is misnamed — the firmware reports the lift
arm **angle in radians** (live: ≈ -0.2 with forks down, ≈ 0.8 with them
up = `MIN_LIFT_ANGLE`/`MAX_LIFT_ANGLE` for the 66 mm arm on a 45 mm
pivot). `client._on_robot_state` converts it through
`robot.lift_position_from_raw()` (`sin(a)·66 + 45`, clamped to the arm's
travel), so `cli.lift_position.height.mm`, `cozmo.sensors.lift_height`,
the carry logs and the web HUD's LIFT row read real millimetres
(32…92). Carry still keys on `LIFT_IN_POS` first — height is only the
fallback, never the settle signal.
_Avoid_: reading `pkt.lift_height_mm` raw (radians, not mm); using a
height threshold as the "lift settled" signal.

**Cube visit** (`pycozmo_ng/navigation/cube_visit.py`):
One trip to a cube — a grab approach or a stack — run as a single motion
job with exclusive rights to the wheels. The `CubeVisit` class owns the
decision: `begin(job_name, run)` returns True only when the visit
actually started (False when another visit is in flight — refused, never
queued), holds the lock until the job's `finally`, and exposes `busy`.
`Navigator` builds the one instance and injects the *same* object into
`CubeApproach` and `CubeStack`, so the two can never interleave and
cannot be wired onto separate locks by accident. `approach_cube`,
`approach_cube_at`, and `stack_cube` return that answer verbatim, which
is what makes `cozmo.stack_cube(id)`'s "True if started" contract honest.
_Avoid_: a bare lock passed between classes (`CubeApproach.approach_lock`
— deleted); starting the visit from a class that does not hold the
lock; reporting `True` for a refused visit; queueing a second visit.

**Cube stacking** (`pycozmo_ng/navigation/cube_stack.py`):
Putting the carried cube on top of a second cube. Trigger: click another
cube in the viewer while `carry.on_forks`, or the explicit
`cozmo.navigator.stack_cube(id)` — `Navigator.approach_cube_at` is the
single routing seam (no browser-side state): held cube, or nothing on the
forks → the normal grab approach; a *different* cube while holding →
`CubeStack`. The sequence reuses `CubeApproach.drive_and_align(...,
lower_forks=False)` — the forks stay **up** the whole way (the usual
forks-down prelude would dump the held cube) — then places: creep
`GRAB_FORWARD_MM` over the target (same geometry as the grab), lower to
`PLACE_LIFT_MM` (72 mm, live-tuned: bring the cube down to the stack
while keeping the prongs clear of the 44 mm cube below), back away
`STACK_BACK_AWAY_MM` — the prongs slide out from under the carried cube
and it drops the remaining gap onto the stack — then forks fully down,
`carry.clear()`. A failed back-away skips the final lowering — dropping
the forks while still over the stack would push it over — and leaves
them at `PLACE_LIFT_MM`. Shares the one `CubeVisit` with the approach
(one cube visit at a time) and runs as the
`stack` motion job; the post-lower settle wait is time-based
(`set_lift_height` returns as soon as the command is sent).
_Avoid_: lowering the forks before alignment; `note_grab` in a stack
flow; timing the release from lift feedback (the sequence is
command-timed); assuming the
target cube stays visible at close standoffs with the carried cube in
front of the camera (dwell re-detection gates the approach).

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
