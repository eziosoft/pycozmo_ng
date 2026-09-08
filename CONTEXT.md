# PyCozmo World Map Context

The world-localization subsystem that fuses Cozmo's onboard pose, camera-based
cube detection, and the driven path into one coherent map, rendered in a 3D viewer.

## Language

**World frame**:
The single canonical coordinate frame for the whole map: **X forward, Y left,
Z up**, in millimetres (**Anki convention** — verified from `pycozmo/util.py`
`Matrix44`, forward on the X row). Yaw is rotation about Z (CCW positive).
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
The fixed translation from the camera origin to the body origin, `[0, 0, 40]` mm
(camera sits on the front-centre, 40 mm above the ground). Applied when moving a
measurement from camera frame to body frame.
_Avoid_: camera height, `+40` hack

**World map**:
The single source of truth holding Cozmo's pose, cube poses, and the path, all in
world frame. Owned by the `WorldMap` library class; the web server and viewer are
thin adapters over it.
_Avoid_: scene state, viewer state

**Scene yaw**:
The world-frame yaw maps to scene `rotation.y` **unchanged** (`scene_yaw = world_yaw`).
The world→scene basis change is a proper rotation (det = +1) taking world forward
(+X) to scene forward (+Z). Chirality is preserved: a clockwise drive-out in real
life still renders clockwise in the top-down 3D view; a right turn renders as a
right turn.
_Avoid_: negating the yaw, or using a reflection (det = -1) world→scene basis —
either one mirrors every turn/circle (real right turn appears as a left turn).

**Fusion**:
The process of combining noisy, repeated cube measurements into one stable
world-frame estimate using confidence-gated, confidence-weighted exponential
smoothing (position) and SLERP (rotation).
_Avoid_: averaging, filtering (when it erases world-frame stability)

## Example dialogue

**Dev**: The cube drifts when Cozmo turns. Why?
**Expert**: Cubes were stored in camera frame, so they moved with the camera. The
world map must store them in world frame — transform each camera-frame measurement
through body frame (add the camera-to-body offset) then through the robot's yaw and
position into world frame, and fuse there. Then a stationary cube stays put as Cozmo
drives.
**Dev**: And the box misalignment?
**Expert**: The box yaw was derived and applied without accounting for the
world-to-scene reflection. Use Cozmo's true yaw and one shared world-to-scene
transform; the browser just renders the scene-frame coordinates the library sends.
