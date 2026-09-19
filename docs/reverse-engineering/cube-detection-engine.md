# Cozmo Engine Cube Detection - Reverse Engineering

Source: `libcozmoEngine.so` (ARM 32-bit, armeabi-v7a) from Cozmo APK.
Analyzed with Binary Ninja `.bndb` + pseudo-C exports + radare2.

---

## Architecture Overview

The vision pipeline lives in `Anki::Cozmo::VisionSystem` and is driven
frame-by-frame by `Anki::Cozmo::VisionComponent`.

### Class Hierarchy (relevant)

```
Anki::Cozmo::VisionComponent          -- top-level orchestrator, dedicated thread
  +-- Anki::Cozmo::VisionSystem       -- owns detectors, modes, result lists
       +-- MarkerDetector             -- custom ArUco-like marker detection
       +-- FaceTracker                -- face detection + recognition (OpenCV dnn)
       +-- MotionDetector             -- frame-differencing motion centroid
       +-- OverheadEdgesDetector      -- edge/line detection for overhead view
       +-- RollingShutterCorrector    -- pixel-shift compensation
```

---

## VisionSystem::Update -- Main Pipeline (0x6c4d5c)

Called each frame with `(VisionPoseData, ImageCache)`. Steps:

1. **Guard**: Must be initialized (+0x58 flag) and have calibrated camera (+0x64).
   Otherwise logs "VisionSystem.Update.NotReady" and returns.

2. **Update pose data**: `UpdatePoseData(arg1)` -- copies latest robot pose.

3. **Get grayscale image**: `ImageCache::GetGray()`.

4. **Swap result lists**: Move-assigns 11 result lists from previous frame,
   then clears the VisionSystem's lists:
   - RobotObservedMotion (+0x390)
   - ObservedMarker (+0x39c)
   - TrackedFace (+0x3a8)
   - TrackedPet (+0x3b4)
   - OverheadEdgeFrame (+0x3c0)
   - UpdatedFaceID (+0x3cc)
   - ToolCodeInfo (+0x3d8)
   - CameraCalibration (+0x3e4)
   - RobotObservedLaserPoint (+0x3f0)
   - Debug images gray (+0x3fc) and RGB (+0x408)

5. **Process mode queue**: Pends VisionMode enable/disable from deque (+0xc0/+0xc4),
   applies via `EnableMode()`.

6. **Process scheduled modes**: AllVisionModesSchedule from deque (+0xe4/+0xe8).

7. **Apply CLAHE**: `ApplyCLAHE(grayImage, CLAHE mode)` -- tile size 4.

8. **Rolling shutter correction** (if enabled at +0x50):
   `RollingShutterCorrector::ComputePixelShifts()`.

9. **Compute image mean**: `ComputeMean(image, 10)` -- stored at +0x37c bit 8.

10. **Detection stages** (each gated by `ShouldProcessVisionMode()`):

    | Order | Detector                         | Result flag | Description              |
    |-------|----------------------------------|-------------|--------------------------|
    | 1     | DetectMarkersWithCLAHE()         | bit 1 (0x2) | Marker detection         |
    | 2     | DetectFaces()                    | bit 2 (0x4) | Human face detection     |
    | 3     | DetectPets()                     | bit 9 (0x200)| Pet detection           |
    | 4     | MotionDetector::Detect()         | bit 3 (0x8) | Motion centroid          |
    | 5     | OverheadEdgesDetector::Detect()  | bit 4 (0x10)| Overhead edge detection  |
    | 6     | ReadToolCode()                   | bit 5 (0x20)| Tool code reading        |
    | 7     | ComputeCalibration()             | bit 6 (0x40)| Auto-calibration         |
    | 8     | DetectLaserPoints()              | bit 15 (0x8000)| Laser detection       |
    | 9     | CheckImageQuality()              | bit 7 (0x80)| Image quality assessment |

11. **Push results**: VisionProcessingResult to deque under mutex lock.

---

## DetectMarkersWithCLAHE (0x6c4780)

Signature: `(Image, Image, vector<Rectangle<int>>, MarkerDetectionCLAHE)`

### CLAHE Modes (switch on arg4)

| Mode | Behavior                                            |
|------|-----------------------------------------------------|
| 0    | Detect on arg2 (CLAHE image)                        |
| 1    | Detect on arg3 (rectangles)                         |
| 2    | Detect on arg2 first, then arg3 if first returns 0  |
| 3, 4 | Detect on arg2 or arg3 based on flag at +0x358      |
| >4   | Return 0 (no detection)                             |

### Detection Loop

For each detected marker region:
1. Convert 2D quad corners to Rectangle<int>
2. If rolling shutter enabled (+0x50): compute pixel shifts per corner,
   apply corrections
3. Create ObservedMarker with code ID, 2D quad, camera, CLAHE mode
4. Copy quad corners and camera pose into marker result
5. Store in output list

---

## MarkerDetector::Detect (0x8853b6)

Low-level detection. Takes `(Image, list<ObservedMarker>)`.

### Pipeline:
1. Reset memory buffers: `Memory::ResetBuffers()`
2. Determine detection mode from image flag at +0x7c:
   - 0: Single mode (one detection pass)
   - 1: Alternate mode (one detection pass, different code set)
   - 2: Dual mode (two detection passes, different marker families)
3. For each pass:
   - Create ROI from image
   - Apply thresholding / edge detection
   - Find quadrilateral candidates
   - Decode marker codes
   - Create ObservedMarker for each valid detection

---

## ObservedMarker Construction (0x4cf144)

```
ObservedMarker(code_id, marker_type, quad_2d, camera, clah_mode)
```

- code_id: uint16 marker identifier
- marker_type: reference to marker family
- quad_2d: Quadrilateral<2, float> -- four 2D corner positions
- camera: camera intrinsics for pose estimation
- Pose estimated later via KnownMarker::EstimateObservedPose()

---

## KnownMarker -- Pose Estimation

- **EstimateObservedPose (0x4dfddc)**: PnP with canonical 3D corners
  (`_canonicalCorners3d`) and observed 2D corners.

- **Get3dCorners (0x4ca9a4)**: Returns 3D corner positions given pose.

- **ComputeNormal (0x4b4918)**: Surface normal from pose -- used for visibility.

- **IsVisibleFrom (0x4dfd94)**: Checks visibility from camera, considering
  field of view, distance, occlusion. Outputs `NotVisibleReason`.

---

## Marker Types (from .rodata strings)

### Cube Markers (6 faces x 3 cubes = 18 markers)

Each light cube has 6 face markers:

- MARKER_LIGHTCUBEI_BACK/BOTTOM/FRONT/LEFT/RIGHT/TOP (Cube I)
- MARKER_LIGHTCUBEJ_BACK/BOTTOM/FRONT/LEFT/RIGHT/TOP (Cube J)
- MARKER_LIGHTCUBEK_BACK/BOTTOM/FRONT/LEFT/RIGHT/TOP (Cube K)

### Robot/Charger Markers

- MARKER_ARROW
- MARKER_BULLSEYE2
- MARKER_CHARGER
- MARKER_STAR5

### SDK Custom Markers (16 total)

MARKER_SDK_{2-5}{CIRCLES,DIAMONDS,HEXAGONS,TRIANGLES}

### Special

- MARKER_UNKNOWN
- MARKER_INVALID

---

## Key Data Structures

### VisionPoseData (key offsets)

| Offset | Description                   |
|--------|-------------------------------|
| +0x1c  | VisionSystem pointer          |
| +0x378-0x38c | Pose parameters         |
| +0x390-0x408 | Result lists             |

### VisionSystem Object Layout

| Offset | Type                   | Description                    |
|--------|------------------------|--------------------------------|
| +0x40  | RollingShutterCorrector| Rolling shutter corrector      |
| +0x50  | bool                   | Rolling shutter enabled        |
| +0x54  | ImageCache*            | Image cache                    |
| +0x58  | bool                   | Initialized flag               |
| +0x5c  | CozmoContext*          | Context pointer                |
| +0x64  | CameraCalibration*     | Calibrated camera              |
| +0xa0  | bool                   | Mode change pending            |
| +0xb4-0xe8 | Deque              | Mode schedule queues           |
| +0x304 | MarkerDetector*        | Marker detector                |
| +0x30c | MotionDetector*        | Motion detector                |
| +0x310 | OverheadEdgesDetector* | Edge detector                  |
| +0x350 | int                    | CLAHE tile size (default 4)    |
| +0x358 | bool                   | CLAHE enabled flag             |
| +0x37c | int32                  | Processing result flags (bitmask) |
| +0x390 | list                   | RobotObservedMotion results    |
| +0x39c | list                   | ObservedMarker results         |
| +0x3a8 | list                   | TrackedFace results            |
| +0x3b4 | list                   | TrackedPet results             |
| +0x3c0 | list                   | OverheadEdgeFrame results      |
| +0x3cc | list                   | UpdatedFaceID results          |
| +0x3d8 | list                   | ToolCodeInfo results           |
| +0x3e4 | list                   | CameraCalibration results      |
| +0x3f0 | list                   | RobotObservedLaserPoint results|

---

## Comparison with pycozmo Implementation

### What pycozmo already implements

- VisionProcessor class with OpenCV-based marker detection
- CubeDetector using ArUco + custom dictionary
- Camera calibration retrieval from robot NVRAM
- Head tilt compensation
- Cube face template matching (JPG templates in cube_faces/)
- CubeFusionTracker for multi-face fusion

### What the engine does differently

1. **Custom marker detector** -- not standard ArUco. Uses
   `Anki::Vision::MarkerDetector` with its own code families
2. **CLAHE preprocessing** -- tile size 4 before detection
3. **Rolling shutter correction** -- compensates for body rotation
4. **Multi-mode detection** -- multiple passes with different marker families
5. **Integrated pipeline** -- markers, faces, pets, motion, edges unified
6. **Pose estimation** -- PnP with canonical 3D corners per marker type

### OpenCV dependency

Links against libopencv_objdetect.so -- used for face detection.

---

## Functions Not Fully Decomposed

These crashed Binary Ninja during export, need radare2 analysis:

- MarkerDetector::Detect -- full detection loop body
- ObservedMarker::ObservedMarker -- construction details
- KnownMarker::EstimateObservedPose -- PnP implementation
- FaceTracker::Impl::DetectFaceParts -- face parts detection
- MotionDetector::Detect -- motion detection algorithm
- OverheadEdgesDetector::Detect -- edge detection algorithm

---

## Key Addresses

| Function                           | Address  |
|------------------------------------|----------|
| VisionSystem::Update (EncodedImage)| 0x6c4b68 |
| VisionSystem::Update (ImageCache)  | 0x6c4d5c |
| VisionSystem::Init                 | 0x6c0658 |
| VisionSystem::EnableMode           | 0x6c19c0 |
| VisionSystem::ApplyCLAHE           | 0x6c44e0 |
| VisionSystem::DetectMarkersWithCLAHE| 0x6c4780 |
| VisionSystem::DetectFaces          | 0x6c3548 |
| VisionSystem::DetectPets           | 0x6c3940 |
| VisionSystem::ComputeMean          | 0x6c2cdc |
| VisionSystem::ShouldProcessVisionMode| 0x6c5aa4 |
| VisionSystem::ComputeCalibration   | 0x6c84d8 |
| MarkerDetector::Detect             | 0x8853b6 |
| MarkerDetector::Init               | 0x8852e0 |
| ObservedMarker::ObservedMarker     | 0x4cf144 |
| KnownMarker::EstimateObservedPose  | 0x4dfddc |
| KnownMarker::Get3dCorners          | 0x4ca9a4 |
| KnownMarker::ComputeNormal         | 0x4b4918 |
| KnownMarker::IsVisibleFrom         | 0x4dfd94 |
| KnownMarker::SetPose               | 0x4c8550 |
| FaceTracker::Impl::DetectFaceParts | 0x4df914 |
| Camera::Project3dPoint             | 0x4cebe0 |
| Camera::Project3dPoints            | 0x4cf27c |
| Camera::ComputePanAndTiltAngles    | 0x4bc790 |
| Camera::IsWithinFieldOfView        | 0x4ca2cc |
| CameraCalibration constructor      | 0x4b829c |
