# Cozmo Engine Marker Data (extracted)

Source: `libcozmoEngine.so` (ARM 32-bit, armeabi-v7a) from the Cozmo Android APK.
Extracted with a small Python ELF reader (no dependencies).

## What these files are

The Cozmo engine embeds marker data in multiple forms:

1. **Marker code dictionary** (`marker_names.json`) \
   `Marker::GetNameForCode()` maps a 16-bit marker code (0..0x27) to a canonical
   name. Verified against 5 identical table copies in the binary.

2. **Probe-sampling grid** (`marker_probes.json`, `probe_centers_*.bin`) \
   Detection binarizes a detected marker quad by sampling a fixed 32x32 grid
   (GRIDSIZE=32, NUM_PROBES=1024) of probe centers. The centers are int16
   coordinates in a 15-fractional-bit space (32768 units), ranging 7149..25619
   with step 596 in both axes (regular lattice).

3. **Probe points & threshold probes** (in `marker_probes.json`) \
   - `probe_points_x/y` (5 each): offsets from each center used to average
     brightness: center, right, left, down, up (offsets 0 and +/-205).
   - `threshold_dark_x/y` and `threshold_bright_x/y` (12 each): probe locations
     used to determine bright/dark thresholds per code.
   - `NUM_FRACTIONAL_BITS = 15`, `NUM_PROBE_POINTS = 5`, `NUM_THRESHOLD_PROBES = 12`.

4. **Nearest-neighbor library** is constructed at runtime from these arrays
   (`VisionMarker::GetNearestNeighborLibrary` @ 0x8aed1c).

5. **Template bitmap images** (`template_64x512_1bit.png`) \
   Embedded at file offset 0xbe302e–0xbe402e (4096 bytes).
   Format: 64×512 1-bit bitmap (each byte = 8 pixels, MSB first).
   Contains 472 marker instances (8 per row × 59 rows) at multiple
   resolutions. Last 5 rows overlap with string data. Markers arranged
   horizontally in groups of 8 (same marker at different scales).
   `template_64x512.png` = same data as 8-bit grayscale.
   `template_raw.bin` = raw 4096 bytes.
   Individual marker extraction needs user's old script.

## Marker codes (definitive)

| Code | Name |
|------|------|
| 0  | MARKER_ARROW |
| 1  | MARKER_BULLSEYE2 |
| 2  | MARKER_CHARGER |
| 3  | MARKER_INVALID |
| 4-9  | MARKER_LIGHTCUBEI_{BACK,BOTTOM,FRONT,LEFT,RIGHT,TOP} |
| 10-15 | MARKER_LIGHTCUBEJ_{BACK,BOTTOM,FRONT,LEFT,RIGHT,TOP} |
| 16-21 | MARKER_LIGHTCUBEK_{BACK,BOTTOM,FRONT,LEFT,RIGHT,TOP} |
| 22-25 | MARKER_SDK_2{CIRCLES,DIAMONDS,HEXAGONS,TRIANGLES} |
| 26-29 | MARKER_SDK_3{...} |
| 30-33 | MARKER_SDK_4{...} |
| 34-37 | MARKER_SDK_5{...} |
| 38 | MARKER_STAR5 |
| 39 | MARKER_UNKNOWN |

Special: code 0x7fff → "MARKER_ANY" (wildcard).

Light cube markers: 3 cubes × 6 faces. Face order per cube: BACK=0/BOTTOM=1/FRONT=2/LEFT=3/RIGHT=4/TOP=5.

## Key binary locations

| Data | Offset |
|------|--------|
| Marker name strings | 0xbe402e .. 0xbe436e |
| Template bitmap | 0xbe302e .. 0xbe402e (4096 bytes, 64×512 1-bit) |
| Marker name pointer table (x5) | 0x101be80, 0x102cc74, 0x103093c, 0x1037220, 0x1037348 |
| GRIDSIZE | 0x00dc7f64 (=32) |
| NUM_PROBES | 0x00dc7f68 (=1024) |
| NUM_FRACTIONAL_BITS | 0x00dc7f6c (=15) |
| NUM_PROBE_POINTS | 0x00dc7f70 (=5) |
| NUM_THRESHOLD_PROBES | 0x00dc7f88 (=12) |
| ProbePoints_X/Y (int16[5]) | 0x00dc7f74 / 0x00dc7f7e |
| ThresholdDarkProbe_X/Y (int16[12]) | 0x00dc7f8c / 0x00dc7fa4 |
| ThresholdBrightProbe_X/Y (int16[12]) | 0x00dc7fbc / 0x00dc7fd4 |
| ProbeCenters_X (int16[1024]) | 0x00dc7fec |
| ProbeCenters_Y (int16[1024]) | 0x00dc87ec |

## Relevant engine functions

- `Marker::GetNameForCode(int16)` @ 0x88e0d4
- `VisionSystem::DetectMarkersWithCLAHE` @ 0x6c4780
- `MarkerDetector::Detect` @ 0x8853b6
- `VisionMarker::GetNearestNeighborLibrary` @ 0x8aed1c
- `VisionMarkerImages::MatchExhaustive` @ 0x8b0280
- `FiducialMarkerParser::DecodeId` @ 0x8ae568

## Files

- `extract_markers.py` - regeneration script
- `marker_names.json` - code -> name dictionary (40 entries)
- `marker_probes.json` - all constants + probe/threshold arrays
- `probe_centers_x.bin` / `probe_centers_y.bin` - raw int16 little-endian grids
- `template_raw.bin` - raw 4096 bytes (64×512 1-bit)
- `template_64x512_1bit.png` - correct rendering (1-bit)
- `template_64x512.png` - same data as 8-bit grayscale
- `template_metadata.json` - template data metadata
- `README.md` - this file
