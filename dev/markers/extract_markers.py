#!/usr/bin/env python3
"""Extract Cozmo engine marker template data from libcozmoEngine.so.

The engine does NOT embed printable marker bitmaps. Markers are defined by:
  1. A code -> name dictionary (Marker::GetNameForCode table).
  2. A fixed 32x32 probe grid (GRIDSIZE=32, NUM_PROBES=1024) of int16 centers.
  3. 5 probe points per center + 12 bright/dark threshold probes, used to
     binarize a detected marker quad and classify it via nearest-neighbor.

All offsets below are raw file offsets in the ELF (verified via symbols:
_ZN4Anki8Embedded12VisionMarker...*E).
"""
import struct
import json

PATH = "/Users/bartosz.szczygiel/Downloads/pycozmo/APK/cozmo_apk/lib/armeabi-v7a/libcozmoEngine.so"
data = open(PATH, "rb").read()

# ---- 1. code -> name dictionary -----------------------------------------
# Marker::GetNameForCode(0x88e0d4) returns (&data_1048220)[code].
# The pointer table lives at file offset 0x101be80 (40 entries pointer->0xbe40xx).
DICT_TABLE = 0x101be80
names = []
for i in range(40):
    p = struct.unpack_from("<I", data, DICT_TABLE + 4 * i)[0]
    s = data[p:p + 80].split(b"\x00")[0].decode("ascii", "replace")
    names.append(s)

# ---- 2. probe geometry constants ----------------------------------------
def u32(va):
    return struct.unpack_from("<I", data, va)[0]

def i16s(va, n):
    return [struct.unpack_from("<h", data, va + 2 * i)[0] for i in range(n)]

constants = {
    "GRIDSIZE": u32(0x00dc7f64),
    "NUM_PROBES": u32(0x00dc7f68),
    "NUM_FRACTIONAL_BITS": u32(0x00dc7f6c),
    "NUM_PROBE_POINTS": u32(0x00dc7f70),
    "NUM_THRESHOLD_PROBES": u32(0x00dc7f88),
}
probe_points_x = i16s(0x00dc7f74, 5)
probe_points_y = i16s(0x00dc7f7e, 5)
threshold_dark_x = i16s(0x00dc7f8c, 12)
threshold_dark_y = i16s(0x00dc7fa4, 12)
threshold_bright_x = i16s(0x00dc7fbc, 12)
threshold_bright_y = i16s(0x00dc7fd4, 12)
probe_centers_x = i16s(0x00dc7fec, 1024)
probe_centers_y = i16s(0x00dc87ec, 1024)

# ---- written outputs ------------------------------------------------------
import os
out = "/Users/bartosz.szczygiel/Downloads/pycozmo/dev/markers"

with open(os.path.join(out, "marker_names.json"), "w") as f:
    json.dump(names, f, indent=1)

with open(os.path.join(out, "marker_probes.json"), "w") as f:
    json.dump({
        "constants": constants,
        "probe_points_x": probe_points_x,
        "probe_points_y": probe_points_y,
        "threshold_dark_x": threshold_dark_x,
        "threshold_dark_y": threshold_dark_y,
        "threshold_bright_x": threshold_bright_x,
        "threshold_bright_y": threshold_bright_y,
        "probe_centers_x": probe_centers_x,
        "probe_centers_y": probe_centers_y,
    }, f)

# raw dumps too
for name, va, n in [
    ("probe_centers_x.bin", 0x00dc7fec, 2048),
    ("probe_centers_y.bin", 0x00dc87ec, 2048),
]:
    with open(os.path.join(out, name), "wb") as f:
        f.write(data[va:va + n])

print("GRIDSIZE:", constants["GRIDSIZE"])
print("NUM_PROBES:", constants["NUM_PROBES"])
print("NUM_FRACTIONAL_BITS:", constants["NUM_FRACTIONAL_BITS"])
print("NUM_PROBE_POINTS:", constants["NUM_PROBE_POINTS"])
print("NUM_THRESHOLD_PROBES:", constants["NUM_THRESHOLD_PROBES"])
print()

# Show dictionary grouped to make it readable
print("Marker code -> name dictionary (%d entries):" % len(names))
for i, n in enumerate(names):
    print(f"  [{i:2}] {n}")