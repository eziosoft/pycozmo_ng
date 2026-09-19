#!/usr/bin/env python3
"""Compact visualization of extracted markers."""
DATA = "/Users/bartosz.szczygiel/Downloads/pycozmo/APK/cozmo_apk/lib/armeabi-v7a/libcozmoEngine.so"
f = open(DATA, "rb")
f.seek(0xbe302e)
data = f.read(4096)
f.close()

# Convert to 1-bit pixels: 64 rows x 512 cols
pixels = []
for byte_val in data:
    for bit in range(8):
        pixels.append((byte_val >> (7 - bit)) & 1)

# Print each marker (64x64) as 8x8 grid (each cell = 8x8 pixels)
for m in range(8):
    print(f"\n=== Marker {m} (64x64, 8x8 downsampled) ===")
    for r8 in range(8):
        row = ""
        for c8 in range(8):
            on = 0
            for r in range(r8 * 8, (r8 + 1) * 8):
                for c in range(c8 * 8, (c8 + 1) * 8):
                    on += pixels[r * 512 + m * 64 + c]
            row += "█" if on > 32 else "▓" if on > 16 else "░" if on > 0 else " "
        print(row)

# Also print first 16 rows of marker 0 as compact representation
m = 0
print("\n\n=== Marker 0, first 16 rows (each # = 2 pixels) ===")
for r in range(16):
    row = ""
    for c in range(0, 64, 2):
        p1 = pixels[r * 512 + m * 64 + c]
        p2 = pixels[r * 512 + m * 64 + c + 1] if c + 1 < 64 else 0
        row += "█" if (p1 or p2) else " "
    print(row)
