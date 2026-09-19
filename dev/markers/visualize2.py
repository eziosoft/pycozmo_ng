#!/usr/bin/env python3
"""Print markers as ASCII art (1 pixel = 1 char)."""
DATA = "/Users/bartosz.szczygiel/Downloads/pycozmo/APK/cozmo_apk/lib/armeabi-v7a/libcozmoEngine.so"
f = open(DATA, "rb")
f.seek(0xbe302e)
data = f.read(4096)
f.close()

pixels = []
for byte_val in data:
    for bit in range(8):
        pixels.append((byte_val >> (7 - bit)) & 1)

# Print all 8 markers side by side (8*64=512 cols compressed to 128 chars)
print("=== All 8 markers, rows 0-31 (each # = 4 pixels) ===\n")
for r in range(32):
    row = ""
    for m in range(8):
        for c in range(0, 64, 4):
            on = sum(pixels[r * 512 + m * 64 + c + k] for k in range(4))
            row += "█" if on >= 3 else "▓" if on >= 2 else "░" if on >= 1 else " "
    print(f"R{r:02d} {row}")

print("\n=== Marker 0 as 64x64 ASCII (each # = 1 pixel) ===")
for r in range(64):
    row = ""
    for c in range(64):
        row += "#" if pixels[r * 512 + c] else "."
    print(row)
