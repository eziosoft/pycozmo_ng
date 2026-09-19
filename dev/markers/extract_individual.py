#!/usr/bin/env python3
"""Extract individual markers from 64x512 1-bit template atlas and analyze."""
import struct
import zlib
import os

DATA = "/Users/bartosz.szczygiel/Downloads/pycozmo/APK/cozmo_apk/lib/armeabi-v7a/libcozmoEngine.so"
OUT = "/Users/bartosz.szczygiel/Downloads/pycozmo/dev/markers"

f = open(DATA, "rb")
f.seek(0xbe302e)
data = f.read(4096)
f.close()

# Convert 1-bit to 2D array: 64 rows x 512 cols (1-bit, MSB first)
bits = []
for byte_val in data:
    for bit in range(8):
        bits.append((byte_val >> (7 - bit)) & 1)

# pixels[r][c] = bits[r * 512 + c]
def get_pixel(r, c):
    return bits[r * 512 + c]

# Extract 8 markers (each 64 cols x 64 rows)
markers = []
for m in range(8):
    marker_bits = []
    for r in range(64):
        for c in range(64):
            marker_bits.append(get_pixel(r, m * 64 + c))
    markers.append(marker_bits)

# Save each marker as PNG (64x64, 1-bit packed into 8x64 byte rows = 512 bytes per marker)
def save_png_1bit(filename, pixel_bits, width, height):
    """pixel_bits: list of 0/1. width in pixels, height in rows."""
    raw = bytearray()
    for r in range(height):
        row_bytes = bytearray(width // 8)
        for c in range(width):
            bit = pixel_bits[r * width + c]
            byte_idx = c // 8
            bit_idx = 7 - (c % 8)
            if bit:
                row_bytes[byte_idx] |= (1 << bit_idx)
        raw.extend(row_bytes)
    png = make_png(width, height, raw)
    with open(filename, "wb") as f:
        f.write(png)

def make_png(width, height, raw):
    def chunk(typ, data):
        c = typ + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)
    ihdr = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    raw_with_filter = bytearray()
    stride = width // 8
    for r in range(height):
        raw_with_filter.append(0)
        raw_with_filter.extend(raw[r * stride:(r + 1) * stride])
    idat = zlib.compress(bytes(raw_with_filter))
    sig = b"\x89PNG\r\n\x1a\n"
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")

# Save 8 individual markers
for m in range(8):
    fname = os.path.join(OUT, f"template_marker_{m}.png")
    save_png_1bit(fname, markers[m], 64, 64)
    print(f"Saved template_marker_{m}.png")

# Also save as PBM (simple ASCII 1-bit format) for easy inspection
for m in range(8):
    fname = os.path.join(OUT, f"template_marker_{m}.pbm")
    with open(fname, "w") as f:
        f.write(f"P1\n64 64\n")
        for r in range(64):
            row = ""
            for c in range(64):
                row += str(markers[m][r * 64 + c]) + " "
            f.write(row.strip() + "\n")
    print(f"Saved template_marker_{m}.pbm")

# Compare markers byte-by-byte (each 64-byte column per row)
print("\n=== Byte comparison (64 bytes per row, per marker) ===")
# For each pair, count how many rows have identical 64-byte data
byte_rows = []  # list of 8 lists, each with 64 rows of 64 bytes
for m in range(8):
    rows = []
    for r in range(64):
        row_data = bytes(data[r * 64 + m * 8:r * 64 + m * 8 + 8])
        rows.append(row_data)
    byte_rows.append(rows)

for m1 in range(8):
    for m2 in range(m1 + 1, 8):
        matches = sum(1 for r in range(64) if byte_rows[m1][r] == byte_rows[m2][r])
        if matches > 0:
            print(f"  Markers {m1} & {m2}: {matches}/64 rows match")

# Check which markers share the D5 7F header pattern
print("\n=== Header byte analysis (first 2 bytes per row per marker) ===")
for m in range(8):
    headers = set()
    for r in range(64):
        row = byte_rows[m][r]
        headers.add(row[:2])
    print(f"  Marker {m}: unique headers = {headers}")
