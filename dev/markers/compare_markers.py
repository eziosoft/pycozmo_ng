#!/usr/bin/env python3
"""Compare extracted 64x64 markers with existing cube_faces JPGs."""
import os
import cv2
import numpy as np

MARKER_DIR = "/Users/bartosz.szczygiel/Downloads/pycozmo/dev/markers"
FACES_DIR = "/Users/bartosz.szczygiel/Downloads/pycozmo/pycozmo/vision/cube_detector/cube_faces"

# Load extracted markers (1-bit PNGs we saved)
extracted = []
for m in range(8):
    path = os.path.join(MARKER_DIR, f"template_marker_{m}.png")
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load {path}")
        continue
    # PNG saved as 1-bit grayscale, convert to binary
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    binary = (gray > 127).astype(np.uint8) * 255
    resized = cv2.resize(binary, (32, 32), interpolation=cv2.INTER_AREA)
    extracted.append(resized)
    print(f"Marker {m}: {img.shape} -> {resized.shape}")

# Load cube faces JPGs
faces = {}
for fname in sorted(os.listdir(FACES_DIR)):
    path = os.path.join(FACES_DIR, fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load {path}")
        continue
    # Resize to 32x32 for comparison
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    binary = (resized > 127).astype(np.uint8) * 255
    faces[fname] = binary
    print(f"Face {fname}: {img.shape} -> {binary.shape}")

# Compare each extracted marker with each cube face using template matching
print("\n=== Template matching scores (TM_CCOEFF_NORMED) ===")
for m, marker in enumerate(extracted):
    best = []
    for fname, face in faces.items():
        res = cv2.matchTemplate(marker, face, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        best.append((score, fname))
    best.sort(reverse=True)
    print(f"Marker {m}: best match = {best[0][1]} (score={best[0][0]:.4f}), "
          f"2nd = {best[1][1]} (score={best[1][0]:.4f})")
