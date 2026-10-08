"""Charger marker template loading (COZMO-65).

The charger marker is a printed battery drawing in a black rounded rectangle
(23 x 15 mm). Its photo lives in ``marker/charger.png`` and is preprocessed
with the same protocol as the cube face templates so the match scores of the
two families are directly comparable — that comparability is what lets each
detector reject the other's marker (see ``reject_markers``).
"""

import os

import cv2
import numpy as np

#: Fraction of each image dimension cropped off every side before resizing.
#: Mirrors the cube loader's fixed 18px crop of its 102px sources (17.6%);
#: the charger source (92x70) has no uniform pixel margin, so the crop is
#: fractional. Validated against real camera frames.
TEMPLATE_CROP_FRACTION = 0.18


def load_charger_template(path: str | None = None) -> np.ndarray:
    """Load the charger marker photo as a 32x32 matching template.

    Preprocessing mirrors ``load_marker_templates()``: crop the page margin,
    resize the marker body to 26x26, pad with a 3px white border. The result
    matches what ``CozmoCubeDetector._extract_patch()`` produces for a
    frontal view of the marker (a quad warped to 32x32), so
    ``cv2.matchTemplate`` scores are on the same scale as the cube scores.

    Args:
        path: Template image path. Defaults to ``marker/charger.png``.

    Returns:
        32x32 uint8 grayscale template.

    Raises:
        FileNotFoundError: The template image is missing or unreadable.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "marker", "charger.png")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Charger marker template not found: {path}")

    h, w = img.shape
    ox, oy = int(w * TEMPLATE_CROP_FRACTION), int(h * TEMPLATE_CROP_FRACTION)
    img = img[oy : h - oy, ox : w - ox]
    img = cv2.resize(img, (26, 26), interpolation=cv2.INTER_AREA)
    return cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=255)


__all__ = ["TEMPLATE_CROP_FRACTION", "load_charger_template"]
