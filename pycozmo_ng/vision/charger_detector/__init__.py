"""Charger marker detection — the rectangular dock marker (23 x 15 mm).

The charger's printed marker previously matched the cube template family and
flapped cube IDs every second; it now has its own detector, template, events
and world-map slot (COZMO-65).
"""

from pycozmo_ng.vision.charger_detector.charger_detection import ChargerDetection
from pycozmo_ng.vision.charger_detector.charger_detector import (
    CHARGER_ASPECT_RANGE,
    CHARGER_HEIGHT_MM,
    CHARGER_WIDTH_MM,
    ChargerDetector,
)
from pycozmo_ng.vision.charger_detector.charger_template import (
    TEMPLATE_CROP_FRACTION,
    load_charger_template,
)

__all__ = [
    "CHARGER_ASPECT_RANGE",
    "CHARGER_HEIGHT_MM",
    "CHARGER_WIDTH_MM",
    "TEMPLATE_CROP_FRACTION",
    "ChargerDetection",
    "ChargerDetector",
    "load_charger_template",
]
