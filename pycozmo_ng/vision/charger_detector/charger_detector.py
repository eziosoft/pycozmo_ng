"""ChargerDetector — rectangular marker detection for the Cozmo charger (COZMO-65)."""

from __future__ import annotations

import cv2
import numpy as np

from pycozmo_ng.vision.cube_detector.cube_detector import CozmoCubeDetector

#: Marker width/height in mm (measured from the charger's printed marker).
CHARGER_WIDTH_MM = 23.0
CHARGER_HEIGHT_MM = 15.0

#: Width/height gate on the candidate quad's bounding box (before warping).
#: The marker's physical aspect is 23/15 = 1.53; real camera quads measured
#: 1.32-1.37, so the gate is loose enough to admit perspective distortion
#: while still rejecting squares (cube faces sit near 1.0, outside 1.1).
CHARGER_ASPECT_RANGE: tuple[float, float] = (1.1, 2.0)


class ChargerDetector(CozmoCubeDetector):
    """Detects the charger's rectangular marker (23 x 15 mm).

    A ``CozmoCubeDetector`` chassis with three differences:

    * object points are a rectangle (+-11.5 / +-7.5 mm) solved with
      ``SOLVEPNP_IPPE`` instead of a square with ``SOLVEPNP_IPPE_SQUARE``;
    * the aspect gate is tightened toward the marker's ~1.53 width/height;
    * ``reject_markers`` (the cube face templates) must not out-score the
      charger template, so a cube face is never reported as the charger.

    Like the cube detector's symmetric guard, rejection only runs on quads
    whose charger score already passed the threshold.
    """

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        target_markers,
        reject_markers=None,
        width_mm: float = CHARGER_WIDTH_MM,
        height_mm: float = CHARGER_HEIGHT_MM,
        aspect_range: tuple[float, float] = CHARGER_ASPECT_RANGE,
        max_detections=1,
        min_area=50,
        **kwargs,
    ):
        """
        :param target_markers: List of 32x32 charger template bitmaps (one).
        :param reject_markers: Cube face templates — quads they out-score
            are rejected (COZMO-65).
        :param width_mm: Marker width (default 23.0).
        :param height_mm: Marker height (default 15.0).
        :param aspect_range: Accepted width/height ratio of the quad bbox.
        """
        super().__init__(
            camera_matrix,
            dist_coeffs,
            target_markers,
            reject_markers=reject_markers,
            aspect_range=aspect_range,
            max_detections=max_detections,
            min_area=min_area,
            **kwargs,
        )
        hw, hh = width_mm / 2.0, height_mm / 2.0
        # Same corner order as the base square points: TL, TR, BR, BL
        # (x right, y up); the image points are rolled to template
        # orientation before the solve, so a marker rotated in the image
        # still maps its long axis to the long object axis.
        self.obj_points = np.array(
            [[-hw, hh, 0], [hw, hh, 0], [hw, -hh, 0], [-hw, -hh, 0]],
            dtype=np.float32,
        )
        self.pnp_flags = cv2.SOLVEPNP_IPPE

    def _get_cube_orientation(self, marker_id):
        """Charger markers carry no cube orientation; results are filtered."""
        return 0, "charger", 0

    def detect(self, frame):
        """Detect the charger marker; result dicts carry charger keys only."""
        results = super().detect(frame)
        for r in results:
            # Drop the cube-identity fields inherited from the chassis so a
            # charger result can never be mistaken for a cube measurement.
            r.pop("cube_index", None)
            r.pop("face_type", None)
            r.pop("rotation", None)
            r.pop("id", None)
            r["kind"] = "charger"
        return results


__all__ = [
    "CHARGER_ASPECT_RANGE",
    "CHARGER_HEIGHT_MM",
    "CHARGER_WIDTH_MM",
    "ChargerDetector",
]
