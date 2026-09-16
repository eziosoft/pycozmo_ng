"""Simple face detector with 3D position estimation.

Uses OpenCV Haar cascade for detection and the pinhole camera model
with a known average face width to estimate depth. Returns 3D positions
in camera frame (mm) suitable for feeding into WorldMap.

Usage::

    detector = FaceDetector(camera_calibration)
    faces = detector.detect(pil_image)
    for f in faces:
        print(f"Face at {f.position_mm} (confidence={f.confidence:.2f})")

"""

from dataclasses import dataclass

import cv2
import numpy as np

from pycozmo import logger
from pycozmo.vision.camera_calibration import CameraCalibration

__all__ = ["FaceDetection", "FaceDetector"]

# Average human face width in metres (bizygomatic width).
# Varies ~14-16 cm across adults; 15.5 cm is a reasonable default.
REAL_FACE_WIDTH_M: float = 0.155


@dataclass
class FaceDetection:
    """A detected face with 2D bounding box and estimated 3D position."""

    __slots__ = ("x", "y", "width", "height", "confidence", "position_mm")

    x: int
    y: int
    width: int
    height: int
    confidence: float
    position_mm: np.ndarray  # (3,) camera frame, mm


class FaceDetector:
    """Haar-cascade face detector with pinhole 3D position estimation.

    Parameters
    ----------
    camera_calibration : CameraCalibration
        Intrinsics from the robot camera (fx, fy, cx, cy).
    real_face_width_m : float
        Assumed real-world face width in metres.  Adjust if your subjects
        are mostly children or you have per-person calibration.
    min_confidence : float
        Minimum Haar cascade confidence to report (default 1.2).
    """

    def __init__(
        self,
        camera_calibration: CameraCalibration,
        real_face_width_m: float = REAL_FACE_WIDTH_M,
        min_confidence: float = 1.2,
    ):
        self.fx = camera_calibration.fx
        self.fy = camera_calibration.fy
        self.cx = camera_calibration.cx
        self.cy = camera_calibration.cy
        self.real_width_m = real_face_width_m
        self.min_confidence = min_confidence

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

        logger.info("FaceDetector initialised (real_width=%.3f m)", real_face_width_m)

    def detect(self, image: np.ndarray) -> list[FaceDetection]:
        """Detect faces and estimate 3D camera-frame positions.

        Parameters
        ----------
        image : numpy.ndarray
            BGR image (as from ``cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)``).

        Returns
        -------
        list[FaceDetection]
            One entry per detected face, sorted by x-coordinate.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces_rect = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces_rect) == 0:
            return []

        results: list[FaceDetection] = []
        for x, y, w, h in faces_rect:
            # Centre pixel
            u = x + w / 2.0
            v = y + h / 2.0

            # Depth from apparent width (pinhole)
            Z = (self.fx * self.real_width_m) / max(w, 1)

            # Back-project to 3D
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            position = np.array([X, Y, Z], dtype=np.float64)

            results.append(
                FaceDetection(
                    x=int(x),
                    y=int(y),
                    width=int(w),
                    height=int(h),
                    confidence=1.0,  # Haar cascade doesn't give a 0-1 score
                    position_mm=position,
                )
            )

        results.sort(key=lambda f: f.x)
        return results
