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

import os
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from pycozmo.logger import logger
from pycozmo.vision.calibration.camera_calibration import CameraCalibration

__all__ = ["FaceDetection", "FaceDetector"]

# Average human face width in metres (bizygomatic width).
# Varies ~14-16 cm across adults; 15.5 cm is a reasonable default.
REAL_FACE_WIDTH_M: float = 0.155

# Haar cascade download URL (OpenCV official repo).
_HAARCASCADE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
_HAARCASCADE_NAME = "haarcascade_frontalface_default.xml"


def _ensure_cascade() -> str:
    """Return the path to the frontal-face Haar cascade XML.

    If the file is missing from ``cv2.data.haarcascades`` (common with
    opencv-contrib-python 5.x on Python 3.14), download it automatically.
    """
    data_dir = cv2.data.haarcascades  # type: ignore[attr-defined]
    path = os.path.join(data_dir, _HAARCASCADE_NAME)

    if os.path.isfile(path):
        return path

    logger.info("Haar cascade XML not found — downloading to %s", path)
    try:
        urllib.request.urlretrieve(_HAARCASCADE_URL, path)
        logger.info("Downloaded %s (%d bytes)", path, os.path.getsize(path))
    except Exception as exc:
        raise RuntimeError(
            f"Cannot download Haar cascade to {path}: {exc}. "
            "Install opencv-contrib-python or place the XML manually."
        ) from exc

    return path


@dataclass
class FaceDetection:
    """A detected face with 2D bounding box and estimated 3D position.

    ``face_id`` is assigned per frame by ``FaceIdentity`` (presence
    epoch, 1-based) before annotation and event dispatch read it; it is
    ``None`` before that assignment.
    """

    x: int
    y: int
    width: int
    height: int
    confidence: float
    position_mm: np.ndarray  # (3,) camera frame, mm
    face_id: int | None = None


class FaceDetector:
    """Haar-cascade face detector with pinhole 3D position estimation.

    Parameters
    ----------
    camera_calibration : CameraCalibration
        Intrinsics from the robot camera (fx, fy, cx, cy).
    real_face_width_m : float
        Assumed real-world face width in metres.  Adjust if your subjects
        are mostly children or you have per-person calibration.
    """

    def __init__(
        self,
        camera_calibration: CameraCalibration,
        real_face_width_m: float = REAL_FACE_WIDTH_M,
    ):
        self.fx = camera_calibration.fx
        self.fy = camera_calibration.fy
        self.cx = camera_calibration.cx
        self.cy = camera_calibration.cy
        self.real_width_m = real_face_width_m

        cascade_path = _ensure_cascade()
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

        # Pick the largest detection (closest face).
        x, y, w, h = max(faces_rect, key=lambda r: r[2] * r[3])

        # Centre pixel
        u = x + w / 2.0
        v = y + h / 2.0

        # Depth from apparent width (pinhole).  real_width_m is in metres,
        # so Z comes out in metres — convert to mm for the rest of the stack.
        Z_m = (self.fx * self.real_width_m) / max(w, 1)
        Z = Z_m * 1000.0

        # Back-project to 3D (mm)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        position = np.array([X, Y, Z], dtype=np.float64)

        return [
            FaceDetection(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                confidence=1.0,
                position_mm=position,
            )
        ]
