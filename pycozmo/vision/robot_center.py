"""Robot center detection via two ring-shaped markers on the robot body.

When the head is tilted fully down (MIN_HEAD_ANGLE = -25 deg), two small
black rings with white centres are visible near the bottom of the image on
the robot's chassis. The distance between their centres is 20 mm, and the
midpoint between them is the true centre of the robot in width.

Detection uses a narrow horizontal band at the bottom of the frame and
Hough circle detection on the inverted image (rings become bright circles).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pycozmo import logger

__all__ = ["RobotCenterDetector", "RobotCenterResult"]


@dataclass
class RobotCenterResult:
    """Result of robot center detection."""

    detected: bool

    #: Lateral offset of camera from robot centre (mm, + = camera left of centre).
    lateral_offset_mm: float = 0.0

    #: Pixel coordinates of the two detected circle centres.
    circle_a_px: tuple[float, float] | None = None
    circle_b_px: tuple[float, float] | None = None

    #: Pixel-to-mm scale derived from the known 20 mm separation.
    scale_mm_per_px: float = 0.0


class RobotCenterDetector:
    """Detect two ring markers on the robot body and compute camera lateral offset.

    The detector only produces valid results when the head is tilted to
    approximately ``MIN_HEAD_ANGLE`` (-25 deg), because only then are the
    rings visible near the bottom of the camera image.

    The rings are small black circles with white centres, located on the
    robot's chassis at the very bottom of the frame.  Detection scans a
    narrow band (y 70%-93% of image height) and uses Hough circle detection
    on the inverted image where rings appear as bright circles.
    """

    MIN_HEAD_ANGLE_RAD = np.radians(-25.0)
    DEFAULT_HEAD_TOLERANCE_RAD = np.radians(5.0)

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        head_angle_rad: float = 0.0,
        circle_spacing_mm: float = 20.0,
        head_tolerance_rad: float | None = None,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.head_angle_rad = head_angle_rad
        self.circle_spacing_mm = circle_spacing_mm
        self.head_tolerance_rad = (
            head_tolerance_rad
            if head_tolerance_rad is not None
            else self.DEFAULT_HEAD_TOLERANCE_RAD
        )

    def _head_near_min(self) -> bool:
        return (
            abs(self.head_angle_rad - self.MIN_HEAD_ANGLE_RAD)
            <= self.head_tolerance_rad
        )

    def _find_circles(self, gray: np.ndarray) -> list[tuple[float, float]]:
        """Find two ring markers using Hough detection on inverted band.

        The rings sit in a narrow horizontal band near the bottom of the
        frame.  Inverting the image turns the black rings into bright
        circles that HoughCircles can detect reliably.

        Returns a list of (cx, cy) pixel coordinates, sorted left-to-right.
        """
        h, w = gray.shape[:2]
        img_center_x = w / 2.0

        # Narrow band at the bottom where rings sit (y 70%-93%)
        y_start = int(h * 0.70)
        y_end = int(h * 0.93)
        band = gray[y_start:y_end, :]

        # Invert: black rings become bright circles
        inv_band = cv2.bitwise_not(band)
        blurred = cv2.GaussianBlur(inv_band, (5, 5), 0)

        # Hough detection with conservative parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=8,
            minRadius=3,
            maxRadius=10,
        )

        if circles is None or len(circles[0]) < 2:
            return []

        # Sort by x coordinate (left-to-right)
        detected = []
        for cx, cy, r in circles[0]:
            detected.append((cx, cy + y_start))

        detected.sort(key=lambda p: p[0])

        # Take the two most symmetric candidates about image centre
        if len(detected) == 2:
            return detected

        # More than 2 — pick the most symmetric pair
        best_pair = None
        best_score = float("inf")
        for i in range(len(detected)):
            for j in range(i + 1, len(detected)):
                cx_a, cy_a = detected[i]
                cx_b, cy_b = detected[j]
                mid_x = (cx_a + cx_b) / 2.0
                symmetry = abs(mid_x - img_center_x)
                y_diff = abs(cy_a - cy_b)
                score = symmetry + y_diff * 0.5
                if score < best_score:
                    best_score = score
                    best_pair = (detected[i], detected[j])

        return list(best_pair) if best_pair else []

    def detect(self, frame: np.ndarray) -> RobotCenterResult:
        """Detect robot center from camera frame.

        Args:
            frame: BGR image from camera (numpy array).

        Returns:
            RobotCenterResult with detection status and lateral offset.
        """
        if not self._head_near_min():
            return RobotCenterResult(detected=False)

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        circles = self._find_circles(gray)

        if len(circles) < 2:
            logger.debug("Robot center: fewer than 2 circles detected")
            return RobotCenterResult(detected=False)

        circle_a_px = circles[0]
        circle_b_px = circles[1]

        px_dist = np.hypot(
            circle_b_px[0] - circle_a_px[0],
            circle_b_px[1] - circle_a_px[1],
        )
        if px_dist < 1e-3:
            logger.warning("Robot center: circles too close in pixels")
            return RobotCenterResult(detected=False)

        scale_mm_per_px = self.circle_spacing_mm / px_dist

        mid_x = (circle_a_px[0] + circle_b_px[0]) / 2.0
        img_center_x = gray.shape[1] / 2.0
        lateral_offset_px = mid_x - img_center_x
        lateral_offset_mm = lateral_offset_px * scale_mm_per_px

        logger.info(
            f"Robot center detected: lateral_offset={lateral_offset_mm:+.2f}mm, "
            f"circles at ({circle_a_px[0]:.0f},{circle_a_px[1]:.0f}) and "
            f"({circle_b_px[0]:.0f},{circle_b_px[1]:.0f}), "
            f"scale={scale_mm_per_px:.3f} mm/px"
        )

        return RobotCenterResult(
            detected=True,
            lateral_offset_mm=lateral_offset_mm,
            circle_a_px=circle_a_px,
            circle_b_px=circle_b_px,
            scale_mm_per_px=scale_mm_per_px,
        )
