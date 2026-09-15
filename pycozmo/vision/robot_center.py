"""Robot center detection via two ring markers on the robot body.

When the head is tilted fully down (MIN_HEAD_ANGLE = -25 deg), two small
black rings with white centres are visible near the bottom of the image on
the robot's chassis. The distance between their centres is 20 mm, and the
midpoint between them is the true centre of the robot in width.

Detection inverts the image (rings become bright), thresholds to find dark
regions, then uses connected-component analysis with an area filter to
isolate the small ring spots from the wide centre cable.
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

    Algorithm:
    1. Invert the image (black rings become bright spots).
    2. Take the bottom band (y 77%-92% of image height).
    3. Threshold at 60 to find dark regions on the inverted image.
    4. Connected-component analysis with area < 500 to isolate small
       ring spots from the wide centre cable.
    5. Take the two smallest components (left and right ring).
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
        """Find two ring markers using connected-component analysis on inverted image.

        Returns a list of (cx, cy) pixel coordinates, sorted left-to-right.
        """
        h, w = gray.shape[:2]

        # Invert: black rings become bright, white centres become dark
        neg = cv2.bitwise_not(gray)

        # Bottom band where rings sit (y 77%-92%)
        y_start = int(h * 0.77)
        y_end = int(h * 0.92)
        band = neg[y_start:y_end, :]

        blurred = cv2.GaussianBlur(band, (3, 3), 0)

        # Threshold: dark regions on the inverted image
        _, dark = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Connected components — filter by area to keep only small spots
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)

        spots = []
        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 500:  # rings are small; cable is large
                cx, cy = centroids[i]
                spots.append((cx, cy + y_start))

        if len(spots) < 2:
            return []

        # Sort left-to-right and take the two smallest (most ring-like)
        spots.sort(key=lambda p: p[0])

        # If more than 2, pick the pair with most symmetric midpoint
        if len(spots) == 2:
            return spots

        img_center_x = w / 2.0
        best_pair = None
        best_score = float("inf")
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                mid_x = (spots[i][0] + spots[j][0]) / 2.0
                score = abs(mid_x - img_center_x)
                if score < best_score:
                    best_score = score
                    best_pair = (spots[i], spots[j])

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
