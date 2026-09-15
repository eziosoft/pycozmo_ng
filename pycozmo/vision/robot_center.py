"""Robot center detection via two black circles on the robot body.

When the head is tilted fully down (MIN_HEAD_ANGLE = -25 deg), two black
circles are visible in the lower part of the image. The distance between
their centres is 20 mm, and the midpoint between them is the true centre
of the robot in width.

The lateral offset (camera position relative to robot centre) is computed
from the pixel midpoint of the two circles, scaled by the known 20 mm
separation divided by the pixel distance between them.
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
    """Detect two black circles on the robot body and compute camera lateral offset.

    The detector only produces valid results when the head is tilted to
    approximately ``MIN_HEAD_ANGLE`` (-25 deg), because only then are the
    circles visible in the lower part of the camera image.

    Args:
        camera_matrix: 3x3 intrinsic matrix K from camera calibration.
        dist_coeffs: Distortion coefficients (typically zeros for Cozmo).
        head_angle_rad: Current head tilt in radians.
        circle_spacing_mm: Known distance between circle centres (default 20).
        head_tolerance_rad: How close head angle must be to MIN_HEAD_ANGLE.
    """

    # Head angle must be near MIN_HEAD_ANGLE (-25 deg = -0.436 rad) to see circles.
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
        """Check if the head is tilted close enough to MIN_HEAD_ANGLE."""
        return (
            abs(self.head_angle_rad - self.MIN_HEAD_ANGLE_RAD)
            <= self.head_tolerance_rad
        )

    def _find_circles(self, gray: np.ndarray) -> list[tuple[float, float]]:
        """Find two symmetric dark circles in the lower part of the image.

        Uses a fixed threshold (the circles are consistently dark on a lighter
        floor), then filters contours by area, circularity, and radius.  From
        the candidate circles, picks the pair that is most symmetric about the
        image centre line.

        Returns a list of (cx, cy) pixel coordinates, sorted left-to-right.
        """
        h, w = gray.shape[:2]
        img_center_x = w / 2.0

        # Only look at the lower 50% (circles are on the floor/body underside)
        y_start = int(h * 0.5)
        roi = gray[y_start:, :]

        blurred = cv2.GaussianBlur(roi, (7, 7), 0)

        # Fixed threshold: circles are dark (< 60) on lighter floor
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours to circle-like blobs of reasonable size
        candidates: list[tuple[float, float, float]] = []  # (cx, cy, score)
        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Circles on the real robot have area ~100-500 px
            if area < 80 or area > 800:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            # Real circles have circularity ~0.7+
            if circularity < 0.4:
                continue

            (_cx, _cy), _radius = cv2.minEnclosingCircle(cnt)

            # Filter by radius (real circles are ~8-15 px radius)
            if _radius < 5 or _radius > 25:
                continue

            # Adjust to full-image coordinates
            cy_full = _cy + y_start

            # Score: prefer circles closer to the vertical centre and
            # roughly the right distance from image centre
            candidates.append((_cx, cy_full, circularity))

        if len(candidates) < 2:
            return []

        # From all candidate pairs, find the two that are most symmetric
        # about the image centre line (best pair = smallest |mid_x - center_x|)
        best_pair = None
        best_symmetry = float("inf")
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                cx_a, cy_a, _ = candidates[i]
                cx_b, cy_b, _ = candidates[j]
                mid_x = (cx_a + cx_b) / 2.0
                symmetry = abs(mid_x - img_center_x)
                # Also prefer pairs at similar Y
                y_diff = abs(cy_a - cy_b)
                score = symmetry + y_diff * 0.5
                if score < best_symmetry:
                    best_symmetry = score
                    # Sort left-to-right
                    if cx_a <= cx_b:
                        best_pair = ((cx_a, cy_a), (cx_b, cy_b))
                    else:
                        best_pair = ((cx_b, cy_b), (cx_a, cy_a))

        if best_pair is None:
            return []

        return [best_pair[0], best_pair[1]]

    def detect(self, frame: np.ndarray) -> RobotCenterResult:
        """Detect robot center from camera frame.

        Args:
            frame: BGR image from camera (numpy array).

        Returns:
            RobotCenterResult with detection status and lateral offset.
        """
        if not self._head_near_min():
            return RobotCenterResult(detected=False)

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        circles = self._find_circles(gray)

        if len(circles) < 2:
            logger.debug("Robot center: fewer than 2 circles detected")
            return RobotCenterResult(detected=False)

        circle_a_px = circles[0]  # left circle (x, y)
        circle_b_px = circles[1]  # right circle (x, y)

        # Compute pixel distance and scale
        px_dist = np.hypot(
            circle_b_px[0] - circle_a_px[0],
            circle_b_px[1] - circle_a_px[1],
        )
        if px_dist < 1e-3:
            logger.warning("Robot center: circles too close in pixels")
            return RobotCenterResult(detected=False)

        scale_mm_per_px = self.circle_spacing_mm / px_dist

        # Lateral offset: midpoint pixel vs image centre
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
