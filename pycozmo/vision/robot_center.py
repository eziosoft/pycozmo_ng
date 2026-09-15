"""Robot center detection via two black circles on the robot body.

When the head is tilted fully down (MIN_HEAD_ANGLE = -25 deg), two black
circles are visible in the lower part of the image. The distance between
their centres is 20 mm, and the midpoint between them is the true centre
of the robot in width.

By detecting both circles and using their known 3D positions on the robot
body, we solve for the exact camera-to-body extrinsics (position + rotation)
via ``cv2.solvePnP``.  This replaces the hardcoded
``CAMERA_TO_BODY_OFFSET`` assumption with a measured calibration.
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

    #: Camera position in body frame (mm). Result of solvePnP tvec.
    translation_mm: np.ndarray | None = None

    #: Camera orientation in body frame (3x3 rotation). Result of solvePnP rvec.
    rotation_matrix: np.ndarray | None = None

    #: Lateral offset of camera from robot centre (mm, + = camera left of centre).
    lateral_offset_mm: float = 0.0

    #: Pixel coordinates of the two detected circle centres.
    circle_a_px: tuple[float, float] | None = None
    circle_b_px: tuple[float, float] | None = None

    #: Pixel-to-mm scale derived from the known 20 mm separation.
    scale_mm_per_px: float = 0.0


class RobotCenterDetector:
    """Detect two black circles on the robot body and solve for camera extrinsics.

    The detector only produces valid results when the head is tilted to
    approximately ``MIN_HEAD_ANGLE`` (-25 deg), because only then are the
    circles visible in the lower part of the camera image.

    Args:
        camera_matrix: 3x3 intrinsic matrix K from camera calibration.
        dist_coeffs: Distortion coefficients (typically zeros for Cozmo).
        head_angle_rad: Current head tilt in radians.
        circle_spacing_mm: Known distance between circle centres (default 20).
        circle_a_body: 3D position of circle A in body frame (mm).
        circle_b_body: 3D position of circle B in body frame (mm).
        head_tolerance_rad: How close head angle must be to MIN_HEAD_ANGLE.
        min_circle_area_px: Minimum contour area to consider as a circle.
        max_circle_area_px: Maximum contour area to consider as a circle.
        min_circularity: Minimum circularity score (0-1) for contour filter.
    """

    # Default circle positions in body frame (mm).
    # Body frame: X forward, Y left, Z up. Origin at front-center ground under camera.
    # These are approximate — the circles sit on the underside of the robot,
    # near the front edge, symmetric about the centerline.
    DEFAULT_CIRCLE_A_BODY = np.array([10.0, -10.0, 5.0], dtype=np.float64)
    DEFAULT_CIRCLE_B_BODY = np.array([10.0, 10.0, 5.0], dtype=np.float64)

    # Head angle must be near MIN_HEAD_ANGLE (-25 deg = -0.436 rad) to see circles.
    MIN_HEAD_ANGLE_RAD = np.radians(-25.0)
    DEFAULT_HEAD_TOLERANCE_RAD = np.radians(5.0)

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        head_angle_rad: float = 0.0,
        circle_spacing_mm: float = 20.0,
        circle_a_body: np.ndarray | None = None,
        circle_b_body: np.ndarray | None = None,
        head_tolerance_rad: float | None = None,
        min_circle_area_px: float = 50.0,
        max_circle_area_px: float = 5000.0,
        min_circularity: float = 0.5,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.head_angle_rad = head_angle_rad
        self.circle_spacing_mm = circle_spacing_mm
        self.circle_a_body = (
            circle_a_body.copy()
            if circle_a_body is not None
            else self.DEFAULT_CIRCLE_A_BODY.copy()
        )
        self.circle_b_body = (
            circle_b_body.copy()
            if circle_b_body is not None
            else self.DEFAULT_CIRCLE_B_BODY.copy()
        )
        self.head_tolerance_rad = (
            head_tolerance_rad
            if head_tolerance_rad is not None
            else self.DEFAULT_HEAD_TOLERANCE_RAD
        )
        self.min_circle_area_px = min_circle_area_px
        self.max_circle_area_px = max_circle_area_px
        self.min_circularity = min_circularity

    def _head_near_min(self) -> bool:
        """Check if the head is tilted close enough to MIN_HEAD_ANGLE."""
        return (
            abs(self.head_angle_rad - self.MIN_HEAD_ANGLE_RAD)
            <= self.head_tolerance_rad
        )

    def _find_circles(self, gray: np.ndarray) -> list[tuple[float, float]]:
        """Find dark circular blobs in the lower half of the image.

        Returns a list of (cx, cy) pixel coordinates, sorted left-to-right.
        """
        h, w = gray.shape[:2]

        # Only look at the lower 60 % of the image (circles are on the floor/body)
        y_start = int(h * 0.4)
        roi = gray[y_start:, :]

        # Adaptive threshold for dark objects on lighter background
        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Also try a fixed threshold as fallback (dark circles, lighter floor)
        _, thresh_fixed = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Combine both approaches
        combined = cv2.bitwise_or(thresh, thresh_fixed)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[tuple[float, float, float]] = []  # (cx, cy, area)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_circle_area_px or area > self.max_circle_area_px:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-3:
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Fit minimum enclosing circle for centre
            (_cx, _cy), _radius = cv2.minEnclosingCircle(cnt)
            # Adjust back to full-image coordinates
            cy_full = _cy + y_start
            candidates.append((_cx, cy_full, area))

        if len(candidates) < 2:
            return []

        # Sort by area descending, take top 2
        candidates.sort(key=lambda c: c[2], reverse=True)
        top2 = candidates[:2]

        # Sort left-to-right for consistent ordering
        top2.sort(key=lambda c: c[0])

        return [(c[0], c[1]) for c in top2]

    def _find_circles_hough(self, gray: np.ndarray) -> list[tuple[float, float]]:
        """Alternative: Hough circle detection. Used as fallback."""
        h, w = gray.shape[:2]
        y_start = int(h * 0.4)
        roi = gray[y_start:, :]

        blurred = cv2.GaussianBlur(roi, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=80,
            param2=30,
            minRadius=5,
            maxRadius=50,
        )

        if circles is None:
            return []

        results = []
        for cx, cy, _r in circles[0]:
            results.append((cx, cy + y_start))

        if len(results) < 2:
            return []

        # Take the two largest-ish circles (Hough doesn't return area, so take first 2)
        results.sort(key=lambda c: c[0])  # left-to-right
        return results[:2]

    def detect(self, frame: np.ndarray) -> RobotCenterResult:
        """Detect robot center from camera frame.

        Args:
            frame: BGR image from camera (numpy array).

        Returns:
            RobotCenterResult with detection status and extrinsics.
        """
        if not self._head_near_min():
            return RobotCenterResult(detected=False)

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Try contour-based detection first
        circles = self._find_circles(gray)

        # Fallback to Hough if contour detection failed
        if len(circles) < 2:
            circles = self._find_circles_hough(gray)

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

        # Solve for camera-to-body extrinsics using solvePnP
        # Object points: circle centres in body frame
        obj_points = np.array(
            [self.circle_a_body, self.circle_b_body], dtype=np.float64
        )
        # Image points: detected circle centres
        img_points = np.array([circle_a_px, circle_b_px], dtype=np.float64).reshape(
            -1, 1, 2
        )

        translation_mm = None
        rotation_matrix = None

        try:
            # Use solvePnP with the two points (coplanar, so IPPE won't work;
            # use iterative solver with initial guess from offset estimation)
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )

            if success and rvec is not None and tvec is not None:
                translation_mm = tvec.flatten().copy()
                rotation_mat, _ = cv2.Rodrigues(rvec)
                rotation_matrix = rotation_mat.copy()

                # Override lateral offset with the more accurate solvePnP result
                lateral_offset_mm = float(translation_mm[1])

                logger.info(
                    f"Robot center: solvePnP offset = "
                    f"({translation_mm[0]:.1f}, {translation_mm[1]:.1f}, "
                    f"{translation_mm[2]:.1f}) mm"
                )
        except cv2.error as e:
            logger.warning(f"Robot center: solvePnP failed: {e}")

        logger.info(
            f"Robot center detected: lateral_offset={lateral_offset_mm:.2f}mm, "
            f"circles at ({circle_a_px[0]:.0f},{circle_a_px[1]:.0f}) and "
            f"({circle_b_px[0]:.0f},{circle_b_px[1]:.0f}), "
            f"scale={scale_mm_per_px:.3f} mm/px"
        )

        return RobotCenterResult(
            detected=True,
            translation_mm=translation_mm,
            rotation_matrix=rotation_matrix,
            lateral_offset_mm=lateral_offset_mm,
            circle_a_px=circle_a_px,
            circle_b_px=circle_b_px,
            scale_mm_per_px=scale_mm_per_px,
        )
