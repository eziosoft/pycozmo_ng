"""Robot center detection via two ring markers on the robot body.

When the head is tilted fully down (MIN_HEAD_ANGLE = -25 deg), two small
black rings with white centres are visible near the bottom of the image on
the robot's chassis. The distance between their centres is 20 mm, and the
midpoint between them is the true centre of the robot in width.

Detection inverts the image (rings become bright), thresholds to find dark
regions, then uses connected-component analysis with an area filter to
isolate the small ring spots from the wide centre cable.

The rings barely move in real life and the camera resolution never changes, so
detection only searches two small windows (``search_radius`` px) centred on the
hard-coded anchor positions at the top of this file — the ROI is fixed and
never drifts, which rejects stray spots elsewhere in the band.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pycozmo import logger

__all__ = ["RobotCenterDetector", "RobotCenterResult"]

#: Pixel centres of the left and right ring markers in a fully down-tilted view
#: at the default camera resolution. Measured on the robot (2026-09-15):
#: good detections at (82,199) and (256,198). The camera resolution never
#: changes at runtime, so the rings always stay inside a ``±search_radius``
#: window around these fixed anchors — the ROI is deliberately NOT allowed to
#: drift frame-to-frame.
ANCHOR_A_PX = (82.0, 199.0)
ANCHOR_B_PX = (256.0, 198.0)


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
    3. Adaptive threshold (robust to lighting) on the inverted band.
    4. Connected-component analysis with area < 500 to isolate small
       ring spots from the wide centre cable.
    5. Search only two small ``±search_radius`` px windows centred on the
       hard-coded anchor positions (``ANCHOR_A_PX``/``ANCHOR_B_PX``); take
       the spot nearest to each anchor. No full-band search, no anchor drift.
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
        search_radius: int = 20,
        threshold_block: int = 31,
        threshold_c: int = 5,
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
        #: How far (px) the rings are allowed to move from the anchored ROI
        #: (fixed to the module-level ANCHOR_* constants).
        self.search_radius = search_radius
        #: Adaptive-threshold window size (odd, <= search window) and offset.
        self.threshold_block = threshold_block
        self.threshold_c = threshold_c
        self._anchor_a_px = ANCHOR_A_PX
        self._anchor_b_px = ANCHOR_B_PX
        self._last_offset: float | None = None

    def _head_near_min(self) -> bool:
        return (
            abs(self.head_angle_rad - self.MIN_HEAD_ANGLE_RAD)
            <= self.head_tolerance_rad
        )

    def _component_spots(
        self, sub: np.ndarray, x_off: float, y_off: float, w: int
    ) -> list[tuple[float, float]]:
        """Find small dark spots in ``sub`` via connected components.

        ``sub`` is a crop of the inverted image band; ``x_off``/``y_off`` are the
        offsets of that crop in the full frame so returned coords are absolute.
        Returns a list of (cx, cy) centroids, filtered by area and image edges.
        """
        blurred = cv2.GaussianBlur(sub, (3, 3), 0)

        # Adaptive threshold: robust to lighting changes / gradients, unlike a
        # fixed grey value. Works on the inverted band so bright spots (rings)
        # become the foreground.
        bs = min(self.threshold_block, *blurred.shape)
        bs -= 1 - (bs % 2)  # must be odd
        bs = max(3, bs)
        dark = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            bs,
            self.threshold_c,
        )

        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark, 8)

        spots = []
        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            # Filter: small spots only, exclude image edges
            if area < 500 and cx + x_off > 20 and cx + x_off < w - 20:
                spots.append((cx + x_off, cy + y_off))
        return spots

    def _find_circle_near(
        self, band: np.ndarray, y_start: int, w: int, pred: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Find the ring spot closest to ``pred`` within ``search_radius`` px.

        ``band`` is the inverted bottom band of the full frame, so returned
        coordinates are absolute (x, y) in the full frame.
        """
        radius = self.search_radius
        x0 = int(max(0, pred[0] - radius))
        x1 = int(min(w, pred[0] + radius))
        y0 = int(max(0, pred[1] - y_start - radius))
        y1 = int(min(band.shape[0], pred[1] - y_start + radius))

        if x1 - x0 < 4 or y1 - y0 < 4:
            return None

        spots = self._component_spots(band[y0:y1, x0:x1], x0, y0 + y_start, w)
        if not spots:
            return None
        return min(spots, key=lambda p: (p[0] - pred[0]) ** 2 + (p[1] - pred[1]) ** 2)

    def _find_circles(self, gray: np.ndarray) -> list[tuple[float, float]]:
        """Find two ring markers, sorted left-to-right.

        Searches only two small ``±search_radius`` windows centred on the
        hard-coded anchor positions (``ANCHOR_A_PX``/``ANCHOR_B_PX``). The
        camera resolution never changes, so the rings stay inside these windows;
        no full-band search and no anchor drift.
        """
        h, w = gray.shape[:2]

        # Invert: black rings become bright, white centres become dark
        neg = cv2.bitwise_not(gray)

        # Bottom band where rings sit (y 77%-92%)
        y_start = int(h * 0.77)
        y_end = int(h * 0.92)
        band = neg[y_start:y_end, :]

        a = self._find_circle_near(band, y_start, w, self._anchor_a_px)
        b = self._find_circle_near(band, y_start, w, self._anchor_b_px)

        if a is None or b is None:
            return []
        return sorted([a, b], key=lambda p: p[0])

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

        # Log at INFO only on first detection or when offset changes significantly
        if (
            self._last_offset is None
            or abs(lateral_offset_mm - self._last_offset) > 0.5
        ):
            logger.info(
                f"Robot center: lateral_offset={lateral_offset_mm:+.2f}mm, "
                f"circles at ({circle_a_px[0]:.0f},{circle_a_px[1]:.0f}) and "
                f"({circle_b_px[0]:.0f},{circle_b_px[1]:.0f})"
            )
        self._last_offset = lateral_offset_mm

        return RobotCenterResult(
            detected=True,
            lateral_offset_mm=lateral_offset_mm,
            circle_a_px=circle_a_px,
            circle_b_px=circle_b_px,
            scale_mm_per_px=scale_mm_per_px,
        )
