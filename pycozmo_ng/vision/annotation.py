"""Annotation engine — pure drawing functions for vision detections.

This module provides stateless annotation functions that draw detection
overlays on PIL images. Pure functions enable easy testing and reuse
in headless contexts.
"""

from PIL import Image, ImageDraw, ImageFont

from pycozmo_ng.cubes.identity import to_cube_id
from pycozmo_ng.vision.aruco_detector.aruco_detection import ArucoDetection
from pycozmo_ng.vision.calibration.robot_center import RobotCenterResult
from pycozmo_ng.vision.charger_detector.charger_detection import ChargerDetection
from pycozmo_ng.vision.cube_detector.detection_pipeline import CubeDetection
from pycozmo_ng.vision.face_detector import FaceDetection

#: Font cache shared by every drawing module (detections, distance rings),
#: so font files are read once per size per process.
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def load_font(size: int = 12) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font with platform fallbacks (cached module-wide)."""
    font = _FONT_CACHE.get(size)
    if font is None:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except OSError:
                font = ImageFont.load_default()
        _FONT_CACHE[size] = font
    return font


class AnnotationEngine:
    """Pure drawing functions for vision detections.

    No state, no side effects — just draw on PIL images.
    """

    def __init__(
        self,
        annotation_colors: list[tuple[int, int, int]] | None = None,
        box_thickness: int = 2,
        show_cube_id: bool = True,
        show_confidence: bool = True,
        show_distance: bool = True,
    ):
        """
        Args:
            annotation_colors: List of RGB colors for different cube IDs.
            box_thickness: Bounding box line thickness in pixels.
            show_cube_id: Whether to show cube ID in labels.
            show_confidence: Whether to show confidence score in labels.
            show_distance: Whether to show distance in labels.
        """
        self.annotation_colors = annotation_colors or [(0, 255, 255)]
        self.box_thickness = box_thickness
        self.show_cube_id = show_cube_id
        self.show_confidence = show_confidence
        self.show_distance = show_distance

        # Cache fonts (avoid file I/O per detection per frame)
        self._fonts: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    def _get_font(self, size: int = 12):
        """Load a font with fallbacks (cached)."""
        if size not in self._fonts:
            self._fonts[size] = load_font(size)
        return self._fonts[size]

    def draw_detections(self, image: Image.Image, detections: list[CubeDetection]) -> None:
        """
        Draw detection bounding boxes and labels on an image (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            detections: List of cube detections to draw
        """
        draw = ImageDraw.Draw(image)

        font = self._get_font(12)

        for det in detections:
            color = self.annotation_colors[det.cube_index % len(self.annotation_colors)]

            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=max(1, self.box_thickness),
            )

            label_parts = []
            if self.show_cube_id:
                label_parts.append(f"ID:{to_cube_id(det.cube_index)}")
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if self.show_distance and det.distance is not None:
                label_parts.append(f"{det.distance:.1f}m")

            if label_parts:
                label = " | ".join(label_parts)

                bbox = draw.textbbox((x1, y1 - 15), label, font=font)

                draw.rectangle(
                    [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)],
                    fill=(0, 0, 0, 180),
                )

                draw.text((x1, y1 - 15), label, fill=color, font=font)

    def draw_face_detections(self, image: Image.Image, detections: list[FaceDetection]) -> None:
        """Draw face bounding boxes on the image (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            detections: List of face detections to draw.
        """
        if not detections:
            return

        draw = ImageDraw.Draw(image)

        font = self._get_font(11)

        color = (255, 160, 0)

        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            face_id = det.face_id
            z_mm = det.position_mm[2]
            label = f"Face {face_id} | {z_mm:.0f}mm"
            draw.text((x1, y1 - 13), label, fill=color, font=font)

    def draw_robot_center(self, image: Image.Image, result: RobotCenterResult | None = None) -> None:
        """Draw robot center detection overlay on the image (in place).

        Shows the two detected circles, the midpoint (robot center), and
        the lateral offset value.

        Args:
            image: PIL Image to annotate (mutated in place).
            result: RobotCenterResult to draw. If None, returns without changes.
        """
        if result is None or not result.detected:
            return

        draw = ImageDraw.Draw(image)

        font = self._get_font(11)

        if result.circle_a_px is not None:
            cx, cy = result.circle_a_px
            r = 6
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2)

        if result.circle_b_px is not None:
            cx, cy = result.circle_b_px
            r = 6
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2)

        if result.circle_a_px is not None and result.circle_b_px is not None:
            mid_x = (result.circle_a_px[0] + result.circle_b_px[0]) / 2.0
            mid_y = (result.circle_a_px[1] + result.circle_b_px[1]) / 2.0

            img_h = image.size[1]

            draw.line([(mid_x, 0), (mid_x, img_h)], fill=(0, 255, 255), width=1)

            draw.ellipse(
                [(mid_x - 4, mid_y - 4), (mid_x + 4, mid_y + 4)],
                fill=(0, 255, 255),
            )

        label = f"center: {result.lateral_offset_mm:+.1f}mm"
        draw.text((5, 5), label, fill=(0, 255, 255), font=font)

    def draw_charger_detections(self, image: Image.Image, detections: list[ChargerDetection]) -> None:
        """Draw the charger marker bounding box on the image (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            detections: List of charger detections to draw (0 or 1).
        """
        if not detections:
            return

        draw = ImageDraw.Draw(image)

        font = self._get_font(11)

        # Green: distinct from cube cyan and face orange
        color = (0, 255, 64)

        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

            label = f"Charger | {det.confidence:.2f} | {det.distance:.2f}m"
            draw.text((x1, y1 - 13), label, fill=color, font=font)

    def draw_aruco_detections(self, image: Image.Image, detections: list[ArucoDetection]) -> None:
        """Draw ArUco marker quads and labels on the image (in place).

        Draws the actual corner quad (not just the bbox) — ArUco gives
        exact marker corners — labelled with ID, configured size, distance.

        Args:
            image: PIL Image to annotate (mutated in place).
            detections: List of ArUco detections to draw (0..n).
        """
        if not detections:
            return

        draw = ImageDraw.Draw(image)

        font = self._get_font(11)

        # Magenta: distinct from cube cyan, face orange, charger green
        color = (255, 0, 255)

        for det in detections:
            if det.corners is not None and len(det.corners) == 4:
                pts = [tuple(p) for p in det.corners]
                draw.polygon(pts, outline=color)
                x1, y1 = min(p[0] for p in pts), min(p[1] for p in pts)
            else:
                x1, y1 = det.x, det.y
                draw.rectangle([(det.x, det.y), (det.x + det.width, det.y + det.height)], outline=color, width=2)

            label = f"ArUco {det.marker_id} | {det.size_mm:g}mm | {det.distance:.2f}m"
            draw.text((x1, y1 - 13), label, fill=color, font=font)

    def detect_and_annotate(
        self,
        image: Image.Image,
        cube_detections: list[CubeDetection],
        face_detections: list[FaceDetection],
        robot_center_result: RobotCenterResult | None = None,
        charger_detections: list[ChargerDetection] | None = None,
        aruco_detections: list[ArucoDetection] | None = None,
    ) -> Image.Image:
        """
        Apply all annotations to an image.

        Args:
            image: Base PIL Image
            cube_detections: Cube detections to draw
            face_detections: Face detections to draw
            robot_center_result: Optional robot center result to draw
            charger_detections: Optional charger marker detections to draw
            aruco_detections: Optional ArUco marker detections to draw

        Returns:
            Fully annotated PIL Image (single copy of the original)
        """
        annotated = image.copy()
        self.draw_detections(annotated, cube_detections)
        self.draw_face_detections(annotated, face_detections)
        if robot_center_result is not None:
            self.draw_robot_center(annotated, robot_center_result)
        if charger_detections:
            self.draw_charger_detections(annotated, charger_detections)
        if aruco_detections:
            self.draw_aruco_detections(annotated, aruco_detections)
        return annotated
