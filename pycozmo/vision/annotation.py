"""Annotation engine — pure drawing functions for vision detections.

This module provides stateless annotation functions that draw detection
overlays on PIL images. Pure functions enable easy testing and reuse
in headless contexts.
"""

from PIL import Image, ImageDraw, ImageFont

from pycozmo.vision.detection_pipeline import CubeDetection
from pycozmo.vision.face_detector import FaceDetection
from pycozmo.vision.robot_center import RobotCenterResult


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
        if size in self._fonts:
            return self._fonts[size]
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", size)
            except OSError:
                font = ImageFont.load_default()
        self._fonts[size] = font
        return font

    def draw_detections(
        self, image: Image.Image, detections: list[CubeDetection]
    ) -> None:
        """
        Draw detection bounding boxes and labels on an image (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            detections: List of cube detections to draw
        """
        draw = ImageDraw.Draw(image)

        font = self._get_font(12)

        for i, det in enumerate(detections):
            if det.cube_id is not None:
                color = self.annotation_colors[
                    det.cube_id % len(self.annotation_colors)
                ]
            else:
                color = self.annotation_colors[i % len(self.annotation_colors)]

            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.width, det.y + det.height

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=color,
                width=max(1, self.box_thickness),
            )

            label_parts = []
            if self.show_cube_id and det.cube_id is not None:
                label_parts.append(f"ID:{det.cube_id}")
            if self.show_confidence:
                label_parts.append(f"{det.confidence:.2f}")
            if self.show_distance and det.distance is not None:
                label_parts.append(f"{det.distance:.1f}m")

            if label_parts:
                label = " | ".join(label_parts)

                try:
                    bbox = draw.textbbox((x1, y1 - 15), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except AttributeError:
                    text_width, text_height = draw.textsize(label, font=font)
                    bbox = (x1, y1 - 15, x1 + text_width, y1 - 15 + text_height)

                draw.rectangle(
                    [(bbox[0] - 2, bbox[1] - 2), (bbox[2] + 2, bbox[3] + 2)],
                    fill=(0, 0, 0, 180),
                )

                draw.text((x1, y1 - 15), label, fill=color, font=font)

    def draw_face_detections(
        self, image: Image.Image, detections: list[FaceDetection]
    ) -> None:
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

            face_id = getattr(det, "_face_id", "?")
            z_mm = det.position_mm[2]
            label = f"Face {face_id} | {z_mm:.0f}mm"
            draw.text((x1, y1 - 13), label, fill=color, font=font)

    def draw_robot_center(
        self, image: Image.Image, result: RobotCenterResult | None = None
    ) -> None:
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
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2
            )

        if result.circle_b_px is not None:
            cx, cy = result.circle_b_px
            r = 6
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)], outline=(0, 255, 255), width=2
            )

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

    def detect_and_annotate(
        self,
        image: Image.Image,
        cube_detections: list[CubeDetection],
        face_detections: list[FaceDetection],
        robot_center_result: RobotCenterResult | None = None,
    ) -> Image.Image:
        """
        Apply all annotations to an image.

        Args:
            image: Base PIL Image
            cube_detections: Cube detections to draw
            face_detections: Face detections to draw
            robot_center_result: Optional robot center result to draw

        Returns:
            Fully annotated PIL Image (single copy of the original)
        """
        annotated = image.copy()
        self.draw_detections(annotated, cube_detections)
        self.draw_face_detections(annotated, face_detections)
        if robot_center_result is not None:
            self.draw_robot_center(annotated, robot_center_result)
        return annotated
