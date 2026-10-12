"""Distance rings — ground range circles drawn in perspective on the camera frame.

Draws concentric circles on the floor plane (body Z = 0), centred on the
body-frame origin, and projects them through the camera model the same way
``FloorProjector`` builds its bird's-eye lookup table: body floor point →
camera frame (head-tilt rotation about the camera-to-body offset) → pixels.
The ring therefore lands where it actually lies on the ground, so anything
touching a ring is that far from the robot's ground origin — a distance cue
for a human viewer and for an LLM reading the annotated frame.

Drawn *under* the detection overlays: ``FrameStage`` calls :meth:`draw` on
the frame before ``AnnotationEngine.detect_and_annotate`` copies it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from PIL import Image, ImageDraw

from pycozmo_ng.util.coords import CAMERA_TO_BODY_OFFSET, _head_tilt_matrix
from pycozmo_ng.vision.annotation import load_font

__all__ = ["DEFAULT_RING_RADII_MM", "DistanceRings"]

#: Default ring radii in mm (250 mm steps out to 1 m) — cube-scale near,
#: stance-scale far, without crowding a 320x240 frame.
DEFAULT_RING_RADII_MM: tuple[int, ...] = (250, 500, 750, 1000)

#: Horizontal label offsets (px) from the principal column, cycled per ring.
#: The far rings bunch up within a few pixels of the horizon, so stacking
#: every label on its apex would bury the middle ones.
_LABEL_COLUMN_OFFSETS: tuple[float, ...] = (0.0, -80.0, 80.0)


class DistanceRings:
    """Draws perspective-correct ground distance rings on a camera frame.

    Stateless per draw: everything it needs (intrinsics, radii) is fixed at
    construction; the head angle arrives per call because it moves.

    Args:
        camera_matrix: 3x3 intrinsic matrix K from CameraCalibration.
        radii_mm: Ring radii in mm, centred on the body-frame origin.
        color: Ring and label colour as an RGB tuple.
        thickness: Ring line width in pixels.
        samples: Points sampled per ring (one full turn). The near ring
            only puts a handful of samples inside the frame at this
            default, so it is sized for smooth arcs at the edges.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        radii_mm: tuple[int, ...] = DEFAULT_RING_RADII_MM,
        color: tuple[int, int, int] = (255, 255, 255),
        thickness: int = 1,
        samples: int = 240,
    ):
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self.radii_mm = tuple(float(r) for r in radii_mm if r > 0)
        self.color = color
        self.thickness = max(1, int(thickness))
        self.samples = max(8, int(samples))

    # ------------------------------------------------------------------
    # Projection (body-frame ground point → pixel)
    # ------------------------------------------------------------------

    def project_ground_points(self, xy_mm: np.ndarray, head_angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
        """Project body-frame ground points (N, 2) into pixel coordinates.

        Returns ``(uv, valid)``: ``uv`` is an (N, 2) array of ``(u, v)``
        pixels (NaN where invalid) and ``valid`` marks the points in front
        of the camera (``cam_z > 1`` mm) — the same rule the floor
        projector's LUT uses.
        """
        pts = np.empty((len(xy_mm), 3), dtype=np.float64)
        pts[:, :2] = np.asarray(xy_mm, dtype=np.float64)
        pts[:, 2] = 0.0  # ground plane

        # Camera → body is R (head tilt); body → camera is R^T, applied to
        # row vectors as ``d @ R``.
        d = pts - CAMERA_TO_BODY_OFFSET
        cam = d @ _head_tilt_matrix(head_angle_rad)

        z = cam[:, 2]
        valid = z > 1.0
        safe_z = np.where(valid, z, 1.0)
        u = self._K[0, 0] * cam[:, 0] / safe_z + self._K[0, 2]
        v = self._K[1, 1] * cam[:, 1] / safe_z + self._K[1, 2]

        uv = np.stack([u, v], axis=1)
        uv[~valid] = np.nan
        return uv, valid

    def _ring_points(self, radius_mm: float) -> np.ndarray:
        """(N, 2) body-frame ground samples of one ring, first = last."""
        theta = np.linspace(0.0, 2.0 * np.pi, self.samples)
        return radius_mm * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, image: Image.Image, head_angle_rad: float) -> None:
        """Draw all rings and their labels on ``image`` (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            head_angle_rad: Current head tilt (rad, positive = looking up).
        """
        if not self.radii_mm:
            return

        width, height = image.size
        draw = ImageDraw.Draw(image)

        for radius in self.radii_mm:
            uv, valid = self.project_ground_points(self._ring_points(radius), head_angle_rad)
            self._draw_ring(draw, uv, valid, width, height)

        self._draw_labels(draw, head_angle_rad, width, height)

    def _draw_ring(self, draw: ImageDraw.ImageDraw, uv: np.ndarray, valid: np.ndarray, width: int, height: int) -> None:
        """Draw one ring as clipped line segments, breaking at invalid points.

        Each visible stretch of the ring is clipped against the frame
        (Liang-Barsky) so off-screen geometry never lands a stray chord
        across the image; a point behind the camera breaks the polyline —
        interpolation across it would swing through infinity.
        """
        run: list[tuple[float, float]] = []
        for (u, v), ok in zip(uv, valid):
            if not ok or not np.isfinite(u) or not np.isfinite(v):
                self._flush_run(draw, run, width, height)
                run = []
                continue
            run.append((float(u), float(v)))
        self._flush_run(draw, run, width, height)

    def _flush_run(
        self,
        draw: ImageDraw.ImageDraw,
        run: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> None:
        """Draw one unbroken stretch of ring as clipped segments."""
        for p0, p1 in pairwise(run):
            clipped = _clip_segment(p0, p1, width, height)
            if clipped is not None:
                draw.line([clipped[0], clipped[1]], fill=self.color, width=self.thickness)

    def _draw_labels(self, draw: ImageDraw.ImageDraw, head_angle_rad: float, width: int, height: int) -> None:
        """Label each ring where it crosses one of the staggered columns.

        The far rings bunch up within a few pixels of the horizon, so
        stacking every label on its apex buries the middle ones; the column
        offset cycles per ring instead. A label that would overlap one
        already placed is dropped — the ring line is the ground truth, a
        buried label is worse than none.
        """
        font = load_font(11)
        placed: list[tuple[float, float, float, float]] = []

        for index, radius in enumerate(self.radii_mm):
            uv, valid = self.project_ground_points(self._ring_points(radius), head_angle_rad)
            visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            if not visible.any():
                continue

            target_u = self._K[0, 2] + _LABEL_COLUMN_OFFSETS[index % len(_LABEL_COLUMN_OFFSETS)]
            nearest = int(np.argmin(np.abs(np.where(visible, uv[:, 0], np.inf) - target_u)))
            u, v = uv[nearest]

            label = f"{radius:g} mm"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = min(max(float(u) - text_w / 2.0, 2.0), max(2.0, width - text_w - 2.0))
            y = min(max(float(v) + 4.0, 2.0), max(2.0, height - text_h - 4.0))
            rect = (x - 2, y - 2, x + text_w + 2, y + text_h + 2)

            if any(_rects_overlap(rect, other) for other in placed):
                continue
            placed.append(rect)

            draw.rectangle([(rect[0], rect[1]), (rect[2], rect[3])], fill=(0, 0, 0))
            draw.text((x, y), label, fill=self.color, font=font)


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """True when two ``(x0, y0, x1, y1)`` rectangles share any pixel."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _clip_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: int,
    height: int,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky clip of a segment to ``[0, width) x [0, height)``.

    Returns the clipped endpoints, or ``None`` when the segment misses the
    frame entirely.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, (width - 1) - x0),
        (-dy, y0),
        (dy, (height - 1) - y0),
    ):
        if p == 0.0:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)

    return (
        (x0 + t0 * dx, y0 + t0 * dy),
        (x0 + t1 * dx, y0 + t1 * dy),
    )
