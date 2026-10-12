"""Distance bands — coloured ground-range bands drawn in perspective on the camera frame.

Draws concentric coloured bands on the floor plane (body Z = 0), centred on
the body-frame origin, and projects them through the camera model the same way
``FloorProjector`` builds its bird's-eye lookup table: body ground point →
camera frame (head-tilt rotation about the camera-to-body offset) → pixels.

Each band marks a ground distance from the robot — green at 250 mm (near),
yellow at 500 mm (mid), red at 1 000 mm (far) — a colour cue that is far
easier for a vision model to read on a 320×240 frame than thin lines. A thin
outline at the band centreline carries the exact distance label.

Drawn *under* the detection overlays: ``FrameStage`` calls :meth:`draw` on
the frame before ``AnnotationEngine.detect_and_annotate`` copies it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from PIL import Image, ImageDraw

from pycozmo_ng.util.coords import CAMERA_TO_BODY_OFFSET, _head_tilt_matrix
from pycozmo_ng.vision.annotation import load_font

__all__ = [
    "DEFAULT_BAND_HALF_WIDTH_MM",
    "DEFAULT_OUTLINE_WIDTH_PX",
    "DEFAULT_RING_COLORS",
    "DEFAULT_RING_RADII_MM",
    "DistanceRings",
]

#: Default band radii in mm — three geometric steps (250 / 500 / 1000)
#: spanning the robot's useful workspace without crowding a 320×240 frame.
DEFAULT_RING_RADII_MM: tuple[int, ...] = (250, 500, 1000)

#: Distinct, high-contrast colours per band — green (near), yellow (mid),
#: red (far).  Extra bands cycle back through the list.
DEFAULT_RING_COLORS: tuple[tuple[int, int, int], ...] = (
    (0, 220, 0),  # green — 250 mm (near, step-back zone)
    (255, 220, 0),  # yellow — 500 mm (mid-range)
    (255, 80, 0),  # red — 1000 mm (far)
)

#: Half-width of each coloured filled band (mm).  Total ≈ 40 mm.
DEFAULT_BAND_HALF_WIDTH_MM: float = 20.0

#: Outline width at the band centreline (px).
DEFAULT_OUTLINE_WIDTH_PX: int = 2

#: Horizontal label offsets (px) from the principal column, cycled per band.
#: The 250 mm and 1 000 mm bands land at very different image rows so they
#: share the centred slot without overlap; the 500 mm band shifts left to
#: avoid crowding the others.
_LABEL_COLUMN_OFFSETS: tuple[float, ...] = (0.0, -100.0)


class DistanceRings:
    """Draws perspective-correct coloured ground distance bands on a camera frame.

    Stateless per draw: intrinsics, radii, colours, and band geometry are
    fixed at construction; the head angle arrives per call because it moves.

    Args:
        camera_matrix: 3×3 intrinsic matrix K from CameraCalibration.
        radii_mm: Band centres in mm, measured from the body-frame ground origin.
        colors: One RGB colour per radius; extra radii cycle through the list.
        band_half_width_mm: Half-width of each filled band in mm.
        outline_width: Pixel width of the centreline outline drawn on top.
        samples: Points sampled per band (one full turn).  Sized for smooth
            arcs; only the visible slice lands inside the frame.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        radii_mm: tuple[int, ...] = DEFAULT_RING_RADII_MM,
        colors: tuple[tuple[int, int, int], ...] = DEFAULT_RING_COLORS,
        band_half_width_mm: float = DEFAULT_BAND_HALF_WIDTH_MM,
        outline_width: int = DEFAULT_OUTLINE_WIDTH_PX,
        samples: int = 240,
    ):
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self.radii_mm = tuple(float(r) for r in radii_mm if r > 0)
        self._colors = colors
        self._band_half_width_mm = float(band_half_width_mm)
        self._outline_width = max(1, int(outline_width))
        self.samples = max(8, int(samples))

        # Cached intrinsics
        self._cx = float(self._K[0, 2])
        self._cy = float(self._K[1, 2])

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
        """``(N, 2)`` body-frame ground samples of one ring, first ≅ last."""
        theta = np.linspace(0.0, 2.0 * np.pi, self.samples)
        return radius_mm * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, image: Image.Image, head_angle_rad: float) -> None:
        """Draw all coloured bands, outlines, and labels on ``image`` (in place).

        Args:
            image: PIL Image to annotate (mutated in place).
            head_angle_rad: Current head tilt (rad, positive = looking up).
        """
        if not self.radii_mm:
            return

        width, height = image.size
        draw = ImageDraw.Draw(image)

        # 1. Filled coloured bands (underneath)
        for index, radius in enumerate(self.radii_mm):
            color = self._colors[index % len(self._colors)]
            self._draw_band(draw, radius, color, head_angle_rad, width, height)

        # 2. Thin outlines at each band's centreline (on top of the fill)
        for radius in self.radii_mm:
            self._draw_outline(draw, radius, head_angle_rad, width, height)

        # 3. Distance labels
        self._draw_labels(draw, head_angle_rad, width, height)

    def _draw_band(
        self,
        draw: ImageDraw.ImageDraw,
        radius_mm: float,
        color: tuple[int, int, int],
        head_angle_rad: float,
        width: int,
        height: int,
    ) -> None:
        """Fill a coloured annular band around *radius_mm* on the ground.

        The band spans ``radius_mm ± band_half_width_mm``.  Points closer
        to the robot project *larger* in the image, so the inner ground
        radius becomes the outer image boundary and vice-versa.
        """
        hw = self._band_half_width_mm
        near_r = max(0.01, radius_mm - hw)  # closer to robot → larger in image
        far_r = radius_mm + hw  # farther → smaller in image

        near_xy = self._ring_points(near_r)
        far_xy = self._ring_points(far_r)

        near_uv, near_valid = self.project_ground_points(near_xy, head_angle_rad)
        far_uv, far_valid = self.project_ground_points(far_xy, head_angle_rad)

        ok = (
            near_valid
            & far_valid
            & np.isfinite(near_uv[:, 0])
            & np.isfinite(near_uv[:, 1])
            & np.isfinite(far_uv[:, 0])
            & np.isfinite(far_uv[:, 1])
        )

        if not ok.any():
            return

        for indices in _contiguous_runs(ok):
            if len(indices) < 2:
                continue
            self._fill_band_arc(draw, near_uv, far_uv, indices, color)

    def _fill_band_arc(
        self,
        draw: ImageDraw.ImageDraw,
        near_uv: np.ndarray,
        far_uv: np.ndarray,
        indices: list[int],
        color: tuple[int, int, int],
    ) -> None:
        """Fill one contiguous visible arc of the band as a polygon.

        The polygon traces the near (outer-in-image) edge forward, then the
        far (inner-in-image) edge backward.  PIL's polygon fill handles
        arbitrary winding and clips to the image bounds automatically.
        """
        poly: list[tuple[float, float]] = []
        for i in indices:
            poly.append((float(near_uv[i, 0]), float(near_uv[i, 1])))
        for i in reversed(indices):
            poly.append((float(far_uv[i, 0]), float(far_uv[i, 1])))
        draw.polygon(poly, fill=color)

    def _draw_outline(
        self,
        draw: ImageDraw.ImageDraw,
        radius_mm: float,
        head_angle_rad: float,
        width: int,
        height: int,
    ) -> None:
        """Draw a thin white centreline at *radius_mm* for precise distance reading."""
        uv, valid = self.project_ground_points(self._ring_points(radius_mm), head_angle_rad)

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
        """Draw one unbroken stretch of ring as clipped line segments."""
        for p0, p1 in pairwise(run):
            clipped = _clip_segment(p0, p1, width, height)
            if clipped is not None:
                draw.line(
                    [clipped[0], clipped[1]],
                    fill=(255, 255, 255),
                    width=self._outline_width,
                )

    def _draw_labels(self, draw: ImageDraw.ImageDraw, head_angle_rad: float, width: int, height: int) -> None:
        """Label each band where it crosses one of the staggered columns.

        Labels sit on the band's centreline arc near the target column.  A
        label that would overlap one already placed is dropped — the band
        colour and outline are the ground truth, a buried label is worse
        than none.
        """
        font = load_font(11)
        placed: list[tuple[float, float, float, float]] = []

        for index, radius in enumerate(self.radii_mm):
            uv, valid = self.project_ground_points(self._ring_points(radius), head_angle_rad)
            visible = valid & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            if not visible.any():
                continue

            target_u = self._cx + _LABEL_COLUMN_OFFSETS[index % len(_LABEL_COLUMN_OFFSETS)]
            nearest = int(np.argmin(np.abs(np.where(visible, uv[:, 0], np.inf) - target_u)))
            u, v = uv[nearest]

            color = self._colors[index % len(self._colors)]
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
            draw.text((x, y), label, fill=color, font=font)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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
    """Liang-Barsky clip of a segment to ``[0, width) × [0, height)``.

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


def _contiguous_runs(mask: np.ndarray) -> list[list[int]]:
    """Return lists of consecutive ``True`` indices in *mask*.

    Handles the ring's closed-loop topology: a visible arc straddling the
    0 / 2π boundary produces two linear runs that are joined into one.
    Each returned run has at least 2 indices so the polygon has area.
    """
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        return []

    # Find linear runs via sentinel-padded transitions.
    padded = np.concatenate([[False], mask, [False]])
    transitions = np.diff(padded.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    runs: list[list[int]] = [list(range(s, e)) for s, e in zip(starts, ends)]

    # Merge wraparound: if both edges are True, first and last runs are one arc.
    if mask[0] and mask[-1] and len(runs) > 1:
        merged = runs[-1] + runs[0]
        runs = [merged] + runs[1:-1]

    return [r for r in runs if len(r) >= 2]
