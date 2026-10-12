"""Distance zones — translucent ground-distance zones drawn on the camera frame.

Zones are concentric regions on the floor plane (body Z = 0), centred on the
body-frame ground origin.  By default there are two: the image bottom up to
the 250 mm boundary is tinted a strong orange, and the 250 → 500 mm zone a
lighter orange of the same hue.  The tint is alpha-blended with the original
frame — never opaque — so the floor stays visible underneath.  Zones read far
better on a 320×240 frame than thin lines, which collapse to a few pixels at
the far end.

Each pixel is classified by inverting the camera model (pixel → ray → ground
plane hit → radial distance from the body origin), so the zones are exact in
perspective and follow the head tilt.  Boundary distances are labelled down
the left edge of the frame, inside their zone.

Drawn *under* the detection overlays: ``FrameStage`` calls :meth:`draw` on
the frame before ``AnnotationEngine.detect_and_annotate`` copies it.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from pycozmo_ng.util.coords import CAMERA_TO_BODY_OFFSET, _head_tilt_matrix
from pycozmo_ng.vision.annotation import load_font

__all__ = [
    "DEFAULT_ZONE_ALPHAS",
    "DEFAULT_ZONE_COLORS",
    "DEFAULT_ZONE_RADII_MM",
    "DistanceRings",
]

#: Default zone boundary radii in mm, measured from the body-frame ground
#: origin.  Two zones: bottom of image → 250 mm (near), 250 → 500 mm (mid).
DEFAULT_ZONE_RADII_MM: tuple[int, ...] = (250, 500)

#: Orange tint shared by both zones — the hue carries the "distance cue"
#: role, the strength varies with alpha.  Extra zones cycle through.
DEFAULT_ZONE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 130, 0),  # near zone — bottom of image → 250 mm
    (255, 130, 0),  # mid zone — 250 → 500 mm (lighter, same hue)
)

#: Per-zone alpha (0–255) over the original frame.  The near zone is the
#: stronger tint; extras cycle through the list.
DEFAULT_ZONE_ALPHAS: tuple[int, ...] = (120, 55)

#: Horizontal inset of the zone labels from the left frame edge (px).
_LABEL_INSET_PX = 4.0


class DistanceRings:
    """Draws translucent ground-distance zones on a camera frame.

    Stateless per draw: intrinsics, boundaries, colours, and alphas are
    fixed at construction; the head angle arrives per call because it moves.

    Args:
        camera_matrix: 3×3 intrinsic matrix K from CameraCalibration.
        radii_mm: Zone boundary radii in mm from the body-frame ground
            origin.  Zone *i* spans ``radii[i-1] → radii[i]``; zone 0
            starts at 0 (the bottom of the image).
        colors: One RGB tint per zone; extra boundaries cycle through.
        alphas: One alpha (0–255) per zone, blended over the original frame.
        samples: Points sampled per boundary ring for label placement.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        radii_mm: tuple[int, ...] = DEFAULT_ZONE_RADII_MM,
        colors: tuple[tuple[int, int, int], ...] = DEFAULT_ZONE_COLORS,
        alphas: tuple[int, ...] = DEFAULT_ZONE_ALPHAS,
        samples: int = 240,
    ):
        self._K = np.asarray(camera_matrix, dtype=np.float64)
        self.radii_mm = tuple(float(r) for r in radii_mm if r > 0)
        self._colors = colors
        self._alphas = tuple(min(255, max(0, int(a))) for a in alphas)
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
        of the camera (``cam_z > 1`` mm).
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
        """``(N, 2)`` body-frame ground samples of one boundary ring."""
        theta = np.linspace(0.0, 2.0 * np.pi, self.samples)
        return radius_mm * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def _ground_distance_map(self, size: tuple[int, int], head_angle_rad: float) -> np.ndarray:
        """Per-pixel radial ground distance (mm); NaN where the ray misses the floor.

        Inverts the camera model: pixel → camera-frame ray → body-frame ray
        (head-tilt rotation) → ground-plane hit (body Z = 0) → distance from
        the body-frame ground origin.  Rays at or above the horizon have no
        ground hit and stay NaN, which keeps them outside every zone.
        """
        width, height = size
        u = np.arange(width, dtype=np.float64)
        v = np.arange(height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        x = (uu - self._cx) / self._K[0, 0]
        y = (vv - self._cy) / self._K[1, 1]
        rays = np.stack([x, y, np.ones_like(x)], axis=-1)

        # Camera → body on direction vectors (translation does not apply;
        # R is orthonormal, so its inverse is its transpose).
        body = rays @ _head_tilt_matrix(head_angle_rad).T
        dz = body[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dz < 0.0, CAMERA_TO_BODY_OFFSET[2] / -dz, np.nan)
        gx = CAMERA_TO_BODY_OFFSET[0] + t * body[..., 0]
        gy = CAMERA_TO_BODY_OFFSET[1] + t * body[..., 1]
        return np.hypot(gx, gy)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, image: Image.Image, head_angle_rad: float) -> None:
        """Tint the ground-distance zones onto ``image`` (in place), then label them.

        Each zone is alpha-blended with the original pixels — the floor
        stays visible underneath, and no pixel is ever fully replaced.

        Args:
            image: PIL Image to annotate (mutated in place).
            head_angle_rad: Current head tilt (rad, positive = looking up).
        """
        if not self.radii_mm:
            return

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return

        distance = self._ground_distance_map(image.size, head_angle_rad)
        rgb = arr[..., :3].astype(np.float32)
        inner = 0.0
        for index, outer in enumerate(self.radii_mm):
            color = np.asarray(self._colors[index % len(self._colors)], dtype=np.float32)
            alpha = self._alphas[index % len(self._alphas)] / 255.0
            zone = np.isfinite(distance) & (distance > inner) & (distance <= outer)
            if zone.any():
                rgb[zone] = rgb[zone] * (1.0 - alpha) + color * alpha
            inner = outer

        blended = np.clip(rgb, 0.0, 255.0).round().astype(np.uint8)
        has_alpha = arr.shape[-1] == 4
        if has_alpha:
            blended = np.concatenate([blended, arr[..., 3:4]], axis=-1)
        image.paste(Image.fromarray(blended))

        # Boundary labels — after the tint so they stay fully legible.
        self._draw_labels(ImageDraw.Draw(image), head_angle_rad, *image.size)

    def _draw_labels(
        self,
        draw: ImageDraw.ImageDraw,
        head_angle_rad: float,
        width: int,
        height: int,
    ) -> None:
        """Label each zone boundary down the left edge of the frame.

        A label sits at the row where its boundary ring crosses the left
        column.  A label that would overlap one already placed is dropped —
        a buried label is worse than none.
        """
        font = load_font(11)
        placed: list[tuple[float, float, float, float]] = []

        for index, radius in enumerate(self.radii_mm):
            label = f"{radius:g} mm"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            x = _LABEL_INSET_PX
            target_u = x + text_w / 2.0

            uv, valid = self.project_ground_points(self._ring_points(radius), head_angle_rad)
            visible = valid & np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
            if not visible.any():
                continue
            nearest = int(np.argmin(np.abs(np.where(visible, uv[:, 0], np.inf) - target_u)))
            v = float(uv[nearest, 1])

            y = min(max(v - text_h / 2.0, 2.0), max(2.0, height - text_h - 4.0))
            rect = (x - 2, y - 2, x + text_w + 2, y + text_h + 2)

            if any(_rects_overlap(rect, other) for other in placed):
                continue
            placed.append(rect)

            draw.rectangle([(rect[0], rect[1]), (rect[2], rect[3])], fill=(0, 0, 0))
            draw.text((x, y), label, fill=self._colors[index % len(self._colors)], font=font)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _rects_overlap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    """True when two ``(x0, y0, x1, y1)`` rectangles share any pixel."""
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
