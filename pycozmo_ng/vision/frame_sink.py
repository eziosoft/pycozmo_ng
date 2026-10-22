"""Frame sink — the display-side seam behind the ``FrameStage``.

``FrameStage`` (:mod:`pycozmo_ng.vision.frame_stage`) is a deep module: one
camera frame in, world map and browser updated. Everything it publishes to
the display crosses this one declared interface, so ``vision/`` never
imports ``web/`` and no call site reaches into a display adapter's
internals.

Two adapters satisfy the interface:

* :class:`pycozmo_ng.web.viewer_server.WebViewerServer` — production;
* the recording fake in ``tests/test_frame_stage.py`` — testing.

The interface is the test surface: the per-frame sequence is asserted by
what arrives here, not by reaching past the seam.

Avoid: importing ``WebViewerServer`` from ``vision/``; reaching through
the sink into its parts (``sink.occupancy.push`` — deleted); one setter
per floor-grid field.
"""

from typing import Protocol

__all__ = ["FrameSink"]


class FrameSink(Protocol):
    """What a display adapter must offer the frame stage."""

    def push_raw_frame(self, image) -> None:
        """Publish the raw camera frame (MJPEG stream)."""
        ...

    def push_annotated_frame(self, image) -> None:
        """Publish the annotated camera frame (MJPEG stream)."""
        ...

    def push_floor_frame(self, image) -> None:
        """Publish the floor-projected frame (MJPEG stream)."""
        ...

    def push_face_image(self, pil_image) -> None:
        """Publish the procedural face image."""
        ...

    def push_occupancy_frame(self, image, origin_x_mm: float, origin_y_mm: float) -> None:
        """Publish the occupancy grid image and its world-frame origin (mm)."""
        ...

    def set_floor_grid(self, size_mm: tuple[float, float], center_mm: tuple[float, float]) -> None:
        """Set the floor grid dimensions and center (mm) for the browser overlay."""
        ...

    def broadcast(self) -> None:
        """Mark the world state changed so the next broadcast picks it up."""
        ...
