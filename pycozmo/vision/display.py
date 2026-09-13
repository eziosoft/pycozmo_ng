"""Vision display — frame handlers, overlay, FPS, save, debug images."""

import time

import cv2
import numpy as np


class VisionDisplay:
    """Handles display of normal and annotated frames + web viewer streaming.

    References shared state on ``self.brain`` (the :class:`Brain` core).
    """

    def __init__(self, brain):
        self.brain = brain

    def on_raw_camera_image(self, cli, image):
        """Handler for raw camera images."""
        self.brain.latest_raw_frame = image
        self.brain.frame_count += 1
        self.brain._last_frame_time = time.time()

        if image is not None:
            self.brain.web_viewer.push_frame(image)

        elapsed = time.time() - self.brain.start_time
        if elapsed > 1.0:
            self.brain.fps = self.brain.frame_count / elapsed
            self.brain.frame_count = 0
            self.brain.start_time = time.time()

    def on_annotated_camera_image(self, cli, annotated_image, detections):
        """Handler for annotated camera images."""
        self.brain.latest_annotated_frame = annotated_image
        with self.brain._detections_lock:
            self.brain.latest_detections = detections
        self._stamp_detections_seen()
        self.brain.web.update_web_viewer()

    def _stamp_detections_seen(self):
        """Record the current frame index for every detected cube."""
        with self.brain._detections_lock:
            detected_cube_ids = {det.cube_id for det in self.brain.latest_detections}
        for cube_id in detected_cube_ids:
            self.brain._cube_last_seen_frame[cube_id] = self.brain.frame_count

    def on_debug_images(self, cli, debug_images, detections):
        """Handler for debug images from cube detection pipeline."""
        self.brain.latest_debug_images = debug_images

    def on_floor_projected_image(self, cli, floor_image, occupancy_grid):
        """Handler for floor-projected (bird's-eye) images."""
        self.brain.latest_floor_image = floor_image
        self.brain.latest_occupancy_grid = occupancy_grid
        self.brain.web_viewer.push_floor_frame(floor_image)
        # Sync floor grid dimensions to browser on first frame
        if self.brain.vision_processor and self.brain.vision_processor.floor_projector:
            fp = self.brain.vision_processor.floor_projector
            self.brain.web_viewer.set_floor_size(
                fp._floor_size[0], fp._floor_size[1]
            )

    def pil_to_cv2(self, pil_image):
        """Convert PIL Image to OpenCV format."""
        if pil_image is None:
            return None
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def add_text_overlay(self, frame, text_lines, position=(10, 30)):
        """Add text overlay to frame."""
        x, y = position
        for i, line in enumerate(text_lines):
            y_pos = y + i * 25
            (text_width, text_height), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x - 5, y_pos - text_height - 5),
                (x + text_width + 5, y_pos + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                frame, line, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    def display_frames(self):
        """Display raw, annotated, and floor-projected frames."""
        if self.brain.headless:
            return

        raw_cv2 = self.pil_to_cv2(self.brain.latest_raw_frame)
        annotated_cv2 = self.pil_to_cv2(self.brain.latest_annotated_frame)

        if raw_cv2 is None:
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                waiting_frame,
                "Waiting for camera frames...",
                (100, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow(self.brain.window_name, waiting_frame)
            return

        if annotated_cv2 is None:
            annotated_cv2 = raw_cv2.copy()

        height, width = raw_cv2.shape[:2]
        if annotated_cv2.shape[:2] != (height, width):
            annotated_cv2 = cv2.resize(annotated_cv2, (width, height))

        raw_labeled = raw_cv2.copy()
        annotated_labeled = annotated_cv2.copy()

        cv2.rectangle(raw_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            raw_labeled,
            "Normal Camera Feed",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(annotated_labeled, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            annotated_labeled,
            "Annotated (Detections)",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        if self.brain.show_fps:
            info_lines = [
                f"FPS: {self.brain.fps:.1f}",
                f"Cubes: {self._safe_detection_count()}",
                f"Detection: {'ON' if self.brain.detection_enabled else 'OFF'}",
            ]
            self.add_text_overlay(raw_labeled, info_lines, (10, height - 90))

            with self.brain._detections_lock:
                detections_snapshot = list(self.brain.latest_detections)
            if detections_snapshot:
                det_lines = [f"Detected Cubes: {len(detections_snapshot)}"]
                for i, det in enumerate(detections_snapshot[:3]):
                    det_lines.append(f"  ID:{det.cube_id} Conf:{det.confidence:.2f}")
                self.add_text_overlay(annotated_labeled, det_lines, (10, height - 90))

        top_row = np.hstack([raw_labeled, annotated_labeled])

        combined = top_row

        # Floor-projected view below the camera panels
        floor_cv2 = self.brain.latest_floor_image
        if floor_cv2 is not None:
            top_w = top_row.shape[1]
            fh, fw = floor_cv2.shape[:2]
            scale = max(1, top_w // fw)
            floor_up = cv2.resize(
                floor_cv2, (fw * scale, fh * scale), interpolation=cv2.INTER_NEAREST
            )
            fh_scaled, fw_scaled = floor_up.shape[:2]
            cv2.rectangle(floor_up, (0, 0), (fw_scaled, 30), (0, 0, 0), -1)
            cv2.putText(
                floor_up,
                "Floor (Bird's Eye)",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 200, 255),
                2,
            )
            # Pad or crop width to match top row, then vstack
            if fw_scaled < top_w:
                pad = np.zeros((fh_scaled, top_w - fw_scaled, 3), dtype=np.uint8)
                floor_up = np.hstack([floor_up, pad])
            elif fw_scaled > top_w:
                floor_up = floor_up[:, :top_w]
            combined = np.vstack([top_row, floor_up])

        cv2.imshow(self.brain.window_name, combined)
        # Resize window on first frame or when dimensions change
        ch, cw = combined.shape[:2]
        if not hasattr(self, "_last_combined_size") or self._last_combined_size != (
            ch,
            cw,
        ):
            cv2.resizeWindow(self.brain.window_name, cw, ch)
            self._last_combined_size = (ch, cw)

        for name, img in self.brain.latest_debug_images.items():
            cv2.imshow(f"Debug: {name}", img)

    def save_frames(self):
        """Save current frames to disk."""
        if self.brain.latest_raw_frame is None:
            print("No frames to save yet")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")

        raw_filename = f"frame_raw_{timestamp}.png"
        self.brain.latest_raw_frame.save(raw_filename)
        print(f"Saved raw frame: {raw_filename}")

        if self.brain.latest_annotated_frame is not None:
            annotated_filename = f"frame_annotated_{timestamp}.png"
            self.brain.latest_annotated_frame.save(annotated_filename)
            print(f"Saved annotated frame: {annotated_filename}")

        with self.brain._detections_lock:
            n_det = len(self.brain.latest_detections)
        print(f"   Detections: {n_det}")

    def _safe_detection_count(self) -> int:
        """Read the detection count under the lock (COZMO-3)."""
        with self.brain._detections_lock:
            return len(self.brain.latest_detections)
