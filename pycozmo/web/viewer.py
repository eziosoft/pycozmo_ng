"""Web viewer bridge — feeds measurements into world map, robot state trail."""

from pycozmo.cubes.cube_tracking_state import CubeTrackingState
from pycozmo.logger import logger
from pycozmo.navigation.dock_state import DockState
from pycozmo.vision.frame_state import FrameState


class WebBridge:
    """Bridges vision/cube data to the web viewer for 3D display.

    Uses typed state containers via DI: :class:`FrameState`,
    :class:`CubeTrackingState`, and :class:`DockState`.
    """

    def __init__(
        self,
        frame_state: FrameState,
        cube_tracking: CubeTrackingState,
        dock_state: DockState,
    ):
        self.fs = frame_state
        self.ct = cube_tracking
        self.ds = dock_state
        # Set by Brain after construction
        self.web_viewer = None
        self.cli = None
        self.vision_processor = None

    def update_web_viewer(self):
        """Feed the latest cube and face measurements into the world map.

        Cubes that have been invalidated (moved, forget_cube) are
        suppressed until they are re-detected stably. (COZMO-34)
        """
        if not self.vision_processor:
            return

        head_angle = self.cli.head_angle.radians if self.cli is not None else 0.0

        # Read persistent lateral offset (updated automatically during
        # startup calibration and each cube approach)
        lateral_offset = self.vision_processor.camera_lateral_offset_mm

        suppressed = self.ct.suppress.get("ids", set())
        fresh_frames = self.ct.suppress.get("fresh_frames", {})
        new_fresh_frames = {}

        for m in self.vision_processor.get_cube_measurements():
            cube_id = m["cube_id"]
            if cube_id in suppressed:
                n = fresh_frames.get(cube_id, 0) + 1
                new_fresh_frames[cube_id] = n
                if n < self.ct.suppress["fresh_frames_needed"]:
                    continue
                suppressed.discard(cube_id)
                logger.info(
                    "Cube %d re-settled - re-enabling measurements", cube_id + 1
                )

            self.web_viewer.update_cube_measurement(
                cube_id,
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
                m.get("num_faces"),
                camera_lateral_offset_mm=lateral_offset,
                rotation_source=m.get("rotation_source"),
            )

        self.ct.suppress["fresh_frames"] = new_fresh_frames

        # Feed face measurements into world map
        for fm in self.vision_processor.get_face_measurements():
            self.web_viewer.update_face_measurement(
                fm["face_id"],
                fm["position"],
                fm["confidence"],
                head_angle,
                camera_lateral_offset_mm=lateral_offset,
                label=f"Face {fm['face_id']}",
            )

        self.web_viewer.broadcast()

    def _on_robot_state_for_trail(self, c):
        """Push robot pose into world map for path tracking. (COZMO-27)"""
        try:
            pose = c.pose
            pos = pose.position.x_y_z
            yaw = pose.rotation.angle_z.radians
            self.web_viewer.update_robot_pose(
                pos, yaw, origin_id=pose.origin_id, record_path=True
            )
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to update robot pose: {e}")

        try:
            with self.ds.lock:
                on_charger = self.ds.on_charger
                charging = self.ds.charging
            self.web_viewer.update_telemetry(
                charging=charging,
                on_charger=on_charger,
                battery_voltage=round(c.battery_voltage, 3),
                left_wheel_speed=round(c.left_wheel_speed.mmps, 1),
                right_wheel_speed=round(c.right_wheel_speed.mmps, 1),
                head_angle=round(c.head_angle.degrees, 1),
                lift_height=round(c.lift_position.height.mm, 1),
                cliff_data=c.cliff_data,
            )
        except (AttributeError, ValueError) as e:
            logger.debug(f"Failed to update telemetry: {e}")

        self.web_viewer.broadcast()
