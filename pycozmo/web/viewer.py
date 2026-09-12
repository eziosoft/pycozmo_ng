"""Web viewer bridge — feeds measurements into world map, robot state trail."""

from pycozmo.logger import logger_cubes


class WebBridge:
    """Bridges vision/cube data to the web viewer for 3D display.

    References shared state on ``self.brain`` (the :class:`App` core).
    """

    def __init__(self, app):
        self.brain = app

    def update_web_viewer(self):
        """Feed the latest cube measurements into the world map.

        Cubes that have been invalidated (moved, forget_cube) are
        suppressed until they are re-detected stably. (COZMO-34)
        """
        if not self.brain.vision_processor:
            return

        head_angle = (
            self.brain.cli.head_angle.radians if self.brain.cli is not None else 0.0
        )

        suppressed = self.brain._cube_suppress.get("ids", set())
        fresh_frames = self.brain._cube_suppress.get("fresh_frames", {})
        new_fresh_frames = {}

        for m in self.brain.vision_processor.get_cube_measurements():
            cube_id = m["cube_id"]
            if cube_id in suppressed:
                n = fresh_frames.get(cube_id, 0) + 1
                new_fresh_frames[cube_id] = n
                if n < self.brain._cube_suppress["fresh_frames_needed"]:
                    continue
                suppressed.discard(cube_id)
                logger_cubes.info(
                    "Cube %d re-settled - re-enabling measurements", cube_id + 1
                )

            self.brain.web_viewer.update_cube_measurement(
                cube_id,
                m["position"],
                m["rotation_matrix"],
                m["confidence"],
                head_angle,
                m.get("num_faces"),
            )

        self.brain._cube_suppress["fresh_frames"] = new_fresh_frames

        self.brain.web_viewer.broadcast()

    def _on_robot_state_for_trail(self, c):
        """Push robot pose into world map for path tracking. (COZMO-27)"""
        try:
            pose = c.pose
            pos = pose.position.x_y_z
            yaw = pose.rotation.angle_z.radians
            self.brain.web_viewer.update_robot_pose(
                pos, yaw, origin_id=pose.origin_id, record_path=True
            )
        except Exception:  # noqa: BLE001
            pass

        try:
            with self.brain._dock_lock:
                on_charger = self.brain._on_charger
                charging = self.brain._charging
            self.brain.web_viewer.update_telemetry(
                charging=charging,
                on_charger=on_charger,
                battery_voltage=round(c.battery_voltage, 3),
                left_wheel_speed=round(c.left_wheel_speed.mmps, 1),
                right_wheel_speed=round(c.right_wheel_speed.mmps, 1),
                head_angle=round(c.head_angle.degrees, 1),
                lift_height=round(c.lift_position.height.mm, 1),
            )
        except Exception:  # noqa: BLE001
            pass

        self.brain.web_viewer.broadcast()
