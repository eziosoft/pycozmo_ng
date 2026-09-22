"""

Cozmo protocol client and high-level API.

"""

import io
import json
import math
import time
from collections.abc import Callable, Sequence
from threading import Event, Lock

import numpy as np
from PIL import Image

from pycozmo.animation import anim, anim_controller, anim_encoder
from pycozmo.protocol import protocol_declaration, protocol_encoder
from pycozmo.sound import audio, tts
from pycozmo.transport import conn
from pycozmo.util import filter, object, robot, robot_debug
from pycozmo.vision import camera

from . import (
    event,
    exception,
    image_encoder,
    lights,
    logger,
    logger_animation,
    logger_robot,
    util,
)

__all__ = [
    "Client",
]


def build_polyline_packets(
    points: Sequence[tuple[float, float]],
    start: tuple[float, float],
    event_id: int,
    speed_mmps: float = 90.0,
    accel_mmps2: float = 40.0,
    decel_mmps2: float = 40.0,
    final_heading_rad: float | None = None,
) -> list:
    """Build the packet sequence for a multi-segment firmware path.

    Returns ``[ClearPath, AppendPathSegLine..., (AppendPathSegPointTurn),
    ExecutePath]``. Consecutive points closer than 10 mm to the previous
    kept point (starting from *start*) are dropped so the firmware never
    sees zero-length segments. Empty list when there is nothing to
    execute.
    """
    waypoints: list[tuple[float, float]] = []
    fx, fy = float(start[0]), float(start[1])
    for x, y in points:
        x, y = float(x), float(y)
        if math.hypot(x - fx, y - fy) >= 10.0:
            waypoints.append((x, y))
            fx, fy = x, y

    if not waypoints and final_heading_rad is None:
        return []

    packets: list = [protocol_encoder.ClearPath()]
    px, py = float(start[0]), float(start[1])
    for tx, ty in waypoints:
        packets.append(
            protocol_encoder.AppendPathSegLine(
                from_x=px,
                from_y=py,
                to_x=tx,
                to_y=ty,
                speed_mmps=speed_mmps,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
            )
        )
        px, py = tx, ty

    if final_heading_rad is not None:
        packets.append(
            protocol_encoder.AppendPathSegPointTurn(
                x=px,
                y=py,
                angle_rad=float(final_heading_rad),
                angle_tolerance_rad=0.05,
                speed_mmps=25.0,
                accel_mmps2=accel_mmps2,
                decel_mmps2=decel_mmps2,
            )
        )

    packets.append(protocol_encoder.ExecutePath(event_id=event_id))
    return packets


class Client(event.Dispatcher):
    """Cozmo protocol client and high-level API class."""

    def __init__(
        self,
        robot_addr: tuple[str, int] | None = None,
        protocol_log_messages: list | None = None,
        auto_initialize: bool = True,
        enable_animations: bool = True,
        enable_procedural_face: bool = True,
        tts_rate: int = 150,
    ) -> None:
        super().__init__()
        # Whether to automatically initialize the robot when connection is established.
        self.auto_initialize = bool(auto_initialize)

        self.conn = conn.Connection(robot_addr, protocol_log_messages)
        self.conn.add_child_dispatcher(self)
        self.anim_controller = anim_controller.AnimationController(self)
        self.anim_controller.enable_animations(auto_initialize and enable_animations)
        self.anim_controller.enable_procedural_face(
            auto_initialize and enable_animations and enable_procedural_face
        )

        self.serial_number_head = None
        self.robot_fw_sig = None
        self.serial_number = None
        self.body_hw_version = None
        self.body_color = None
        # Robot state
        # Heading in X-Y plane.
        self.pose_frame_id = 0
        self.pose = util.Pose(
            0.0, 0.0, 0.0, angle_z=util.Angle(radians=0.0), origin_id=1
        )
        self.pose_pitch = util.Angle(radians=0.0)
        # Incrementing event_id for execute_polyline paths (avoids clashing
        # with go_to_pose's fixed event_id=1).
        self._path_event_id = 100
        self.head_angle = util.Angle(radians=robot.MIN_HEAD_ANGLE.radians)
        self.left_wheel_speed = util.Speed(mmps=0.0)
        self.right_wheel_speed = util.Speed(mmps=0.0)
        self.lift_position = robot.LiftPosition(height=robot.MIN_LIFT_HEIGHT)
        self.battery_voltage = 0.0
        self.accel = util.Vector3(0.0, 0.0, 0.0)
        self.gyro = util.Vector3(0.0, 0.0, 0.0)
        self.robot_status = 0
        self.robot_orientation = robot.RobotOrientation.ON_THREADS
        self.robot_picked_up = False
        self.robot_moving = False
        self.cliff_data = 0
        # Animation state
        self.num_anim_bytes_played = 0
        self.num_audio_frames_played = 0
        self.enabled_anim_tracks = 0
        self.tag = 0
        self.client_drop_count = 0
        # Camera state
        self.last_image_timestamp = None
        # Object state
        self.available_objects = {}
        self.connected_objects = {}
        # Filters
        self.packet_type_filter = filter.Filter()
        self.packet_type_filter.deny_ids({protocol_declaration.PacketType.PING.value})
        self.packet_id_filter = filter.Filter()
        self._reset_partial_state()
        # Animations
        self._clip_metadata = {}
        self._clips = {}
        self._ppclips = {}
        self._next_anim_id = 1
        self.animation_groups = {}
        self._tts = tts.TTS(rate=tts_rate)

    def start(self) -> None:
        logger.debug("Starting client...")
        self.add_handler(protocol_encoder.HardwareInfo, self._on_hardware_info)
        self.add_handler(
            protocol_encoder.FirmwareSignature, self._on_firmware_signature
        )
        self.add_handler(protocol_encoder.BodyInfo, self._on_body_info)
        self.add_handler(protocol_encoder.ImageChunk, self._on_image_chunk)
        self.add_handler(protocol_encoder.RobotState, self._on_robot_state)
        self.add_handler(protocol_encoder.AnimationState, self._on_animation_state)
        self.add_handler(protocol_encoder.ObjectAvailable, self._on_object_available)
        self.add_handler(
            protocol_encoder.ObjectConnectionState, self._on_object_connection_state
        )
        self.add_handler(protocol_encoder.DebugData, self._on_debug_data)
        self.add_handler(event.EvtRobotPickedUpChange, self._on_robot_picked_up)
        self.add_handler(event.EvtRobotWheelsMovingChange, self._on_robot_moving)
        self.conn.start()

    def stop(self) -> None:
        logger.debug("Stopping client...")
        self.conn.stop()
        self.anim_controller.stop()
        self.del_all_handlers()

    def connect(self) -> None:
        logger.debug("Connecting...")
        self.conn.connect()

    def disconnect(self) -> None:
        logger.debug("Disconnecting...")
        self.conn.disconnect()

    def _enable_robot(self):
        # Enable
        pkt = protocol_encoder.Enable()
        self.conn.send(pkt)
        self.conn.send(pkt)  # This repetition seems to trigger BodyInfo

    def _initialize_robot(self):
        # Set world frame origin to (0,0,0), frame ID to 0, and origin ID to 1.
        pkt = protocol_encoder.SetOrigin()
        self.conn.send(pkt)
        # Set timestamp to 0. Also enables RobotState and ObjectAvailable events. Requires Enable (0x25).
        pkt = protocol_encoder.SyncTime()
        self.conn.send(pkt)

        # TODO: Proper waiting for motor calibration to complete.
        time.sleep(0.5)

        self.anim_controller.start()

        self.dispatch(event.EvtRobotReady, self)

    def _on_hardware_info(self, cli, pkt: protocol_encoder.HardwareInfo):
        del cli
        self.serial_number_head = pkt.serial_number_head

    def _on_firmware_signature(self, cli, pkt: protocol_encoder.FirmwareSignature):
        del cli
        self.robot_fw_sig = json.loads(pkt.signature)
        logger.info("Firmware version %s.", self.robot_fw_sig["version"])
        if self.robot_fw_sig.get("build") == "FACTORY":
            logger.warning(
                "Factory/recovery firmware detected. Functionality is degraded."
            )
        elif self.robot_fw_sig["version"] < protocol_declaration.FIRMWARE_VERSION:
            logger.warning(
                f"Old firmware detected. PyCozmo works best with v{protocol_declaration.FIRMWARE_VERSION}. Functionality may be degraded."
            )
        self._enable_robot()

    def _on_body_info(self, cli, pkt: protocol_encoder.BodyInfo):
        del cli
        self.serial_number = pkt.serial_number
        self.body_hw_version = pkt.body_hw_version
        self.body_color = pkt.body_color
        logger.info(
            "Body S/N 0x%08x, HW version %i, color %i.",
            self.serial_number,
            self.body_hw_version,
            self.body_color.value,
        )
        if self.auto_initialize:
            self._initialize_robot()
        self.dispatch(event.EvtRobotFound, self)

    def wait_for(self, evt, timeout: float | None = None):
        e = Event()
        self.add_handler(evt, lambda cli: e.set(), one_shot=True)
        if not e.wait(timeout):
            raise exception.Timeout(f"Timeout waiting for event {evt}")

    def wait_for_factory_ids(
        self,
        factory_ids,
        connect: bool = True,
        timeout: float = 10.0,
    ):
        """Wait for ObjectConnectionState from each requested factory_id.

        Returns a dict  ``{factory_id: object_id}``  mapping every
        *factory_id* to the object_id the robot assigned.

        Raises ``Timeout`` if not all cubes report within *timeout*.
        """
        remaining = set(factory_ids)
        result = {}
        lock = Lock()
        done = Event()

        def _handler(cli, pkt):
            del cli
            if pkt.connected != connect:
                return
            if pkt.factory_id not in remaining:
                return
            with lock:
                remaining.discard(pkt.factory_id)
                result[pkt.factory_id] = pkt.object_id
                if not remaining:
                    done.set()

        self.add_handler(protocol_encoder.ObjectConnectionState, _handler)
        try:
            if not done.wait(timeout):
                raise exception.Timeout(
                    "Timed out waiting for factory_ids: {}".format(
                        ", ".join(f"0x{f:08x}" for f in remaining)
                    )
                )
        finally:
            self.del_handler(protocol_encoder.ObjectConnectionState, _handler)
        return result

    def wait_for_robot(self, timeout: float = 5.0) -> None:
        if not self.robot_fw_sig:
            try:
                self.wait_for(event.EvtRobotFound, timeout=timeout)
            except exception.Timeout as e:
                raise exception.ConnectionTimeout("Failed to connect to Cozmo.") from e

        if not self.serial_number:
            try:
                self.wait_for(event.EvtRobotReady, timeout=timeout)
            except exception.Timeout as e:
                raise exception.ConnectionTimeout("Failed to initialize Cozmo.") from e

    def _reset_partial_state(self):
        self._partial_image_timestamp = None
        self._partial_data = None
        self._partial_image_id = None
        self._partial_invalid = False
        self._partial_size = 0
        self._partial_image_encoding = None
        self._partial_image_resolution = None
        self._last_chunk_id = -1

    def _on_image_chunk(self, cli, pkt: protocol_encoder.ImageChunk):
        del cli
        if self._partial_image_id is not None and pkt.chunk_id == 0:
            if not self._partial_invalid:
                logger.debug("Lost final chunk of image - discarding.")
            self._partial_image_id = None

        if self._partial_image_id is None:
            if pkt.chunk_id != 0:
                if not self._partial_invalid:
                    logger.debug("Received chunk of broken image.")
                self._partial_invalid = True
                return
            # Discard any previous in-progress image
            self._reset_partial_state()
            self._partial_image_timestamp = pkt.frame_timestamp
            self._partial_image_id = pkt.image_id
            self._partial_image_encoding = protocol_encoder.ImageEncoding(
                pkt.image_encoding
            )
            self._partial_image_resolution = protocol_encoder.ImageResolution(
                pkt.image_resolution
            )

            image_resolution = protocol_encoder.ImageResolution(pkt.image_resolution)
            width, height = camera.RESOLUTIONS[image_resolution]
            max_size = width * height * 3  # 3 bytes per pixel (RGB)
            self._partial_data = np.empty(max_size, dtype=np.uint8)

        if (
            pkt.chunk_id != (self._last_chunk_id + 1)
            or pkt.image_id != self._partial_image_id
        ):
            logger.debug(
                "Image missing chunks - discarding (last_chunk_id=%d partial_image_id=%s).",
                self._last_chunk_id,
                self._partial_image_id,
            )
            self._discard_partial_image()
            return

        offset = self._partial_size
        self._partial_data[offset : offset + len(pkt.data)] = np.frombuffer(
            pkt.data, dtype=np.uint8
        )
        self._partial_size += len(pkt.data)
        self._last_chunk_id = pkt.chunk_id

        if pkt.chunk_id == pkt.image_chunk_count - 1:
            self._process_completed_image()
            self._reset_partial_state()

    def _discard_partial_image(self) -> None:
        """Discard an incomplete camera frame (e.g. from AP frame loss)."""
        self._reset_partial_state()
        self._partial_invalid = True

    def _process_completed_image(self):
        data = self._partial_data[0 : self._partial_size]

        # The first byte of the image is whether or not it is in color
        is_color_image = data[0] != 0

        if (
            self._partial_image_encoding
            == protocol_encoder.ImageEncoding.JPEGMinimizedGray
        ):
            width, height = camera.RESOLUTIONS[self._partial_image_resolution]

            if is_color_image:
                # Color images are half width
                width = width // 2
                data = camera.minicolor_to_jpeg(data, width, height)
            else:
                data = camera.minigray_to_jpeg(data, width, height)

        image = Image.open(io.BytesIO(data)).convert("RGB")

        # Color images need to be resized to the proper resolution
        if is_color_image:
            size = camera.RESOLUTIONS[self._partial_image_resolution]
            image = image.resize(size)

        self._latest_image = image
        self.last_image_timestamp = self._partial_image_timestamp

        self.dispatch(event.EvtNewRawCameraImage, self, image)

    def _on_robot_state(self, cli, pkt: protocol_encoder.RobotState):
        del cli
        self.pose_frame_id = pkt.pose_frame_id
        self.pose = util.Pose(
            pkt.pose_x,
            pkt.pose_y,
            pkt.pose_z,
            angle_z=util.Angle(radians=pkt.pose_angle_rad),
            origin_id=pkt.pose_origin_id,
        )
        self.pose_pitch = util.Angle(radians=pkt.pose_pitch_rad)
        self.head_angle = util.Angle(radians=pkt.head_angle_rad)
        self.left_wheel_speed = util.Speed(mmps=pkt.lwheel_speed_mmps)
        self.right_wheel_speed = util.Speed(mmps=pkt.rwheel_speed_mmps)
        self.lift_position = robot.LiftPosition(
            height=util.Distance(mm=pkt.lift_height_mm)
        )
        self.battery_voltage = pkt.battery_voltage
        self.accel = util.Vector3(pkt.accel_x, pkt.accel_y, pkt.accel_z)
        self.gyro = util.Vector3(pkt.gyro_x, pkt.gyro_y, pkt.gyro_z)
        self.cliff_data = pkt.cliff_data_raw[0] if pkt.cliff_data_raw else 0
        old_status = self.robot_status
        self.robot_status = pkt.status
        self.dispatch(event.EvtRobotStateUpdated, self)
        # Dispatch status flag change events.
        for flag, evt in event.STATUS_EVENTS.items():
            if (old_status & flag) != (pkt.status & flag):
                state = (pkt.status & flag) != 0
                logger.debug("%s: %i", robot.RobotStatusFlagNames[flag], state)
                self.dispatch(evt, self, state)
        # Orientation
        if pkt.pose_angle_rad < -0.4:
            robot_orientation = robot.RobotOrientation.ON_LEFT_SIDE
        elif pkt.pose_angle_rad > 0.4:
            robot_orientation = robot.RobotOrientation.ON_RIGHT_SIDE
        elif pkt.pose_pitch_rad < -1.0:
            robot_orientation = robot.RobotOrientation.ON_FACE
        elif pkt.pose_pitch_rad > 1.0:
            robot_orientation = robot.RobotOrientation.ON_BACK
        else:
            robot_orientation = robot.RobotOrientation.ON_THREADS
        if self.robot_orientation != robot_orientation:
            self.robot_orientation = robot_orientation
            self.dispatch(event.EvtRobotOrientationChange, self, robot_orientation)

    def _on_robot_picked_up(self, cli, state):
        del cli
        if state:
            self.robot_picked_up = True
        else:
            # Robot put down - reset world frame origin.
            self.robot_picked_up = False
            pkt = protocol_encoder.SetOrigin(
                pose_frame_id=self.pose_frame_id + 1,
                pose_origin_id=self.pose.origin_id + 1,
            )
            self.conn.send(pkt)

    def _on_robot_moving(self, cli, state):
        self.robot_moving = state

    def _on_animation_state(self, cli, pkt: protocol_encoder.AnimationState):
        del cli
        self.num_anim_bytes_played = pkt.num_anim_bytes_played
        self.num_audio_frames_played = pkt.num_audio_frames_played
        self.enabled_anim_tracks = pkt.enabled_anim_tracks
        self.tag = pkt.tag
        self.client_drop_count = pkt.client_drop_count

    def _on_object_available(self, cli, pkt: protocol_encoder.ObjectAvailable):
        del cli
        factory_id = pkt.factory_id
        object_type = protocol_encoder.ObjectType(pkt.object_type)
        obj = object.Object(factory_id=factory_id, object_type=object_type)
        if factory_id not in self.available_objects:
            self.available_objects[factory_id] = obj
            logger.debug(
                "Object of type %s with S/N 0x%08x available.",
                str(obj.object_type),
                obj.factory_id,
            )

    def _on_object_connection_state(
        self, cli, pkt: protocol_encoder.ObjectConnectionState
    ):
        del cli
        if pkt.connected:
            # Connected
            self.connected_objects[pkt.object_id] = {
                "factory_id": pkt.factory_id,
                "object_type": pkt.object_type,
            }
        else:
            # Disconnected
            if pkt.object_id in self.connected_objects:
                del self.connected_objects[pkt.object_id]

    def _on_debug_data(self, cli, pkt: protocol_encoder.DebugData):
        del cli
        msg = robot_debug.get_debug_message(pkt.name_id, pkt.format_id, pkt.args)
        logger_robot.log(robot_debug.get_log_level(pkt.level), msg)

    def set_head_angle(
        self,
        angle: float,
        accel: float = 10.0,
        max_speed: float = 10.0,
        duration: float = 0.0,
    ):
        pkt = protocol_encoder.SetHeadAngle(
            angle_rad=angle,
            accel_rad_per_sec2=accel,
            max_speed_rad_per_sec=max_speed,
            duration_sec=duration,
        )
        self.conn.send(pkt)

    def move_head(self, speed: float) -> None:
        pkt = protocol_encoder.MoveHead(speed_rad_per_sec=speed)
        self.conn.send(pkt)

    def set_lift_height(
        self,
        height: float,
        accel: float = 10.0,
        max_speed: float = 10.0,
        duration: float = 0.0,
    ):
        pkt = protocol_encoder.SetLiftHeight(
            height_mm=height,
            accel_rad_per_sec2=accel,
            max_speed_rad_per_sec=max_speed,
            duration_sec=duration,
        )
        self.conn.send(pkt)

    def move_lift(self, speed: float) -> None:
        pkt = protocol_encoder.MoveLift(speed_rad_per_sec=speed)
        self.conn.send(pkt)

    def drive_wheels(
        self,
        lwheel_speed: float,
        rwheel_speed: float,
        lwheel_acc: float | None = 0.0,
        rwheel_acc: float | None = 0.0,
        duration: float | None = None,
    ) -> None:
        pkt = protocol_encoder.DriveWheels(
            lwheel_speed_mmps=lwheel_speed,
            rwheel_speed_mmps=rwheel_speed,
            lwheel_accel_mmps2=lwheel_acc,
            rwheel_accel_mmps2=rwheel_acc,
        )
        self.conn.send(pkt)
        if duration is not None:
            time.sleep(duration)
            self.stop_all_motors()

    def stop_all_motors(self) -> None:
        pkt = protocol_encoder.StopAllMotors()
        self.conn.send(pkt)

    def go_to_pose(
        self,
        pose: util.Pose,
        relative_to_robot: bool = False,
        timeout_s: float | None = 30.0,
        position_tol_mm: float = 15.0,
        angle_tol_rad: float = 0.15,
    ) -> None:
        """Move to a specific pose (position and orientation).

        Sends the path and blocks until either:
        1. A ``PathFollowingEvent`` reports completion/abort, **or**
        2. The robot pose is within ``position_tol_mm`` mm and
           ``angle_tol_rad`` rad of the target (pose polling).

        Falls back to polling because Cozmo firmware often omits
        ``PathFollowingEvent`` for simple paths (COZMO-53). On timeout
        only a warning is logged — the robot may still be moving.
        """

        if relative_to_robot:
            pose = util.Pose(
                self.pose.position.x,
                self.pose.position.y,
                self.pose.position.z,
                angle_z=self.pose.rotation.angle_z,
            ).define_pose_relative_this(pose)

        e = Event()
        completed = False

        def event_wait(_, pkt2: protocol_encoder.PathFollowingEvent):
            nonlocal completed
            if pkt2.event_type != protocol_encoder.PathEventType.PATH_STARTED:
                completed = True
                e.set()

        handler = self.add_handler(protocol_encoder.PathFollowingEvent, event_wait)
        try:
            pkt = protocol_encoder.AppendPathSegLine(
                from_x=self.pose.position.x,
                from_y=self.pose.position.y,
                to_x=pose.position.x,
                to_y=pose.position.y,
                speed_mmps=100.0,
                accel_mmps2=20.0,
                decel_mmps2=20.0,
            )
            self.conn.send(pkt)
            pkt = protocol_encoder.AppendPathSegPointTurn(
                x=pose.position.x,
                y=pose.position.y,
                angle_rad=pose.rotation.angle_z.radians,
                angle_tolerance_rad=0.01,
                speed_mmps=25.0,
                accel_mmps2=20.0,
                decel_mmps2=20.0,
            )
            self.conn.send(pkt)
            pkt = protocol_encoder.ExecutePath(event_id=1)
            self.conn.send(pkt)

            # Wait for either the event OR pose proximity — whichever comes first.
            deadline = time.perf_counter() + (timeout_s or 30.0)
            dist = float("inf")
            angle_err = float("inf")
            while not completed and time.perf_counter() < deadline:
                # Fast path: event arrived
                if e.wait(timeout=0.25):
                    break
                # Pose-polling fallback: check if we're close enough
                dx = self.pose.position.x - pose.position.x
                dy = self.pose.position.y - pose.position.y
                dist = math.hypot(dx, dy)
                angle_err = abs(
                    math.remainder(
                        self.pose.rotation.angle_z.radians
                        - pose.rotation.angle_z.radians,
                        2 * math.pi,
                    )
                )
                if dist <= position_tol_mm and angle_err <= angle_tol_rad:
                    completed = True
                    break
        finally:
            self.del_handler(protocol_encoder.PathFollowingEvent, handler)

        if not completed:
            logger.warning(
                "go_to_pose: timed out after %.0fs — dist=%.1f mm, angle_err=%.1f deg",
                timeout_s if timeout_s is not None else 0,
                dist,
                math.degrees(angle_err),
            )

    def execute_polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        speed_mmps: float = 90.0,
        accel_mmps2: float = 40.0,
        decel_mmps2: float = 40.0,
        final_heading_rad: float | None = None,
        timeout_s: float | None = None,
        position_tol_mm: float = 20.0,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Follow a multi-segment path with the robot's built-in path follower.

        Sends ``ClearPath`` + one ``AppendPathSegLine`` per segment
        (optionally a final ``AppendPathSegPointTurn``) + ``ExecutePath``
        and blocks until the firmware reports completion, the pose
        reaches the final point, timeout, or *should_interrupt* returns
        True.

        Unlike the Python-side closed loop (``Navigator._drive_to_point``),
        the trajectory — accel/decel ramps and wheel control — runs on the
        robot at high rate, independent of WiFi latency. This is the
        recommended way to drive along a planned (e.g. A*) polyline.

        Args:
            points: Waypoints in world frame (mm, X forward / Y left), in
                order. The first segment starts at the robot's current
                pose. Degenerate segments (< 10 mm) are dropped.
            speed_mmps: Cruise speed for the line segments.
            accel_mmps2: Acceleration limit (mm/s²).
            decel_mmps2: Deceleration limit (mm/s²).
            final_heading_rad: If set, finish with an in-place turn to
                this world heading at the last point.
            timeout_s: Overall timeout. Defaults to a length-based
                estimate (3× straight-line time at *speed_mmps*, min 10 s).
            position_tol_mm: Pose-proximity tolerance for the polling
                fallback (firmware sometimes omits ``PathFollowingEvent``
                for simple paths — COZMO-53).
            should_interrupt: Polled ~10× per second while waiting; when
                it returns True the motors are stopped and the method
                returns False. Used for cancellation and mid-path
                replanning.

        Returns:
            True when the path completed (firmware event or pose
            proximity), False when interrupted or timed out short of the
            goal.
        """
        self._path_event_id = self._path_event_id % 65535 + 1
        event_id = self._path_event_id
        packets = build_polyline_packets(
            points,
            (self.pose.position.x, self.pose.position.y),
            event_id,
            speed_mmps=speed_mmps,
            accel_mmps2=accel_mmps2,
            decel_mmps2=decel_mmps2,
            final_heading_rad=final_heading_rad,
        )
        if not packets:
            # Nothing to drive — already at (or within 10 mm of) the goal.
            return True

        end_x = self.pose.position.x
        end_y = self.pose.position.y
        path_len = 0.0
        for pkt in packets:
            if isinstance(pkt, protocol_encoder.AppendPathSegLine):
                path_len += math.hypot(pkt.to_x - pkt.from_x, pkt.to_y - pkt.from_y)
                end_x, end_y = pkt.to_x, pkt.to_y
            elif isinstance(pkt, protocol_encoder.AppendPathSegPointTurn):
                end_x, end_y = pkt.x, pkt.y

        if timeout_s is None:
            timeout_s = max(10.0, path_len / max(speed_mmps, 1.0) * 3.0 + 5.0)

        state: dict = {"event": None}

        def on_event(_, pkt2: protocol_encoder.PathFollowingEvent) -> None:
            if pkt2.event_id == event_id:
                state["event"] = pkt2.event_type

        handler = self.add_handler(protocol_encoder.PathFollowingEvent, on_event)
        t0 = time.perf_counter()
        try:
            for pkt in packets:
                self.conn.send(pkt)

            deadline = t0 + timeout_s
            while time.perf_counter() < deadline:
                ev = state["event"]
                if ev == protocol_encoder.PathEventType.PATH_COMPLETED:
                    return True
                if ev == protocol_encoder.PathEventType.PATH_INTERRUPTED:
                    return False
                if should_interrupt is not None and should_interrupt():
                    self.stop_all_motors()
                    return False
                # Pose-polling fallback (COZMO-53). Gated by a 1 s grace
                # period so a path that starts within the tolerance (e.g.
                # a turn-only command) isn't reported done before the
                # firmware has had a chance to run.
                if time.perf_counter() - t0 >= 1.0:
                    dx = self.pose.position.x - end_x
                    dy = self.pose.position.y - end_y
                    if math.hypot(dx, dy) <= position_tol_mm:
                        return True
                time.sleep(0.1)

            dx = self.pose.position.x - end_x
            dy = self.pose.position.y - end_y
            return math.hypot(dx, dy) <= position_tol_mm
        finally:
            self.del_handler(protocol_encoder.PathFollowingEvent, handler)

    def set_backpack_lights(
        self, left_light, front_light, center_light, rear_light, right_light
    ) -> None:
        pkt = protocol_encoder.LightStateCenter(
            states=(front_light, center_light, rear_light)
        )
        self.conn.send(pkt)
        pkt = protocol_encoder.LightStateSide(states=(left_light, right_light))
        self.conn.send(pkt)

    def set_center_backpack_lights(self, light) -> None:
        self.set_backpack_lights(
            lights.off_light, light, light, light, lights.off_light
        )

    def set_all_backpack_lights(self, light) -> None:
        self.set_backpack_lights(light, light, light, light, light)

    def set_backpack_lights_off(self) -> None:
        self.set_backpack_lights(
            lights.off_light,
            lights.off_light,
            lights.off_light,
            lights.off_light,
            lights.off_light,
        )

    def set_head_light(self, enable: bool) -> None:
        pkt = protocol_encoder.SetHeadLight(enable=enable)
        self.conn.send(pkt)

    def enable_camera(
        self,
        enable: bool = True,
        color: bool = False,
        resolution: "protocol_encoder.ImageResolution" = None,
    ) -> None:
        """Enable or disable camera image streaming in color or grayscale.

        Args:
            enable: If True, request streaming; if False, turn the camera off.
            color: If True, ask for color images; if False, grayscale.
            resolution: ImageResolution enum (e.g. ImageResolution.QVGA = 320x240
                — the default, ImageResolution.QQVGA = 160x120, …). Smaller
                resolutions are ~4x less Wi-Fi bandwidth per frame, which
                matters because the Cozmo AP is lossy (97-99% delivery in
                practice) and a saturated stream drops the connection
                within ~1 minute. (COZMO-20)
        """
        if enable:
            image_send_mode = protocol_encoder.ImageSendMode.Stream
        else:
            image_send_mode = protocol_encoder.ImageSendMode.Off
        image_resolution = (
            resolution
            if resolution is not None
            else protocol_encoder.ImageResolution.QVGA
        )
        pkt = protocol_encoder.EnableCamera(
            image_send_mode=image_send_mode,
            image_resolution=image_resolution,
        )
        self.conn.send(pkt)
        pkt = protocol_encoder.EnableColorImages(enable=color)
        self.conn.send(pkt)

    def set_camera_params(
        self,
        gain: float = 0.0,
        exposure_ms: int = 0,
        auto_exposure_enabled: bool = False,
    ) -> None:
        """Set camera parameters: gain, exposure time, and auto exposure."""
        pkt = protocol_encoder.SetCameraParams(
            gain=gain,
            exposure_ms=exposure_ms,
            auto_exposure_enabled=auto_exposure_enabled,
        )
        self.conn.send(pkt)

    def clear_screen(self) -> None:
        pkt = protocol_encoder.DisplayImage(image=b"\x3f\x3f")
        self.anim_controller.display_image(pkt)

    def display_image(self, im: Image, duration: float | None = None) -> None:
        encoder = image_encoder.ImageEncoder(im)
        buf = bytes(encoder.encode())
        pkt = protocol_encoder.DisplayImage(image=buf)
        self.anim_controller.display_image(pkt)
        if duration is not None:
            time.sleep(duration)
            self.clear_screen()

    def _load_clips(self, fspec: str) -> None:
        start_time = time.perf_counter()

        if fspec.endswith(".bin"):
            clips = anim_encoder.AnimClips.from_fb_file(fspec)
        elif fspec.endswith(".json"):
            clips = anim_encoder.AnimClips.from_json_file(fspec)
        else:
            raise ValueError("Unsupported animation file format.")
        for clip in clips.clips:
            self._clips[clip.name] = clip

        logger.debug(f"Loaded {fspec} in {time.perf_counter() - start_time:.02f} s.")

    def play_anim_ppclip(self, ppclip: anim.PreprocessedClip) -> None:
        # Ensure no other animation is playing.
        self.cancel_anim()

        # Start animation.
        pkt = protocol_encoder.StartAnimation(anim_id=self._next_anim_id)
        self.anim_controller.play_anim_frame(None, None, (pkt,))
        self._next_anim_id += 1

        # Send frames to the animation controller.
        frames = sorted(ppclip.keyframes.keys())
        num_frames = len(frames)
        time_ms = 0
        for i in range(num_frames):
            audio_pkt = None
            image_pkt = None
            pkts = []
            for action in ppclip.keyframes[frames[i]]:
                if isinstance(action, protocol_encoder.OutputAudio):
                    audio_pkt = action
                elif isinstance(action, protocol_encoder.DisplayImage):
                    image_pkt = action
                elif isinstance(action, protocol_encoder.Packet):
                    pkts.append(action)
            self.anim_controller.play_anim_frame(audio_pkt, image_pkt, pkts)
            time_ms += 33

            # Pause.
            if i < num_frames - 1:
                target_ms = time_ms + frames[i + 1] - frames[i]
                while target_ms > time_ms:
                    self.anim_controller.play_anim_frame(None, None, None)
                    time_ms += 33

        # End animation.
        pkt = protocol_encoder.EndAnimation()
        self.anim_controller.play_anim_frame(None, None, (pkt,))

    def play_anim(self, name: str) -> None:
        if not self._clip_metadata:
            raise ValueError("Animations not loaded.")
        elif name not in self._clip_metadata:
            raise ValueError("Unknown clip name.")

        if name not in self._ppclips:
            if name not in self._clips:
                self._load_clips(self._clip_metadata[name].fspec)
            clip = self._clips[name]
            self._ppclips[name] = anim.PreprocessedClip.from_anim_clip(clip)

        ppclip = self._ppclips[name]
        self.play_anim_ppclip(ppclip)

    def cancel_anim(self) -> None:
        self.anim_controller.cancel_anim()

    def play_anim_group(self, anim_group_name: str) -> None:
        logger_animation.info(f"Playing animation group {anim_group_name}")
        animation_group = self.animation_groups.get(anim_group_name)
        if not animation_group:
            logger_animation.error(f"Failed to find animation group {anim_group_name}.")
            return
        member = animation_group.choose_member()
        logger_animation.info(f"Playing animation {member.name}")
        self.play_anim(member.name)

    def load_anims(self) -> None:
        util.check_assets()
        anim_dir = str(util.get_cozmo_anim_dir())
        self._clip_metadata = anim_encoder.get_clip_metadata(anim_dir)
        self._clips = {}
        resource_dir = str(util.get_cozmo_asset_dir())
        self.animation_groups = anim.load_animation_groups(resource_dir)

    def get_anim_names(self) -> set:
        return set(self._clip_metadata.keys())

    @property
    def anim_names(self) -> set:
        return self.get_anim_names()

    def set_volume(self, level: int) -> None:
        """Set audio output volume to a level in the range 0-65535."""
        pkt = protocol_encoder.SetRobotVolume(level=level)
        self.conn.send(pkt)

    def play_audio(self, fspec: str) -> None:
        pkts = audio.load_wav(fspec)
        self.anim_controller.play_audio(pkts)

    def say(self, text: str) -> None:
        """Say text through the Cozmo robot using offline TTS.

        Synthesizes text to a WAV file (cached for repeated calls)
        and plays it through the robot's audio pipeline.

        Args:
            text: The text to speak.
        """
        self._tts.say(text, self)

    def activate_behavior(self, behavior):
        self.add_child_dispatcher(behavior)
        behavior.activate()

    def deactivate_behavior(self, behavior):
        self.del_child_dispatcher(behavior)
        behavior.deactivate()

    def enable_animations(self, enabled: bool = True) -> None:
        self.anim_controller.enable_animations(enabled)

    def enable_procedural_face(self, enabled: bool = True) -> None:
        self.anim_controller.enable_procedural_face(enabled)
