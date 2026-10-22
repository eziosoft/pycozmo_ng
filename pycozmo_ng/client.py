"""

Cozmo protocol client and high-level API.

"""

import io
import json
import time
from collections.abc import Callable, Sequence
from threading import Event, Lock
from typing import Any

import numpy as np
from PIL import Image

from pycozmo_ng.animation import anim_controller
from pycozmo_ng.logger import logger, logger_robot
from pycozmo_ng.navigation.path_executor import PathExecutor
from pycozmo_ng.protocol import protocol_declaration, protocol_encoder
from pycozmo_ng.sound import audio, tts
from pycozmo_ng.transport import conn
from pycozmo_ng.util import filter, object, robot, robot_debug
from pycozmo_ng.vision import camera, image_encoder

from . import (
    event,
    exception,
    lights,
    util,
)

__all__ = [
    "Client",
]


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
        self.anim_controller.enable_procedural_face(auto_initialize and enable_animations and enable_procedural_face)

        self.serial_number_head = None
        self.robot_fw_sig: dict[str, Any] | None = None
        self.serial_number = None
        self.body_hw_version = None
        self.body_color: protocol_encoder.BodyColor | None = None
        # Robot state
        # Heading in X-Y plane.
        self.pose_frame_id = 0
        self.pose = util.Pose(0.0, 0.0, 0.0, angle_z=util.Angle(radians=0.0), origin_id=1)
        self.pose_pitch = util.Angle(radians=0.0)
        # Firmware path execution — the path state machine (settle, event-id
        # allocation, ClearPath → Append… → ExecutePath, build-and-wait) and
        # the lock that serializes it. This object implements the narrow
        # PathRobot port this class exposes (send / stop_all_motors / pose /
        # wheel speeds / handler registry); go_to_pose() and
        # execute_polyline() below are its delegations.
        self.paths = PathExecutor(self)
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
        # Rolling-shutter correctors read the frame's id to stamp IMU samples
        self.last_image_id = None
        # Object state
        self.available_objects: dict[int, object.Object] = {}
        self.connected_objects: dict[int, dict[str, Any]] = {}
        # Filters
        self.packet_type_filter = filter.Filter()
        self.packet_type_filter.deny_ids({protocol_declaration.PacketType.PING.value})
        self.packet_id_filter = filter.Filter()
        self._reset_partial_state()
        self._tts = tts.TTS(rate=tts_rate)

    def start(self) -> None:
        logger.debug("Starting client...")
        self.add_handler(protocol_encoder.HardwareInfo, self._on_hardware_info)
        self.add_handler(protocol_encoder.FirmwareSignature, self._on_firmware_signature)
        self.add_handler(protocol_encoder.BodyInfo, self._on_body_info)
        self.add_handler(protocol_encoder.ImageChunk, self._on_image_chunk)
        self.add_handler(protocol_encoder.RobotState, self._on_robot_state)
        self.add_handler(protocol_encoder.AnimationState, self._on_animation_state)
        self.add_handler(protocol_encoder.ObjectAvailable, self._on_object_available)
        self.add_handler(protocol_encoder.ObjectConnectionState, self._on_object_connection_state)
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
        sig: dict[str, Any] = json.loads(pkt.signature)
        self.robot_fw_sig = sig
        logger.info("Firmware version %s.", sig["version"])
        if sig.get("build") == "FACTORY":
            logger.warning("Factory/recovery firmware detected. Functionality is degraded.")
        elif sig["version"] < protocol_declaration.FIRMWARE_VERSION:
            logger.warning(
                f"Old firmware detected. PyCozmo-NG works best with v{protocol_declaration.FIRMWARE_VERSION}. Functionality may be degraded."
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
                    "Timed out waiting for factory_ids: {}".format(", ".join(f"0x{f:08x}" for f in remaining))
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
            self._partial_image_encoding = protocol_encoder.ImageEncoding(pkt.image_encoding)
            self._partial_image_resolution = protocol_encoder.ImageResolution(pkt.image_resolution)

            image_resolution = protocol_encoder.ImageResolution(pkt.image_resolution)
            width, height = camera.RESOLUTIONS[image_resolution]
            max_size = width * height * 3  # 3 bytes per pixel (RGB)
            self._partial_data = np.empty(max_size, dtype=np.uint8)

        if pkt.chunk_id != (self._last_chunk_id + 1) or pkt.image_id != self._partial_image_id:
            logger.debug(
                "Image missing chunks - discarding (last_chunk_id=%d partial_image_id=%s).",
                self._last_chunk_id,
                self._partial_image_id,
            )
            self._discard_partial_image()
            return

        offset = self._partial_size
        self._partial_data[offset : offset + len(pkt.data)] = np.frombuffer(pkt.data, dtype=np.uint8)
        self._partial_size += len(pkt.data)
        self._last_chunk_id = pkt.chunk_id

        if pkt.chunk_id == pkt.image_chunk_count - 1:
            self._process_completed_image()
            self._reset_partial_state()

    def _image_timestamp(self) -> int:
        """Timestamp of the just-completed image.

        The engine reads the image timestamp straight from
        ``ImageChunk.frame_timestamp`` — ``EncodedImage::AddChunk`` copies
        that wire field (the first u32 of 0xF2) into ``EncodedImage`` when
        the last chunk arrives, and ``RollingShutterCorrector`` anchors on
        it. Firmware 2458 leaves the field at 0, so that anchor would never
        advance and every frame's sample window collapses. Fall back to a
        millisecond monotonic host clock at frame completion: the relative
        per-band offsets the corrector derives stay valid, and consecutive
        frames keep rising anchors.
        """
        timestamp = self._partial_image_timestamp
        if timestamp:
            return timestamp
        return time.monotonic_ns() // 1_000_000

    def _discard_partial_image(self) -> None:
        """Discard an incomplete camera frame (e.g. from AP frame loss)."""
        self._reset_partial_state()
        self._partial_invalid = True

    def _process_completed_image(self):
        data = self._partial_data[0 : self._partial_size]

        # The first byte of the image is whether or not it is in color
        is_color_image = data[0] != 0

        if self._partial_image_encoding == protocol_encoder.ImageEncoding.JPEGMinimizedGray:
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
        self.last_image_timestamp = self._image_timestamp()
        self.last_image_id = self._partial_image_id

        self.dispatch(event.EvtNewRawCameraImage, self, image)

    @property
    def robot_on_charger(self) -> bool:
        """Whether the robot reports sitting on the charger.

        Read live from the status bits rather than from charger-change
        events: a robot already docked when the connection came up
        produces no change event at all, so an event-mirrored flag would
        stay false for the whole session.
        """
        return bool(self.robot_status & robot.RobotStatusFlag.IS_ON_CHARGER)

    @property
    def robot_charging(self) -> bool:
        """Whether the robot reports actively charging (live status bits)."""
        return bool(self.robot_status & robot.RobotStatusFlag.IS_CHARGING)

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
        # NB: "lift_height_mm" is misnamed — the firmware sends the lift
        # arm angle in radians; convert through the lift geometry so all
        # consumers (sensors API, telemetry, HUD) read real millimetres.
        self.lift_position = robot.lift_position_from_raw(pkt.lift_height_mm)
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

    def _on_object_connection_state(self, cli, pkt: protocol_encoder.ObjectConnectionState):
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

    def send(self, pkt: protocol_encoder.Packet) -> None:
        """Send a raw protocol packet to the robot."""
        self.conn.send(pkt)

    def go_to_pose(
        self,
        pose: util.Pose,
        relative_to_robot: bool = False,
        timeout_s: float | None = 30.0,
        position_tol_mm: float = 15.0,
        angle_tol_rad: float = 0.15,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """Move to a specific pose (position and orientation).

        A delegation: :class:`~pycozmo_ng.navigation.path_executor.PathExecutor`
        owns the firmware path state machine (settle, event-id allocation,
        build-and-wait, pose-poll fallback) and the lock that serializes it
        with :meth:`execute_polyline`. See its ``go_to_pose`` for the full
        contract — settle-first, event *or* pose-proximity completion, and
        ``should_interrupt`` for motion jobs.
        """
        return self.paths.go_to_pose(
            pose,
            relative_to_robot=relative_to_robot,
            timeout_s=timeout_s,
            position_tol_mm=position_tol_mm,
            angle_tol_rad=angle_tol_rad,
            should_interrupt=should_interrupt,
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

        A delegation: :class:`~pycozmo_ng.navigation.path_executor.PathExecutor`
        owns the firmware path state machine (settle, event-id allocation,
        buffer chunking, build-and-wait) and the lock that serializes it
        with :meth:`go_to_pose`. See its ``execute_polyline`` for the full
        contract — batching within the firmware buffer, the shared deadline,
        and ``should_interrupt`` for cancellation and replanning.
        """
        return self.paths.execute_polyline(
            points,
            speed_mmps=speed_mmps,
            accel_mmps2=accel_mmps2,
            decel_mmps2=decel_mmps2,
            final_heading_rad=final_heading_rad,
            timeout_s=timeout_s,
            position_tol_mm=position_tol_mm,
            should_interrupt=should_interrupt,
        )

    def set_backpack_lights(self, left_light, front_light, center_light, rear_light, right_light) -> None:
        pkt: protocol_encoder.Packet = protocol_encoder.LightStateCenter(states=(front_light, center_light, rear_light))
        self.conn.send(pkt)
        pkt = protocol_encoder.LightStateSide(states=(left_light, right_light))
        self.conn.send(pkt)

    def set_center_backpack_lights(self, light) -> None:
        self.set_backpack_lights(lights.off_light, light, light, light, lights.off_light)

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
        resolution: protocol_encoder.ImageResolution | None = None,
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
        image_resolution = resolution if resolution is not None else protocol_encoder.ImageResolution.QVGA
        pkt: protocol_encoder.Packet = protocol_encoder.EnableCamera(
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

    def display_image(self, im: Image.Image, duration: float | None = None) -> None:
        encoder = image_encoder.ImageEncoder(im)
        buf = bytes(encoder.encode())
        pkt = protocol_encoder.DisplayImage(image=buf)
        self.anim_controller.display_image(pkt)
        if duration is not None:
            time.sleep(duration)
            self.clear_screen()

    def set_volume(self, level: int) -> None:
        """Set audio output volume to a level in the range 0-65535."""
        pkt = protocol_encoder.SetRobotVolume(level=level)
        self.conn.send(pkt)

    def play_audio(self, fspec: str) -> None:
        """Load a WAVE file and queue it on the robot's audio pipeline.

        Args:
            fspec: Path to a 16-bit 22050/48000 Hz WAVE file.
        """
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
