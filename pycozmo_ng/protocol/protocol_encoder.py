"""

Cozmo protocol packet encoder classes, based on protocol version 2381.

Generated from protocol_declaration.py by protocol_generator.py .

Do not modify.

Exception: the classes for packets whose 2458 wire shape was recovered from the
engine binary (`SetPropSlot`, `TurnToRecordedHeading`, `ShutdownRobot`) were
edited by hand, because regenerating this file produces a large spurious diff —
it is not byte-reproducible from the generator.  Those classes carry a comment
pointing at not_used/docs/protocol.md.

"""

import enum

from .protocol_ast import PacketType
from .protocol_base import Packet, Struct
from .protocol_utils import (
    BinaryReader,
    BinaryWriter,
    get_object_farray_size,
    get_string_size,
    get_varray_size,
    validate_bool,
    validate_farray,
    validate_float,
    validate_integer,
    validate_object,
    validate_string,
    validate_varray,
)


class BodyColor(enum.Enum):
    UNKNOWN = -1
    WHITE_v10 = 0
    RESERVED = 1
    # White.
    WHITE_v15 = 2
    # Collectors edition, liquid metal.
    CE_LM_v15 = 3
    # Limited edition, blue.
    LE_BL_v16 = 4
    # Development unit.
    DEV = 5


class NvEntryTag(enum.Enum):
    NVEntry_Invalid = 0xFFFFFFFF
    NVEntry_GameSkillLevels = 0x180000
    NVEntry_OnboardingData = 0x181000
    NVEntry_GameUnlocks = 0x182000
    NVEntry_FaceEnrollData = 0x183000
    NVEntry_FaceAlbumData = 0x184000
    NVEntry_NurtureGameData = 0x194000
    NVEntry_InventoryData = 0x195000
    NVEntry_LabAssignments = 0x196000
    NVEntry_SavedCubeIDs = 0x197000
    NVEntry_NEXT_SLOT = 0x198000
    NVEntry_FACTORY_RESERVED1 = 0x1C0000
    NVEntry_FACTORY_RESERVED2 = 0x1DE000
    NVEntry_BirthCertificate = 0x80000000
    NVEntry_CameraCalib = 0x80000001
    NVEntry_ToolCodeInfo = 0x80000002
    NVEntry_CalibPose = 0x80000003
    NVEntry_CalibMetaInfo = 0x80000004
    NVEntry_ObservedCubePose = 0x80000005
    NVEntry_IMUInfo = 0x80000006
    NVEntry_CliffValOnDrop = 0x80000007
    NVEntry_CliffValOnGround = 0x80000008
    NVEntry_PlaypenTestResults = 0x80000010
    NVEntry_FactoryLock = 0x80000011
    NVEntry_VersionMagic = 0x80000012
    NVEntry_CalibImage1 = 0x80010000
    NVEntry_CalibImage2 = 0x80020000
    NVEntry_CalibImage3 = 0x80030000
    NVEntry_CalibImage4 = 0x80040000
    NVEntry_CalibImage5 = 0x80050000
    NVEntry_CalibImage6 = 0x80060000
    NVEntry_ToolCodeImageLeft = 0x80100000
    NVEntry_ToolCodeImageRight = 0x80110000
    NVEntry_PrePlaypenResults = 0xC0000000
    NVEntry_PrePlaypenCentroids = 0xC0000001
    NVEntry_IMUAverages = 0xC0000004
    NVEntry_FactoryBaseTag = 0xDE000
    NVEntry_FactoryBaseTagWithBCOffset = 0xDE030


class NvOperation(enum.Enum):
    NVOP_READ = 0
    NVOP_WRITE = 1
    NVOP_ERASE = 2
    NVOP_WIPEALL = 3


class NvResult(enum.Enum):
    NV_OKAY = 0
    NV_SCHEDULED = 1
    NV_NO_DO = 2
    NV_MORE = 3
    NV_UNKNOWN_4 = 4
    NV_UNKNOWN_5 = 5
    NV_UNKNOWN_6 = 6
    NV_UNKNOWN_7 = 7
    NV_UNKNOWN_8 = 8
    NV_NOT_FOUND = -1
    NV_NO_ROOM = -2
    NV_ERROR = -3
    NV_TIMEOUT = -4
    NV_BUSY = -5
    NV_BAD_ARGS = -6
    NV_NO_MEM = -7
    NV_LOOP = -8
    NV_CORRUPT = -9


class UpAxis(enum.Enum):
    XNegative = 0
    XPositive = 1
    YNegative = 2
    YPositive = 3
    ZNegative = 4
    ZPositive = 5
    NumAxes = 6
    UnknownAxis = 7


class ObjectType(enum.Enum):
    InvalidObject = -1
    UnknownObject = 0
    Block_LIGHTCUBE1 = 1
    Block_LIGHTCUBE2 = 2
    Block_LIGHTCUBE3 = 3
    Block_LIGHTCUBE_GHOST = 4
    FlatMat_GEARS_4x4 = 5
    FlatMat_LETTERS_4x4 = 6
    FlatMat_ANKI_LOGO_8BIT = 7
    FlatMat_LAVA_PLAYTEST = 8
    Platform_LARGE = 9
    Bridge_LONG = 10
    Bridge_SHORT = 11
    Ramp_Basic = 12
    Charger_Basic = 13
    ProxObstacle = 14
    CliffDetection = 15
    CollisionObstacle = 16
    CustomType00 = 17
    CustomType01 = 18
    CustomType02 = 19
    CustomType03 = 20
    CustomType04 = 21
    CustomType05 = 22
    CustomType06 = 23
    CustomType07 = 24
    CustomType08 = 25
    CustomType09 = 26
    CustomType10 = 27
    CustomType11 = 28
    CustomType12 = 29
    CustomType13 = 30
    CustomType14 = 31
    CustomType15 = 32
    CustomType16 = 33
    CustomType17 = 34
    CustomType18 = 35
    CustomType19 = 36
    CustomFixedObstacle = 37


class ImageEncoding(enum.Enum):
    NoneImageEncoding = 0
    RawGray = 1
    RawRGB = 2
    YUYV = 3
    BAYER = 4
    JPEGGray = 5
    JPEGColor = 6
    JPEGColorHalfWidth = 7
    JPEGMinimizedGray = 8
    JPEGMinimizedColor = 9


class ImageResolution(enum.Enum):
    VerificationSnapshot = 0
    QQQQVGA = 1
    QQQVGA = 2
    QQVGA = 3
    # 320x240
    QVGA = 4
    CVGA = 5
    # 640x480
    VGA = 6
    SVGA = 7
    XGA = 8
    SXGA = 9
    UXGA = 10
    QXGA = 11
    QUXGA = 12
    ImageResolutionCount = 13
    ImageResolutionNone = 14


class ImageSendMode(enum.Enum):
    Off = 0
    Stream = 1
    SingleShot = 2


class MotorID(enum.Enum):
    MOTOR_LEFT_WHEEL = 0
    MOTOR_RIGHT_WHEEL = 1
    MOTOR_LIFT = 2
    MOTOR_HEAD = 3


class PathEventType(enum.Enum):
    PATH_STARTED = 0
    PATH_INTERRUPTED = 1
    PATH_COMPLETED = 2


class LightState(Struct):
    __slots__ = (
        "_off_color",  # uint16
        "_off_frames",  # uint8
        "_offset",  # int16
        "_on_color",  # uint16
        "_on_frames",  # uint8
        "_transition_off_frames",  # uint8
        "_transition_on_frames",  # uint8
    )

    def __init__(
        self,
        on_color=0,
        off_color=0,
        on_frames=0,
        off_frames=0,
        transition_on_frames=0,
        transition_off_frames=0,
        offset=0,
    ):
        self.on_color = on_color
        self.off_color = off_color
        self.on_frames = on_frames
        self.off_frames = off_frames
        self.transition_on_frames = transition_on_frames
        self.transition_off_frames = transition_off_frames
        self.offset = offset

    @property
    def on_color(self):
        return self._on_color

    @on_color.setter
    def on_color(self, value):
        self._on_color = validate_integer("on_color", value, 0, 65535)

    @property
    def off_color(self):
        return self._off_color

    @off_color.setter
    def off_color(self, value):
        self._off_color = validate_integer("off_color", value, 0, 65535)

    @property
    def on_frames(self):
        return self._on_frames

    @on_frames.setter
    def on_frames(self, value):
        self._on_frames = validate_integer("on_frames", value, 0, 255)

    @property
    def off_frames(self):
        return self._off_frames

    @off_frames.setter
    def off_frames(self, value):
        self._off_frames = validate_integer("off_frames", value, 0, 255)

    @property
    def transition_on_frames(self):
        return self._transition_on_frames

    @transition_on_frames.setter
    def transition_on_frames(self, value):
        self._transition_on_frames = validate_integer("transition_on_frames", value, 0, 255)

    @property
    def transition_off_frames(self):
        return self._transition_off_frames

    @transition_off_frames.setter
    def transition_off_frames(self, value):
        self._transition_off_frames = validate_integer("transition_off_frames", value, 0, 255)

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = validate_integer("offset", value, -32768, 32767)

    def __len__(self):
        return 2 + 2 + 1 + 1 + 1 + 1 + 2

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"on_color={self._on_color}, "
            f"off_color={self._off_color}, "
            f"on_frames={self._on_frames}, "
            f"off_frames={self._off_frames}, "
            f"transition_on_frames={self._transition_on_frames}, "
            f"transition_off_frames={self._transition_off_frames}, "
            f"offset={self._offset})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._on_color, "H")
        writer.write(self._off_color, "H")
        writer.write(self._on_frames, "B")
        writer.write(self._off_frames, "B")
        writer.write(self._transition_on_frames, "B")
        writer.write(self._transition_off_frames, "B")
        writer.write(self._offset, "h")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        on_color = reader.read("H")
        off_color = reader.read("H")
        on_frames = reader.read("B")
        off_frames = reader.read("B")
        transition_on_frames = reader.read("B")
        transition_off_frames = reader.read("B")
        offset = reader.read("h")
        return cls(
            on_color=on_color,
            off_color=off_color,
            on_frames=on_frames,
            off_frames=off_frames,
            transition_on_frames=transition_on_frames,
            transition_off_frames=transition_off_frames,
            offset=offset,
        )


class PathSegmentSpeed(Struct):
    __slots__ = (
        "_accel_mmps2",  # float
        "_decel_mmps2",  # float
        "_speed_mmps",  # float
    )

    def __init__(self, speed_mmps=0.0, accel_mmps2=0.0, decel_mmps2=0.0):
        # Speed in millimeters per second.
        self.speed_mmps = speed_mmps
        # Acceleration in millimeters per second squared.
        self.accel_mmps2 = accel_mmps2
        # Deceleration in millimeters per second squared.
        self.decel_mmps2 = decel_mmps2

    @property
    def speed_mmps(self):
        return self._speed_mmps

    @speed_mmps.setter
    def speed_mmps(self, value):
        self._speed_mmps = validate_float("speed_mmps", value)

    @property
    def accel_mmps2(self):
        return self._accel_mmps2

    @accel_mmps2.setter
    def accel_mmps2(self, value):
        self._accel_mmps2 = validate_float("accel_mmps2", value)

    @property
    def decel_mmps2(self):
        return self._decel_mmps2

    @decel_mmps2.setter
    def decel_mmps2(self, value):
        self._decel_mmps2 = validate_float("decel_mmps2", value)

    def __len__(self):
        return 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"speed_mmps={self._speed_mmps}, "
            f"accel_mmps2={self._accel_mmps2}, "
            f"decel_mmps2={self._decel_mmps2})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._speed_mmps, "f")
        writer.write(self._accel_mmps2, "f")
        writer.write(self._decel_mmps2, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        speed_mmps = reader.read("f")
        accel_mmps2 = reader.read("f")
        decel_mmps2 = reader.read("f")
        return cls(speed_mmps=speed_mmps, accel_mmps2=accel_mmps2, decel_mmps2=decel_mmps2)


class Connect(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.CONNECT, packet_id=None)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class Disconnect(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.DISCONNECT, packet_id=None)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class Ping(Packet):
    __slots__ = (
        "_counter",  # uint32
        "_last",  # uint32
        "_time_sent_ms",  # double
        "_unknown",  # uint8
    )

    def __init__(self, time_sent_ms=0.0, counter=0, last=0, unknown=0):
        super().__init__(PacketType.PING, packet_id=None)
        self.time_sent_ms = time_sent_ms
        self.counter = counter
        self.last = last
        self.unknown = unknown

    @property
    def time_sent_ms(self):
        return self._time_sent_ms

    @time_sent_ms.setter
    def time_sent_ms(self, value):
        self._time_sent_ms = validate_float("time_sent_ms", value)

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, value):
        self._counter = validate_integer("counter", value, 0, 4294967295)

    @property
    def last(self):
        return self._last

    @last.setter
    def last(self, value):
        self._last = validate_integer("last", value, 0, 4294967295)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 255)

    def __len__(self):
        return 8 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"time_sent_ms={self._time_sent_ms}, "
            f"counter={self._counter}, "
            f"last={self._last}, "
            f"unknown={self._unknown})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._time_sent_ms, "d")
        writer.write(self._counter, "L")
        writer.write(self._last, "L")
        writer.write(self._unknown, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        time_sent_ms = reader.read("d")
        counter = reader.read("L")
        last = reader.read("L")
        unknown = reader.read("B")
        return cls(time_sent_ms=time_sent_ms, counter=counter, last=last, unknown=unknown)


class Keyframe(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.KEYFRAME, packet_id=None)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class LightStateCenter(Packet):
    __slots__ = (
        "_states",  # LightState[3]
        "_unknown",  # uint8
    )

    def __init__(self, states=(), unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0x03)
        # Top, middle, and bottom light state.
        self.states = states
        self.unknown = unknown

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        self._states = validate_farray(
            "states",
            value,
            3,
            lambda name, value_inner: validate_object(name, value_inner, LightState),
        )

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 255)

    def __len__(self):
        return get_object_farray_size(self._states, 3) + 1

    def __repr__(self):
        return f"{type(self).__name__}(states={self._states}, unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_object_farray(self._states, 3)
        writer.write(self._unknown, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        states = reader.read_object_farray(LightState.from_reader, 3)
        unknown = reader.read("B")
        return cls(states=states, unknown=unknown)


class CubeLights(Packet):
    __slots__ = (
        "_states",  # LightState[4]
    )

    def __init__(self, states=()):
        super().__init__(PacketType.COMMAND, packet_id=0x04)
        self.states = states

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        self._states = validate_farray(
            "states",
            value,
            4,
            lambda name, value_inner: validate_object(name, value_inner, LightState),
        )

    def __len__(self):
        return get_object_farray_size(self._states, 4)

    def __repr__(self):
        return f"{type(self).__name__}(states={self._states})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_object_farray(self._states, 4)

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        states = reader.read_object_farray(LightState.from_reader, 4)
        return cls(states=states)


class SetPropSlot(Packet):
    __slots__ = (
        "_factory_id",  # uint32
        "_slot",  # uint8
    )

    def __init__(self, factory_id=0, slot=0):
        super().__init__(PacketType.COMMAND, packet_id=0x05)
        self.factory_id = factory_id
        self.slot = slot

    @property
    def factory_id(self):
        return self._factory_id

    @factory_id.setter
    def factory_id(self, value):
        self._factory_id = validate_integer("factory_id", value, 0, 4294967295)

    @property
    def slot(self):
        return self._slot

    @slot.setter
    def slot(self, value):
        self._slot = validate_integer("slot", value, 0, 255)

    def __len__(self):
        return 4 + 1

    def __repr__(self):
        return f"{type(self).__name__}(factory_id={self._factory_id}, slot={self._slot})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._factory_id, "L")
        writer.write(self._slot, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        factory_id = reader.read("L")
        slot = reader.read("B")
        return cls(factory_id=factory_id, slot=slot)


class StreamObjectAccel(Packet):
    __slots__ = (
        "_enable",  # bool
        "_object_id",  # uint32
    )

    def __init__(self, object_id=0, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0x08)
        self.object_id = object_id
        self.enable = enable

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 4 + 1

    def __repr__(self):
        return f"{type(self).__name__}(object_id={self._object_id}, enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._object_id, "L")
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        object_id = reader.read("L")
        enable = bool(reader.read("b"))
        return cls(object_id=object_id, enable=enable)


class SetAccessoryDiscovery(Packet):
    __slots__ = (
        "_enable",  # bool
    )

    def __init__(self, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0x0A)
        self.enable = enable

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        enable = bool(reader.read("b"))
        return cls(enable=enable)


class SetHeadLight(Packet):
    __slots__ = (
        "_enable",  # bool
    )

    def __init__(self, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0x0B)
        self.enable = enable

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        enable = bool(reader.read("b"))
        return cls(enable=enable)


class CubeId(Packet):
    __slots__ = (
        "_object_id",  # uint32
        "_rotation_period_frames",  # uint8
    )

    def __init__(self, object_id=0, rotation_period_frames=0):
        super().__init__(PacketType.COMMAND, packet_id=0x10)
        self.object_id = object_id
        self.rotation_period_frames = rotation_period_frames

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def rotation_period_frames(self):
        return self._rotation_period_frames

    @rotation_period_frames.setter
    def rotation_period_frames(self, value):
        self._rotation_period_frames = validate_integer("rotation_period_frames", value, 0, 255)

    def __len__(self):
        return 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}(object_id={self._object_id}, rotation_period_frames={self._rotation_period_frames})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._object_id, "L")
        writer.write(self._rotation_period_frames, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        object_id = reader.read("L")
        rotation_period_frames = reader.read("B")
        return cls(object_id=object_id, rotation_period_frames=rotation_period_frames)


class LightStateSide(Packet):
    __slots__ = (
        "_states",  # LightState[2]
        "_unknown",  # uint8
    )

    def __init__(self, states=(), unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0x11)
        # Left and right light state.
        self.states = states
        self.unknown = unknown

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        self._states = validate_farray(
            "states",
            value,
            2,
            lambda name, value_inner: validate_object(name, value_inner, LightState),
        )

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 255)

    def __len__(self):
        return get_object_farray_size(self._states, 2) + 1

    def __repr__(self):
        return f"{type(self).__name__}(states={self._states}, unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_object_farray(self._states, 2)
        writer.write(self._unknown, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        states = reader.read_object_farray(LightState.from_reader, 2)
        unknown = reader.read("B")
        return cls(states=states, unknown=unknown)


class Enable(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x25)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class DriveWheels(Packet):
    __slots__ = (
        "_lwheel_accel_mmps2",  # float
        "_lwheel_speed_mmps",  # float
        "_rwheel_accel_mmps2",  # float
        "_rwheel_speed_mmps",  # float
    )

    def __init__(
        self,
        lwheel_speed_mmps=0.0,
        rwheel_speed_mmps=0.0,
        lwheel_accel_mmps2=0.0,
        rwheel_accel_mmps2=0.0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x32)
        self.lwheel_speed_mmps = lwheel_speed_mmps
        self.rwheel_speed_mmps = rwheel_speed_mmps
        self.lwheel_accel_mmps2 = lwheel_accel_mmps2
        self.rwheel_accel_mmps2 = rwheel_accel_mmps2

    @property
    def lwheel_speed_mmps(self):
        return self._lwheel_speed_mmps

    @lwheel_speed_mmps.setter
    def lwheel_speed_mmps(self, value):
        self._lwheel_speed_mmps = validate_float("lwheel_speed_mmps", value)

    @property
    def rwheel_speed_mmps(self):
        return self._rwheel_speed_mmps

    @rwheel_speed_mmps.setter
    def rwheel_speed_mmps(self, value):
        self._rwheel_speed_mmps = validate_float("rwheel_speed_mmps", value)

    @property
    def lwheel_accel_mmps2(self):
        return self._lwheel_accel_mmps2

    @lwheel_accel_mmps2.setter
    def lwheel_accel_mmps2(self, value):
        self._lwheel_accel_mmps2 = validate_float("lwheel_accel_mmps2", value)

    @property
    def rwheel_accel_mmps2(self):
        return self._rwheel_accel_mmps2

    @rwheel_accel_mmps2.setter
    def rwheel_accel_mmps2(self, value):
        self._rwheel_accel_mmps2 = validate_float("rwheel_accel_mmps2", value)

    def __len__(self):
        return 4 + 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"lwheel_speed_mmps={self._lwheel_speed_mmps}, "
            f"rwheel_speed_mmps={self._rwheel_speed_mmps}, "
            f"lwheel_accel_mmps2={self._lwheel_accel_mmps2}, "
            f"rwheel_accel_mmps2={self._rwheel_accel_mmps2})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._lwheel_speed_mmps, "f")
        writer.write(self._rwheel_speed_mmps, "f")
        writer.write(self._lwheel_accel_mmps2, "f")
        writer.write(self._rwheel_accel_mmps2, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        lwheel_speed_mmps = reader.read("f")
        rwheel_speed_mmps = reader.read("f")
        lwheel_accel_mmps2 = reader.read("f")
        rwheel_accel_mmps2 = reader.read("f")
        return cls(
            lwheel_speed_mmps=lwheel_speed_mmps,
            rwheel_speed_mmps=rwheel_speed_mmps,
            lwheel_accel_mmps2=lwheel_accel_mmps2,
            rwheel_accel_mmps2=rwheel_accel_mmps2,
        )


class TurnInPlaceAtSpeed(Packet):
    __slots__ = (
        "_direction",  # int16
        "_wheel_accel_mmps2",  # float
        "_wheel_speed_mmps",  # float
    )

    def __init__(self, wheel_speed_mmps=0.0, wheel_accel_mmps2=0.0, direction=0):
        super().__init__(PacketType.COMMAND, packet_id=0x33)
        self.wheel_speed_mmps = wheel_speed_mmps
        self.wheel_accel_mmps2 = wheel_accel_mmps2
        self.direction = direction

    @property
    def wheel_speed_mmps(self):
        return self._wheel_speed_mmps

    @wheel_speed_mmps.setter
    def wheel_speed_mmps(self, value):
        self._wheel_speed_mmps = validate_float("wheel_speed_mmps", value)

    @property
    def wheel_accel_mmps2(self):
        return self._wheel_accel_mmps2

    @wheel_accel_mmps2.setter
    def wheel_accel_mmps2(self, value):
        self._wheel_accel_mmps2 = validate_float("wheel_accel_mmps2", value)

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, value):
        self._direction = validate_integer("direction", value, -32768, 32767)

    def __len__(self):
        return 4 + 4 + 2

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"wheel_speed_mmps={self._wheel_speed_mmps}, "
            f"wheel_accel_mmps2={self._wheel_accel_mmps2}, "
            f"direction={self._direction})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._wheel_speed_mmps, "f")
        writer.write(self._wheel_accel_mmps2, "f")
        writer.write(self._direction, "h")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        wheel_speed_mmps = reader.read("f")
        wheel_accel_mmps2 = reader.read("f")
        direction = reader.read("h")
        return cls(
            wheel_speed_mmps=wheel_speed_mmps,
            wheel_accel_mmps2=wheel_accel_mmps2,
            direction=direction,
        )


class MoveLift(Packet):
    __slots__ = (
        "_speed_rad_per_sec",  # float
    )

    def __init__(self, speed_rad_per_sec=0.0):
        super().__init__(PacketType.COMMAND, packet_id=0x34)
        self.speed_rad_per_sec = speed_rad_per_sec

    @property
    def speed_rad_per_sec(self):
        return self._speed_rad_per_sec

    @speed_rad_per_sec.setter
    def speed_rad_per_sec(self, value):
        self._speed_rad_per_sec = validate_float("speed_rad_per_sec", value)

    def __len__(self):
        return 4

    def __repr__(self):
        return f"{type(self).__name__}(speed_rad_per_sec={self._speed_rad_per_sec})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._speed_rad_per_sec, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        speed_rad_per_sec = reader.read("f")
        return cls(speed_rad_per_sec=speed_rad_per_sec)


class MoveHead(Packet):
    __slots__ = (
        "_speed_rad_per_sec",  # float
    )

    def __init__(self, speed_rad_per_sec=0.0):
        super().__init__(PacketType.COMMAND, packet_id=0x35)
        self.speed_rad_per_sec = speed_rad_per_sec

    @property
    def speed_rad_per_sec(self):
        return self._speed_rad_per_sec

    @speed_rad_per_sec.setter
    def speed_rad_per_sec(self, value):
        self._speed_rad_per_sec = validate_float("speed_rad_per_sec", value)

    def __len__(self):
        return 4

    def __repr__(self):
        return f"{type(self).__name__}(speed_rad_per_sec={self._speed_rad_per_sec})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._speed_rad_per_sec, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        speed_rad_per_sec = reader.read("f")
        return cls(speed_rad_per_sec=speed_rad_per_sec)


class SetLiftHeight(Packet):
    __slots__ = (
        "_accel_rad_per_sec2",  # float
        "_action_id",  # uint8
        "_duration_sec",  # float
        "_height_mm",  # float
        "_max_speed_rad_per_sec",  # float
    )

    def __init__(
        self,
        height_mm=0.0,
        max_speed_rad_per_sec=3.0,
        accel_rad_per_sec2=20.0,
        duration_sec=0.0,
        action_id=0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x36)
        self.height_mm = height_mm
        self.max_speed_rad_per_sec = max_speed_rad_per_sec
        self.accel_rad_per_sec2 = accel_rad_per_sec2
        self.duration_sec = duration_sec
        # Not present in v2214 and older.
        self.action_id = action_id

    @property
    def height_mm(self):
        return self._height_mm

    @height_mm.setter
    def height_mm(self, value):
        self._height_mm = validate_float("height_mm", value)

    @property
    def max_speed_rad_per_sec(self):
        return self._max_speed_rad_per_sec

    @max_speed_rad_per_sec.setter
    def max_speed_rad_per_sec(self, value):
        self._max_speed_rad_per_sec = validate_float("max_speed_rad_per_sec", value)

    @property
    def accel_rad_per_sec2(self):
        return self._accel_rad_per_sec2

    @accel_rad_per_sec2.setter
    def accel_rad_per_sec2(self, value):
        self._accel_rad_per_sec2 = validate_float("accel_rad_per_sec2", value)

    @property
    def duration_sec(self):
        return self._duration_sec

    @duration_sec.setter
    def duration_sec(self, value):
        self._duration_sec = validate_float("duration_sec", value)

    @property
    def action_id(self):
        return self._action_id

    @action_id.setter
    def action_id(self, value):
        self._action_id = validate_integer("action_id", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"height_mm={self._height_mm}, "
            f"max_speed_rad_per_sec={self._max_speed_rad_per_sec}, "
            f"accel_rad_per_sec2={self._accel_rad_per_sec2}, "
            f"duration_sec={self._duration_sec}, "
            f"action_id={self._action_id})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._height_mm, "f")
        writer.write(self._max_speed_rad_per_sec, "f")
        writer.write(self._accel_rad_per_sec2, "f")
        writer.write(self._duration_sec, "f")
        writer.write(self._action_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        height_mm = reader.read("f")
        max_speed_rad_per_sec = reader.read("f")
        accel_rad_per_sec2 = reader.read("f")
        duration_sec = reader.read("f")
        action_id = reader.read("B")
        return cls(
            height_mm=height_mm,
            max_speed_rad_per_sec=max_speed_rad_per_sec,
            accel_rad_per_sec2=accel_rad_per_sec2,
            duration_sec=duration_sec,
            action_id=action_id,
        )


class SetHeadAngle(Packet):
    __slots__ = (
        "_accel_rad_per_sec2",  # float
        "_action_id",  # uint8
        "_angle_rad",  # float
        "_duration_sec",  # float
        "_max_speed_rad_per_sec",  # float
    )

    def __init__(
        self,
        angle_rad=0.0,
        max_speed_rad_per_sec=15.0,
        accel_rad_per_sec2=20.0,
        duration_sec=0.0,
        action_id=0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x37)
        self.angle_rad = angle_rad
        self.max_speed_rad_per_sec = max_speed_rad_per_sec
        self.accel_rad_per_sec2 = accel_rad_per_sec2
        self.duration_sec = duration_sec
        # Not present in v2214 and older.
        self.action_id = action_id

    @property
    def angle_rad(self):
        return self._angle_rad

    @angle_rad.setter
    def angle_rad(self, value):
        self._angle_rad = validate_float("angle_rad", value)

    @property
    def max_speed_rad_per_sec(self):
        return self._max_speed_rad_per_sec

    @max_speed_rad_per_sec.setter
    def max_speed_rad_per_sec(self, value):
        self._max_speed_rad_per_sec = validate_float("max_speed_rad_per_sec", value)

    @property
    def accel_rad_per_sec2(self):
        return self._accel_rad_per_sec2

    @accel_rad_per_sec2.setter
    def accel_rad_per_sec2(self, value):
        self._accel_rad_per_sec2 = validate_float("accel_rad_per_sec2", value)

    @property
    def duration_sec(self):
        return self._duration_sec

    @duration_sec.setter
    def duration_sec(self, value):
        self._duration_sec = validate_float("duration_sec", value)

    @property
    def action_id(self):
        return self._action_id

    @action_id.setter
    def action_id(self, value):
        self._action_id = validate_integer("action_id", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"angle_rad={self._angle_rad}, "
            f"max_speed_rad_per_sec={self._max_speed_rad_per_sec}, "
            f"accel_rad_per_sec2={self._accel_rad_per_sec2}, "
            f"duration_sec={self._duration_sec}, "
            f"action_id={self._action_id})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._angle_rad, "f")
        writer.write(self._max_speed_rad_per_sec, "f")
        writer.write(self._accel_rad_per_sec2, "f")
        writer.write(self._duration_sec, "f")
        writer.write(self._action_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        angle_rad = reader.read("f")
        max_speed_rad_per_sec = reader.read("f")
        accel_rad_per_sec2 = reader.read("f")
        duration_sec = reader.read("f")
        action_id = reader.read("B")
        return cls(
            angle_rad=angle_rad,
            max_speed_rad_per_sec=max_speed_rad_per_sec,
            accel_rad_per_sec2=accel_rad_per_sec2,
            duration_sec=duration_sec,
            action_id=action_id,
        )


class TurnInPlace(Packet):
    __slots__ = (
        "_accel_rad_per_sec2",  # float
        "_action_id",  # uint8
        "_angle_rad",  # float
        "_angle_tolerance_rad",  # float
        "_is_absolute",  # bool
        "_speed_rad_per_sec",  # float
        "_unknown4",  # uint8
        "_unknown5",  # uint8
    )

    def __init__(
        self,
        angle_rad=0.0,
        speed_rad_per_sec=0.0,
        accel_rad_per_sec2=0.0,
        angle_tolerance_rad=0.0,
        unknown4=0,
        unknown5=0,
        is_absolute=False,
        action_id=0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x39)
        self.angle_rad = angle_rad
        self.speed_rad_per_sec = speed_rad_per_sec
        self.accel_rad_per_sec2 = accel_rad_per_sec2
        self.angle_tolerance_rad = angle_tolerance_rad
        self.unknown4 = unknown4
        self.unknown5 = unknown5
        self.is_absolute = is_absolute
        self.action_id = action_id

    @property
    def angle_rad(self):
        return self._angle_rad

    @angle_rad.setter
    def angle_rad(self, value):
        self._angle_rad = validate_float("angle_rad", value)

    @property
    def speed_rad_per_sec(self):
        return self._speed_rad_per_sec

    @speed_rad_per_sec.setter
    def speed_rad_per_sec(self, value):
        self._speed_rad_per_sec = validate_float("speed_rad_per_sec", value)

    @property
    def accel_rad_per_sec2(self):
        return self._accel_rad_per_sec2

    @accel_rad_per_sec2.setter
    def accel_rad_per_sec2(self, value):
        self._accel_rad_per_sec2 = validate_float("accel_rad_per_sec2", value)

    @property
    def angle_tolerance_rad(self):
        return self._angle_tolerance_rad

    @angle_tolerance_rad.setter
    def angle_tolerance_rad(self, value):
        self._angle_tolerance_rad = validate_float("angle_tolerance_rad", value)

    @property
    def unknown4(self):
        return self._unknown4

    @unknown4.setter
    def unknown4(self, value):
        self._unknown4 = validate_integer("unknown4", value, 0, 255)

    @property
    def unknown5(self):
        return self._unknown5

    @unknown5.setter
    def unknown5(self, value):
        self._unknown5 = validate_integer("unknown5", value, 0, 255)

    @property
    def is_absolute(self):
        return self._is_absolute

    @is_absolute.setter
    def is_absolute(self, value):
        self._is_absolute = validate_bool("is_absolute", value)

    @property
    def action_id(self):
        return self._action_id

    @action_id.setter
    def action_id(self, value):
        self._action_id = validate_integer("action_id", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 1 + 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"angle_rad={self._angle_rad}, "
            f"speed_rad_per_sec={self._speed_rad_per_sec}, "
            f"accel_rad_per_sec2={self._accel_rad_per_sec2}, "
            f"angle_tolerance_rad={self._angle_tolerance_rad}, "
            f"unknown4={self._unknown4}, "
            f"unknown5={self._unknown5}, "
            f"is_absolute={self._is_absolute}, "
            f"action_id={self._action_id})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._angle_rad, "f")
        writer.write(self._speed_rad_per_sec, "f")
        writer.write(self._accel_rad_per_sec2, "f")
        writer.write(self._angle_tolerance_rad, "f")
        writer.write(self._unknown4, "B")
        writer.write(self._unknown5, "B")
        writer.write(int(self._is_absolute), "b")
        writer.write(self._action_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        angle_rad = reader.read("f")
        speed_rad_per_sec = reader.read("f")
        accel_rad_per_sec2 = reader.read("f")
        angle_tolerance_rad = reader.read("f")
        unknown4 = reader.read("B")
        unknown5 = reader.read("B")
        is_absolute = bool(reader.read("b"))
        action_id = reader.read("B")
        return cls(
            angle_rad=angle_rad,
            speed_rad_per_sec=speed_rad_per_sec,
            accel_rad_per_sec2=accel_rad_per_sec2,
            angle_tolerance_rad=angle_tolerance_rad,
            unknown4=unknown4,
            unknown5=unknown5,
            is_absolute=is_absolute,
            action_id=action_id,
        )


class StopAllMotors(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x3B)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class ClearPath(Packet):
    __slots__ = (
        "_unknown",  # uint16
    )

    def __init__(self, unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0x3C)
        self.unknown = unknown

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 65535)

    def __len__(self):
        return 2

    def __repr__(self):
        return f"{type(self).__name__}(unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._unknown, "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read("H")
        return cls(unknown=unknown)


class AppendPathSegLine(Packet):
    __slots__ = (
        "_accel_mmps2",  # float
        "_decel_mmps2",  # float
        "_from_x",  # float
        "_from_y",  # float
        "_speed_mmps",  # float
        "_to_x",  # float
        "_to_y",  # float
    )

    def __init__(
        self,
        from_x=0.0,
        from_y=0.0,
        to_x=0.0,
        to_y=0.0,
        speed_mmps=0.0,
        accel_mmps2=0.0,
        decel_mmps2=0.0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x3D)
        self.from_x = from_x
        self.from_y = from_y
        self.to_x = to_x
        self.to_y = to_y
        self.speed_mmps = speed_mmps
        self.accel_mmps2 = accel_mmps2
        self.decel_mmps2 = decel_mmps2

    @property
    def from_x(self):
        return self._from_x

    @from_x.setter
    def from_x(self, value):
        self._from_x = validate_float("from_x", value)

    @property
    def from_y(self):
        return self._from_y

    @from_y.setter
    def from_y(self, value):
        self._from_y = validate_float("from_y", value)

    @property
    def to_x(self):
        return self._to_x

    @to_x.setter
    def to_x(self, value):
        self._to_x = validate_float("to_x", value)

    @property
    def to_y(self):
        return self._to_y

    @to_y.setter
    def to_y(self, value):
        self._to_y = validate_float("to_y", value)

    @property
    def speed_mmps(self):
        return self._speed_mmps

    @speed_mmps.setter
    def speed_mmps(self, value):
        self._speed_mmps = validate_float("speed_mmps", value)

    @property
    def accel_mmps2(self):
        return self._accel_mmps2

    @accel_mmps2.setter
    def accel_mmps2(self, value):
        self._accel_mmps2 = validate_float("accel_mmps2", value)

    @property
    def decel_mmps2(self):
        return self._decel_mmps2

    @decel_mmps2.setter
    def decel_mmps2(self, value):
        self._decel_mmps2 = validate_float("decel_mmps2", value)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"from_x={self._from_x}, "
            f"from_y={self._from_y}, "
            f"to_x={self._to_x}, "
            f"to_y={self._to_y}, "
            f"speed_mmps={self._speed_mmps}, "
            f"accel_mmps2={self._accel_mmps2}, "
            f"decel_mmps2={self._decel_mmps2})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._from_x, "f")
        writer.write(self._from_y, "f")
        writer.write(self._to_x, "f")
        writer.write(self._to_y, "f")
        writer.write(self._speed_mmps, "f")
        writer.write(self._accel_mmps2, "f")
        writer.write(self._decel_mmps2, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        from_x = reader.read("f")
        from_y = reader.read("f")
        to_x = reader.read("f")
        to_y = reader.read("f")
        speed_mmps = reader.read("f")
        accel_mmps2 = reader.read("f")
        decel_mmps2 = reader.read("f")
        return cls(
            from_x=from_x,
            from_y=from_y,
            to_x=to_x,
            to_y=to_y,
            speed_mmps=speed_mmps,
            accel_mmps2=accel_mmps2,
            decel_mmps2=decel_mmps2,
        )


class AppendPathSegArc(Packet):
    __slots__ = (
        "_accel_mmps2",  # float
        "_center_x",  # float
        "_center_y",  # float
        "_decel_mmps2",  # float
        "_radius_mm",  # float
        "_speed_mmps",  # float
        "_start_angle_rad",  # float
        "_sweep_rad",  # float
    )

    def __init__(
        self,
        center_x=0.0,
        center_y=0.0,
        radius_mm=0.0,
        start_angle_rad=0.0,
        sweep_rad=0.0,
        speed_mmps=0.0,
        accel_mmps2=0.0,
        decel_mmps2=0.0,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x3E)
        self.center_x = center_x
        self.center_y = center_y
        self.radius_mm = radius_mm
        self.start_angle_rad = start_angle_rad
        self.sweep_rad = sweep_rad
        self.speed_mmps = speed_mmps
        self.accel_mmps2 = accel_mmps2
        self.decel_mmps2 = decel_mmps2

    @property
    def center_x(self):
        return self._center_x

    @center_x.setter
    def center_x(self, value):
        self._center_x = validate_float("center_x", value)

    @property
    def center_y(self):
        return self._center_y

    @center_y.setter
    def center_y(self, value):
        self._center_y = validate_float("center_y", value)

    @property
    def radius_mm(self):
        return self._radius_mm

    @radius_mm.setter
    def radius_mm(self, value):
        self._radius_mm = validate_float("radius_mm", value)

    @property
    def start_angle_rad(self):
        return self._start_angle_rad

    @start_angle_rad.setter
    def start_angle_rad(self, value):
        self._start_angle_rad = validate_float("start_angle_rad", value)

    @property
    def sweep_rad(self):
        return self._sweep_rad

    @sweep_rad.setter
    def sweep_rad(self, value):
        self._sweep_rad = validate_float("sweep_rad", value)

    @property
    def speed_mmps(self):
        return self._speed_mmps

    @speed_mmps.setter
    def speed_mmps(self, value):
        self._speed_mmps = validate_float("speed_mmps", value)

    @property
    def accel_mmps2(self):
        return self._accel_mmps2

    @accel_mmps2.setter
    def accel_mmps2(self, value):
        self._accel_mmps2 = validate_float("accel_mmps2", value)

    @property
    def decel_mmps2(self):
        return self._decel_mmps2

    @decel_mmps2.setter
    def decel_mmps2(self, value):
        self._decel_mmps2 = validate_float("decel_mmps2", value)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"center_x={self._center_x}, "
            f"center_y={self._center_y}, "
            f"radius_mm={self._radius_mm}, "
            f"start_angle_rad={self._start_angle_rad}, "
            f"sweep_rad={self._sweep_rad}, "
            f"speed_mmps={self._speed_mmps}, "
            f"accel_mmps2={self._accel_mmps2}, "
            f"decel_mmps2={self._decel_mmps2})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._center_x, "f")
        writer.write(self._center_y, "f")
        writer.write(self._radius_mm, "f")
        writer.write(self._start_angle_rad, "f")
        writer.write(self._sweep_rad, "f")
        writer.write(self._speed_mmps, "f")
        writer.write(self._accel_mmps2, "f")
        writer.write(self._decel_mmps2, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        center_x = reader.read("f")
        center_y = reader.read("f")
        radius_mm = reader.read("f")
        start_angle_rad = reader.read("f")
        sweep_rad = reader.read("f")
        speed_mmps = reader.read("f")
        accel_mmps2 = reader.read("f")
        decel_mmps2 = reader.read("f")
        return cls(
            center_x=center_x,
            center_y=center_y,
            radius_mm=radius_mm,
            start_angle_rad=start_angle_rad,
            sweep_rad=sweep_rad,
            speed_mmps=speed_mmps,
            accel_mmps2=accel_mmps2,
            decel_mmps2=decel_mmps2,
        )


class AppendPathSegPointTurn(Packet):
    __slots__ = (
        "_accel_mmps2",  # float
        "_angle_rad",  # float
        "_angle_tolerance_rad",  # float
        "_decel_mmps2",  # float
        "_speed_mmps",  # float
        "_unknown",  # bool
        "_x",  # float
        "_y",  # float
    )

    def __init__(
        self,
        x=0.0,
        y=0.0,
        angle_rad=0.0,
        angle_tolerance_rad=0.0,
        speed_mmps=0.0,
        accel_mmps2=0.0,
        decel_mmps2=0.0,
        unknown=False,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x3F)
        self.x = x
        self.y = y
        self.angle_rad = angle_rad
        self.angle_tolerance_rad = angle_tolerance_rad
        self.speed_mmps = speed_mmps
        self.accel_mmps2 = accel_mmps2
        self.decel_mmps2 = decel_mmps2
        self.unknown = unknown

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = validate_float("x", value)

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = validate_float("y", value)

    @property
    def angle_rad(self):
        return self._angle_rad

    @angle_rad.setter
    def angle_rad(self, value):
        self._angle_rad = validate_float("angle_rad", value)

    @property
    def angle_tolerance_rad(self):
        return self._angle_tolerance_rad

    @angle_tolerance_rad.setter
    def angle_tolerance_rad(self, value):
        self._angle_tolerance_rad = validate_float("angle_tolerance_rad", value)

    @property
    def speed_mmps(self):
        return self._speed_mmps

    @speed_mmps.setter
    def speed_mmps(self, value):
        self._speed_mmps = validate_float("speed_mmps", value)

    @property
    def accel_mmps2(self):
        return self._accel_mmps2

    @accel_mmps2.setter
    def accel_mmps2(self, value):
        self._accel_mmps2 = validate_float("accel_mmps2", value)

    @property
    def decel_mmps2(self):
        return self._decel_mmps2

    @decel_mmps2.setter
    def decel_mmps2(self, value):
        self._decel_mmps2 = validate_float("decel_mmps2", value)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_bool("unknown", value)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"x={self._x}, "
            f"y={self._y}, "
            f"angle_rad={self._angle_rad}, "
            f"angle_tolerance_rad={self._angle_tolerance_rad}, "
            f"speed_mmps={self._speed_mmps}, "
            f"accel_mmps2={self._accel_mmps2}, "
            f"decel_mmps2={self._decel_mmps2}, "
            f"unknown={self._unknown})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._x, "f")
        writer.write(self._y, "f")
        writer.write(self._angle_rad, "f")
        writer.write(self._angle_tolerance_rad, "f")
        writer.write(self._speed_mmps, "f")
        writer.write(self._accel_mmps2, "f")
        writer.write(self._decel_mmps2, "f")
        writer.write(int(self._unknown), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        x = reader.read("f")
        y = reader.read("f")
        angle_rad = reader.read("f")
        angle_tolerance_rad = reader.read("f")
        speed_mmps = reader.read("f")
        accel_mmps2 = reader.read("f")
        decel_mmps2 = reader.read("f")
        unknown = bool(reader.read("b"))
        return cls(
            x=x,
            y=y,
            angle_rad=angle_rad,
            angle_tolerance_rad=angle_tolerance_rad,
            speed_mmps=speed_mmps,
            accel_mmps2=accel_mmps2,
            decel_mmps2=decel_mmps2,
            unknown=unknown,
        )


class TrimPath(Packet):
    __slots__ = (
        "_head",  # uint8
        "_tail",  # uint8
    )

    def __init__(self, head=0, tail=0):
        super().__init__(PacketType.COMMAND, packet_id=0x40)
        self.head = head
        self.tail = tail

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = validate_integer("head", value, 0, 255)

    @property
    def tail(self):
        return self._tail

    @tail.setter
    def tail(self, value):
        self._tail = validate_integer("tail", value, 0, 255)

    def __len__(self):
        return 1 + 1

    def __repr__(self):
        return f"{type(self).__name__}(head={self._head}, tail={self._tail})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._head, "B")
        writer.write(self._tail, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        head = reader.read("B")
        tail = reader.read("B")
        return cls(head=head, tail=tail)


class ExecutePath(Packet):
    __slots__ = (
        "_event_id",  # uint16
        "_unknown",  # bool
    )

    def __init__(self, event_id=0, unknown=False):
        super().__init__(PacketType.COMMAND, packet_id=0x41)
        self.event_id = event_id
        self.unknown = unknown

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, value):
        self._event_id = validate_integer("event_id", value, 0, 65535)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_bool("unknown", value)

    def __len__(self):
        return 2 + 1

    def __repr__(self):
        return f"{type(self).__name__}(event_id={self._event_id}, unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._event_id, "H")
        writer.write(int(self._unknown), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        event_id = reader.read("H")
        unknown = bool(reader.read("b"))
        return cls(event_id=event_id, unknown=unknown)


class SetOrigin(Packet):
    __slots__ = (
        "_pose_frame_id",  # uint32
        "_pose_origin_id",  # uint32
        "_pose_x",  # float
        "_pose_y",  # float
        "_unknown0",  # uint32
        "_unknown5",  # uint32
    )

    def __init__(
        self,
        unknown0=0,
        pose_frame_id=0,
        pose_origin_id=1,
        pose_x=0.0,
        pose_y=0.0,
        unknown5=2147483648,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0x45)
        self.unknown0 = unknown0
        self.pose_frame_id = pose_frame_id
        self.pose_origin_id = pose_origin_id
        self.pose_x = pose_x
        self.pose_y = pose_y
        self.unknown5 = unknown5

    @property
    def unknown0(self):
        return self._unknown0

    @unknown0.setter
    def unknown0(self, value):
        self._unknown0 = validate_integer("unknown0", value, 0, 4294967295)

    @property
    def pose_frame_id(self):
        return self._pose_frame_id

    @pose_frame_id.setter
    def pose_frame_id(self, value):
        self._pose_frame_id = validate_integer("pose_frame_id", value, 0, 4294967295)

    @property
    def pose_origin_id(self):
        return self._pose_origin_id

    @pose_origin_id.setter
    def pose_origin_id(self, value):
        self._pose_origin_id = validate_integer("pose_origin_id", value, 0, 4294967295)

    @property
    def pose_x(self):
        return self._pose_x

    @pose_x.setter
    def pose_x(self, value):
        self._pose_x = validate_float("pose_x", value)

    @property
    def pose_y(self):
        return self._pose_y

    @pose_y.setter
    def pose_y(self, value):
        self._pose_y = validate_float("pose_y", value)

    @property
    def unknown5(self):
        return self._unknown5

    @unknown5.setter
    def unknown5(self, value):
        self._unknown5 = validate_integer("unknown5", value, 0, 4294967295)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"unknown0={self._unknown0}, "
            f"pose_frame_id={self._pose_frame_id}, "
            f"pose_origin_id={self._pose_origin_id}, "
            f"pose_x={self._pose_x}, "
            f"pose_y={self._pose_y}, "
            f"unknown5={self._unknown5})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._unknown0, "L")
        writer.write(self._pose_frame_id, "L")
        writer.write(self._pose_origin_id, "L")
        writer.write(self._pose_x, "f")
        writer.write(self._pose_y, "f")
        writer.write(self._unknown5, "L")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown0 = reader.read("L")
        pose_frame_id = reader.read("L")
        pose_origin_id = reader.read("L")
        pose_x = reader.read("f")
        pose_y = reader.read("f")
        unknown5 = reader.read("L")
        return cls(
            unknown0=unknown0,
            pose_frame_id=pose_frame_id,
            pose_origin_id=pose_origin_id,
            pose_x=pose_x,
            pose_y=pose_y,
            unknown5=unknown5,
        )


class SyncTime(Packet):
    __slots__ = (
        "_timestamp",  # uint32
        "_unknown",  # uint32
    )

    def __init__(self, timestamp=0, unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0x4B)
        self.timestamp = timestamp
        self.unknown = unknown

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 4294967295)

    def __len__(self):
        return 4 + 4

    def __repr__(self):
        return f"{type(self).__name__}(timestamp={self._timestamp}, unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._unknown, "L")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        unknown = reader.read("L")
        return cls(timestamp=timestamp, unknown=unknown)


class EnableCamera(Packet):
    __slots__ = (
        "_image_resolution",  # ImageResolution
        "_image_send_mode",  # ImageSendMode
    )

    def __init__(self, image_send_mode=1, image_resolution=4):
        super().__init__(PacketType.COMMAND, packet_id=0x4C)
        self.image_send_mode = ImageSendMode(image_send_mode)
        self.image_resolution = ImageResolution(image_resolution)

    @property
    def image_send_mode(self) -> ImageSendMode:
        return self._image_send_mode

    @image_send_mode.setter
    def image_send_mode(self, value: ImageSendMode) -> None:
        self._image_send_mode = value
        validate_integer("image_send_mode", value.value, -128, 127)

    @property
    def image_resolution(self) -> ImageResolution:
        return self._image_resolution

    @image_resolution.setter
    def image_resolution(self, value: ImageResolution) -> None:
        self._image_resolution = value
        validate_integer("image_resolution", value.value, -128, 127)

    def __len__(self):
        return 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}(image_send_mode={self._image_send_mode}, image_resolution={self._image_resolution})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._image_send_mode.value, "b")
        writer.write(self._image_resolution.value, "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        image_send_mode = reader.read("b")
        image_resolution = reader.read("b")
        return cls(image_send_mode=image_send_mode, image_resolution=image_resolution)


class SetCameraParams(Packet):
    __slots__ = (
        "_auto_exposure_enabled",  # bool
        "_exposure_ms",  # uint16
        "_gain",  # float
    )

    def __init__(self, gain=0.0, exposure_ms=0, auto_exposure_enabled=False):
        super().__init__(PacketType.COMMAND, packet_id=0x57)
        self.gain = gain
        self.exposure_ms = exposure_ms
        self.auto_exposure_enabled = auto_exposure_enabled

    @property
    def gain(self):
        return self._gain

    @gain.setter
    def gain(self, value):
        self._gain = validate_float("gain", value)

    @property
    def exposure_ms(self):
        return self._exposure_ms

    @exposure_ms.setter
    def exposure_ms(self, value):
        self._exposure_ms = validate_integer("exposure_ms", value, 0, 65535)

    @property
    def auto_exposure_enabled(self):
        return self._auto_exposure_enabled

    @auto_exposure_enabled.setter
    def auto_exposure_enabled(self, value):
        self._auto_exposure_enabled = validate_bool("auto_exposure_enabled", value)

    def __len__(self):
        return 4 + 2 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"gain={self._gain}, "
            f"exposure_ms={self._exposure_ms}, "
            f"auto_exposure_enabled={self._auto_exposure_enabled})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._gain, "f")
        writer.write(self._exposure_ms, "H")
        writer.write(int(self._auto_exposure_enabled), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        gain = reader.read("f")
        exposure_ms = reader.read("H")
        auto_exposure_enabled = bool(reader.read("b"))
        return cls(
            gain=gain,
            exposure_ms=exposure_ms,
            auto_exposure_enabled=auto_exposure_enabled,
        )


class StartMotorCalibration(Packet):
    __slots__ = (
        "_head",  # bool
        "_lift",  # bool
    )

    def __init__(self, head=False, lift=False):
        super().__init__(PacketType.COMMAND, packet_id=0x58)
        self.head = head
        self.lift = lift

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, value):
        self._head = validate_bool("head", value)

    @property
    def lift(self):
        return self._lift

    @lift.setter
    def lift(self, value):
        self._lift = validate_bool("lift", value)

    def __len__(self):
        return 1 + 1

    def __repr__(self):
        return f"{type(self).__name__}(head={self._head}, lift={self._lift})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._head), "b")
        writer.write(int(self._lift), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        head = bool(reader.read("b"))
        lift = bool(reader.read("b"))
        return cls(head=head, lift=lift)


class EnableStopOnCliff(Packet):
    __slots__ = (
        "_enable",  # bool
    )

    def __init__(self, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0x60)
        self.enable = enable

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        enable = bool(reader.read("b"))
        return cls(enable=enable)


class SetRobotVolume(Packet):
    __slots__ = (
        "_level",  # uint16
    )

    def __init__(self, level=0):
        super().__init__(PacketType.COMMAND, packet_id=0x64)
        self.level = level

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = validate_integer("level", value, 0, 65535)

    def __len__(self):
        return 2

    def __repr__(self):
        return f"{type(self).__name__}(level={self._level})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._level, "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        level = reader.read("H")
        return cls(level=level)


class EnableColorImages(Packet):
    __slots__ = (
        "_enable",  # bool
    )

    def __init__(self, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0x66)
        self.enable = enable

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        enable = bool(reader.read("b"))
        return cls(enable=enable)


class NvStorageOp(Packet):
    __slots__ = (
        "_data",  # uint8[uint16]
        "_length",  # int32
        "_op",  # NvOperation
        "_tag",  # NvEntryTag
        "_unknown",  # uint8
    )

    def __init__(self, tag=4294967295, length=0, op=0, unknown=0, data=()):
        super().__init__(PacketType.COMMAND, packet_id=0x81)
        self.tag = NvEntryTag(tag)
        self.length = length
        self.op = NvOperation(op)
        self.unknown = unknown
        self.data = data

    @property
    def tag(self) -> NvEntryTag:
        return self._tag

    @tag.setter
    def tag(self, value: NvEntryTag) -> None:
        self._tag = value
        validate_integer("tag", value.value, 0, 4294967295)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = validate_integer("length", value, -2147483648, 2147483647)

    @property
    def op(self) -> NvOperation:
        return self._op

    @op.setter
    def op(self, value: NvOperation) -> None:
        self._op = value
        validate_integer("op", value.value, 0, 255)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 255)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_varray(
            "data",
            value,
            65535,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 4 + 4 + 1 + 1 + get_varray_size(self._data, "H", "B")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"tag={self._tag}, "
            f"length={self._length}, "
            f"op={self._op}, "
            f"unknown={self._unknown}, "
            f"data={self._data})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._tag.value, "L")
        writer.write(self._length, "l")
        writer.write(self._op.value, "B")
        writer.write(self._unknown, "B")
        writer.write_varray(self._data, "B", "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        tag = reader.read("L")
        length = reader.read("l")
        op = reader.read("B")
        unknown = reader.read("B")
        data = reader.read_varray("B", "H")
        return cls(tag=tag, length=length, op=op, unknown=unknown, data=data)


class AbortAnimation(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x8D)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class OutputAudio(Packet):
    __slots__ = (
        "_samples",  # uint8[744]
    )

    def __init__(self, samples=()):
        super().__init__(PacketType.COMMAND, packet_id=0x8E)
        self.samples = samples

    @property
    def samples(self):
        return self._samples

    @samples.setter
    def samples(self, value):
        self._samples = validate_farray(
            "samples",
            value,
            744,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 744

    def __repr__(self):
        return f"{type(self).__name__}(samples={self._samples})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_farray(self._samples, "B", 744)

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        samples = reader.read_farray("B", 744)
        return cls(samples=samples)


class OutputSilence(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x8F)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class RecordHeading(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x91)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class TurnToRecordedHeading(Packet):
    # Hand-edited for the 2458 wire shape: six int16 values followed by one
    # uint8 (13 bytes).  2381 declared no fields, so every send was short.
    # See not_used/docs/protocol.md.
    __slots__ = (
        "_unknown",  # int16[6]
        "_unknown2",  # uint8
    )

    def __init__(self, unknown=(), unknown2=0):
        super().__init__(PacketType.COMMAND, packet_id=0x92)
        self.unknown = unknown
        self.unknown2 = unknown2

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_farray(
            "unknown",
            value,
            6,
            lambda name, value_inner: validate_integer(name, value_inner, -32768, 32767),
        )

    @property
    def unknown2(self):
        return self._unknown2

    @unknown2.setter
    def unknown2(self, value):
        self._unknown2 = validate_integer("unknown2", value, 0, 255)

    def __len__(self):
        return 6 * 2 + 1

    def __repr__(self):
        return f"{type(self).__name__}(unknown={self._unknown}, unknown2={self._unknown2})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_farray(self._unknown, "h", 6)
        writer.write(self._unknown2, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read_farray("h", 6)
        unknown2 = reader.read("B")
        return cls(unknown=unknown, unknown2=unknown2)


class AnimHead(Packet):
    __slots__ = (
        "_angle_deg",  # int8
        "_duration_ms",  # uint8
        "_variability_deg",  # int8
    )

    def __init__(self, duration_ms=0, variability_deg=0, angle_deg=0):
        super().__init__(PacketType.COMMAND, packet_id=0x93)
        self.duration_ms = duration_ms
        self.variability_deg = variability_deg
        self.angle_deg = angle_deg

    @property
    def duration_ms(self):
        return self._duration_ms

    @duration_ms.setter
    def duration_ms(self, value):
        self._duration_ms = validate_integer("duration_ms", value, 0, 255)

    @property
    def variability_deg(self):
        return self._variability_deg

    @variability_deg.setter
    def variability_deg(self, value):
        self._variability_deg = validate_integer("variability_deg", value, -128, 127)

    @property
    def angle_deg(self):
        return self._angle_deg

    @angle_deg.setter
    def angle_deg(self, value):
        self._angle_deg = validate_integer("angle_deg", value, -128, 127)

    def __len__(self):
        return 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"duration_ms={self._duration_ms}, "
            f"variability_deg={self._variability_deg}, "
            f"angle_deg={self._angle_deg})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._duration_ms, "B")
        writer.write(self._variability_deg, "b")
        writer.write(self._angle_deg, "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        duration_ms = reader.read("B")
        variability_deg = reader.read("b")
        angle_deg = reader.read("b")
        return cls(
            duration_ms=duration_ms,
            variability_deg=variability_deg,
            angle_deg=angle_deg,
        )


class AnimLift(Packet):
    __slots__ = (
        "_duration_ms",  # uint8
        "_height_mm",  # uint8
        "_variability_mm",  # uint8
    )

    def __init__(self, duration_ms=0, variability_mm=0, height_mm=0):
        super().__init__(PacketType.COMMAND, packet_id=0x94)
        self.duration_ms = duration_ms
        self.variability_mm = variability_mm
        self.height_mm = height_mm

    @property
    def duration_ms(self):
        return self._duration_ms

    @duration_ms.setter
    def duration_ms(self, value):
        self._duration_ms = validate_integer("duration_ms", value, 0, 255)

    @property
    def variability_mm(self):
        return self._variability_mm

    @variability_mm.setter
    def variability_mm(self, value):
        self._variability_mm = validate_integer("variability_mm", value, 0, 255)

    @property
    def height_mm(self):
        return self._height_mm

    @height_mm.setter
    def height_mm(self, value):
        self._height_mm = validate_integer("height_mm", value, 0, 255)

    def __len__(self):
        return 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"duration_ms={self._duration_ms}, "
            f"variability_mm={self._variability_mm}, "
            f"height_mm={self._height_mm})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._duration_ms, "B")
        writer.write(self._variability_mm, "B")
        writer.write(self._height_mm, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        duration_ms = reader.read("B")
        variability_mm = reader.read("B")
        height_mm = reader.read("B")
        return cls(duration_ms=duration_ms, variability_mm=variability_mm, height_mm=height_mm)


class DisplayImage(Packet):
    __slots__ = (
        "_image",  # uint8[uint16]
    )

    def __init__(self, image=()):
        super().__init__(PacketType.COMMAND, packet_id=0x97)
        self.image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = validate_varray(
            "image",
            value,
            65535,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return get_varray_size(self._image, "H", "B")

    def __repr__(self):
        return f"{type(self).__name__}(image={self._image})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_varray(self._image, "B", "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        image = reader.read_varray("B", "H")
        return cls(image=image)


class AnimBackpackLights(Packet):
    __slots__ = (
        "_colors",  # uint16[5]
    )

    def __init__(self, colors=()):
        super().__init__(PacketType.COMMAND, packet_id=0x98)
        # Left, front, middle, back, and right.
        self.colors = colors

    @property
    def colors(self):
        return self._colors

    @colors.setter
    def colors(self, value):
        self._colors = validate_farray(
            "colors",
            value,
            5,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 65535),
        )

    def __len__(self):
        return 10

    def __repr__(self):
        return f"{type(self).__name__}(colors={self._colors})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_farray(self._colors, "H", 5)

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        colors = reader.read_farray("H", 5)
        return cls(colors=colors)


class AnimBody(Packet):
    __slots__ = (
        "_speed",  # int16
        "_unknown",  # int16
    )

    def __init__(self, speed=0, unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0x99)
        self.speed = speed
        self.unknown = unknown

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = validate_integer("speed", value, -32768, 32767)

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, -32768, 32767)

    def __len__(self):
        return 2 + 2

    def __repr__(self):
        return f"{type(self).__name__}(speed={self._speed}, unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._speed, "h")
        writer.write(self._unknown, "h")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        speed = reader.read("h")
        unknown = reader.read("h")
        return cls(speed=speed, unknown=unknown)


class EndAnimation(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x9A)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class StartAnimation(Packet):
    __slots__ = (
        "_anim_id",  # uint8
    )

    def __init__(self, anim_id=0):
        super().__init__(PacketType.COMMAND, packet_id=0x9B)
        self.anim_id = anim_id

    @property
    def anim_id(self):
        return self._anim_id

    @anim_id.setter
    def anim_id(self, value):
        self._anim_id = validate_integer("anim_id", value, 0, 255)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(anim_id={self._anim_id})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._anim_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        anim_id = reader.read("B")
        return cls(anim_id=anim_id)


class EnableAnimationState(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0x9F)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class ShutdownRobot(Packet):
    # Hand-edited for the 2458 wire shape: a fixed 32-byte payload.  2381
    # declared no fields, so the packet was short and the robot dropped it.
    # The blob's internal structure is not recovered.  See
    # not_used/docs/protocol.md.
    __slots__ = (
        "_unknown",  # uint8[32]
    )

    def __init__(self, unknown=()):
        super().__init__(PacketType.COMMAND, packet_id=0xA9)
        self.unknown = unknown

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_farray(
            "unknown",
            value,
            32,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 32

    def __repr__(self):
        return f"{type(self).__name__}(unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write_farray(self._unknown, "B", 32)

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read_farray("B", 32)
        return cls(unknown=unknown)


class WifiOff(Packet):
    __slots__ = (
        "_enable",  # bool
    )

    def __init__(self, enable=False):
        super().__init__(PacketType.COMMAND, packet_id=0xAE)
        self.enable = enable

    @property
    def enable(self):
        return self._enable

    @enable.setter
    def enable(self, value):
        self._enable = validate_bool("enable", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(enable={self._enable})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._enable), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        enable = bool(reader.read("b"))
        return cls(enable=enable)


class FirmwareUpdate(Packet):
    __slots__ = (
        "_chunk_id",  # uint16
        "_data",  # uint8[1024]
    )

    def __init__(self, chunk_id=0, data=()):
        super().__init__(PacketType.COMMAND, packet_id=0xAF)
        self.chunk_id = chunk_id
        self.data = data

    @property
    def chunk_id(self):
        return self._chunk_id

    @chunk_id.setter
    def chunk_id(self, value):
        self._chunk_id = validate_integer("chunk_id", value, 0, 65535)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_farray(
            "data",
            value,
            1024,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 2 + 1024

    def __repr__(self):
        return f"{type(self).__name__}(chunk_id={self._chunk_id}, data={self._data})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._chunk_id, "H")
        writer.write_farray(self._data, "B", 1024)

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        chunk_id = reader.read("H")
        data = reader.read_farray("B", 1024)
        return cls(chunk_id=chunk_id, data=data)


class DebugData(Packet):
    __slots__ = (
        "_args",  # uint32[uint8]
        "_format_id",  # uint16
        "_level",  # int8
        "_name_id",  # uint16
        "_unused",  # uint16
    )

    def __init__(self, format_id=0, unused=0, name_id=0, level=0, args=()):
        super().__init__(PacketType.COMMAND, packet_id=0xB0)
        # AnkiLogStringTables.json formatTable key.
        self.format_id = format_id
        # Always 0.
        self.unused = unused
        # AnkiLogStringTables.json nameTable key.
        self.name_id = name_id
        # Log level. Observed: -1, 1, 2, 3, 5.
        self.level = level
        self.args = args

    @property
    def format_id(self):
        return self._format_id

    @format_id.setter
    def format_id(self, value):
        self._format_id = validate_integer("format_id", value, 0, 65535)

    @property
    def unused(self):
        return self._unused

    @unused.setter
    def unused(self, value):
        self._unused = validate_integer("unused", value, 0, 65535)

    @property
    def name_id(self):
        return self._name_id

    @name_id.setter
    def name_id(self, value):
        self._name_id = validate_integer("name_id", value, 0, 65535)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._level = validate_integer("level", value, -128, 127)

    @property
    def args(self):
        return self._args

    @args.setter
    def args(self, value):
        self._args = validate_varray(
            "args",
            value,
            255,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 4294967295),
        )

    def __len__(self):
        return 2 + 2 + 2 + 1 + get_varray_size(self._args, "B", "L")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"format_id={self._format_id}, "
            f"unused={self._unused}, "
            f"name_id={self._name_id}, "
            f"level={self._level}, "
            f"args={self._args})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._format_id, "H")
        writer.write(self._unused, "H")
        writer.write(self._name_id, "H")
        writer.write(self._level, "b")
        writer.write_varray(self._args, "L", "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        format_id = reader.read("H")
        unused = reader.read("H")
        name_id = reader.read("H")
        level = reader.read("b")
        args = reader.read_varray("L", "B")
        return cls(format_id=format_id, unused=unused, name_id=name_id, level=level, args=args)


class ObjectMoved(Packet):
    __slots__ = (
        "_active_accel_x",  # float
        "_active_accel_y",  # float
        "_active_accel_z",  # float
        "_axis_of_accel",  # UpAxis
        "_object_id",  # uint32
        "_timestamp",  # uint32
    )

    def __init__(
        self,
        timestamp=0,
        object_id=0,
        active_accel_x=0.0,
        active_accel_y=0.0,
        active_accel_z=0.0,
        axis_of_accel=7,
    ):
        super().__init__(PacketType.COMMAND, packet_id=0xB4)
        self.timestamp = timestamp
        self.object_id = object_id
        self.active_accel_x = active_accel_x
        self.active_accel_y = active_accel_y
        self.active_accel_z = active_accel_z
        self.axis_of_accel = UpAxis(axis_of_accel)

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def active_accel_x(self):
        return self._active_accel_x

    @active_accel_x.setter
    def active_accel_x(self, value):
        self._active_accel_x = validate_float("active_accel_x", value)

    @property
    def active_accel_y(self):
        return self._active_accel_y

    @active_accel_y.setter
    def active_accel_y(self, value):
        self._active_accel_y = validate_float("active_accel_y", value)

    @property
    def active_accel_z(self):
        return self._active_accel_z

    @active_accel_z.setter
    def active_accel_z(self, value):
        self._active_accel_z = validate_float("active_accel_z", value)

    @property
    def axis_of_accel(self) -> UpAxis:
        return self._axis_of_accel

    @axis_of_accel.setter
    def axis_of_accel(self, value: UpAxis) -> None:
        self._axis_of_accel = value
        validate_integer("axis_of_accel", value.value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"object_id={self._object_id}, "
            f"active_accel_x={self._active_accel_x}, "
            f"active_accel_y={self._active_accel_y}, "
            f"active_accel_z={self._active_accel_z}, "
            f"axis_of_accel={self._axis_of_accel})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")
        writer.write(self._active_accel_x, "f")
        writer.write(self._active_accel_y, "f")
        writer.write(self._active_accel_z, "f")
        writer.write(self._axis_of_accel.value, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        active_accel_x = reader.read("f")
        active_accel_y = reader.read("f")
        active_accel_z = reader.read("f")
        axis_of_accel = reader.read("B")
        return cls(
            timestamp=timestamp,
            object_id=object_id,
            active_accel_x=active_accel_x,
            active_accel_y=active_accel_y,
            active_accel_z=active_accel_z,
            axis_of_accel=axis_of_accel,
        )


class ObjectStoppedMoving(Packet):
    __slots__ = (
        "_object_id",  # uint32
        "_timestamp",  # uint32
    )

    def __init__(self, timestamp=0, object_id=0):
        super().__init__(PacketType.COMMAND, packet_id=0xB5)
        self.timestamp = timestamp
        self.object_id = object_id

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    def __len__(self):
        return 4 + 4

    def __repr__(self):
        return f"{type(self).__name__}(timestamp={self._timestamp}, object_id={self._object_id})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        return cls(timestamp=timestamp, object_id=object_id)


class ObjectTapped(Packet):
    __slots__ = (
        "_num_taps",  # uint8
        "_object_id",  # uint32
        "_tap_neg",  # int8
        "_tap_pos",  # int8
        "_tap_time",  # uint8
        "_timestamp",  # uint32
    )

    def __init__(self, timestamp=0, object_id=0, num_taps=0, tap_time=0, tap_neg=0, tap_pos=0):
        super().__init__(PacketType.COMMAND, packet_id=0xB6)
        self.timestamp = timestamp
        self.object_id = object_id
        self.num_taps = num_taps
        self.tap_time = tap_time
        self.tap_neg = tap_neg
        self.tap_pos = tap_pos

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def num_taps(self):
        return self._num_taps

    @num_taps.setter
    def num_taps(self, value):
        self._num_taps = validate_integer("num_taps", value, 0, 255)

    @property
    def tap_time(self):
        return self._tap_time

    @tap_time.setter
    def tap_time(self, value):
        self._tap_time = validate_integer("tap_time", value, 0, 255)

    @property
    def tap_neg(self):
        return self._tap_neg

    @tap_neg.setter
    def tap_neg(self, value):
        self._tap_neg = validate_integer("tap_neg", value, -128, 127)

    @property
    def tap_pos(self):
        return self._tap_pos

    @tap_pos.setter
    def tap_pos(self, value):
        self._tap_pos = validate_integer("tap_pos", value, -128, 127)

    def __len__(self):
        return 4 + 4 + 1 + 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"object_id={self._object_id}, "
            f"num_taps={self._num_taps}, "
            f"tap_time={self._tap_time}, "
            f"tap_neg={self._tap_neg}, "
            f"tap_pos={self._tap_pos})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")
        writer.write(self._num_taps, "B")
        writer.write(self._tap_time, "B")
        writer.write(self._tap_neg, "b")
        writer.write(self._tap_pos, "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        num_taps = reader.read("B")
        tap_time = reader.read("B")
        tap_neg = reader.read("b")
        tap_pos = reader.read("b")
        return cls(
            timestamp=timestamp,
            object_id=object_id,
            num_taps=num_taps,
            tap_time=tap_time,
            tap_neg=tap_neg,
            tap_pos=tap_pos,
        )


class ObjectTapFiltered(Packet):
    __slots__ = (
        "_intensity",  # uint8
        "_object_id",  # uint32
        "_time",  # uint8
        "_timestamp",  # uint32
    )

    def __init__(self, timestamp=0, object_id=0, time=0, intensity=0):
        super().__init__(PacketType.COMMAND, packet_id=0xB9)
        self.timestamp = timestamp
        self.object_id = object_id
        self.time = time
        self.intensity = intensity

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = validate_integer("time", value, 0, 255)

    @property
    def intensity(self):
        return self._intensity

    @intensity.setter
    def intensity(self, value):
        self._intensity = validate_integer("intensity", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"object_id={self._object_id}, "
            f"time={self._time}, "
            f"intensity={self._intensity})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")
        writer.write(self._time, "B")
        writer.write(self._intensity, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        time = reader.read("B")
        intensity = reader.read("B")
        return cls(timestamp=timestamp, object_id=object_id, time=time, intensity=intensity)


class AcknowledgeAction(Packet):
    __slots__ = (
        "_action_id",  # uint8
    )

    def __init__(self, action_id=0):
        super().__init__(PacketType.COMMAND, packet_id=0xC4)
        self.action_id = action_id

    @property
    def action_id(self):
        return self._action_id

    @action_id.setter
    def action_id(self, value):
        self._action_id = validate_integer("action_id", value, 0, 255)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(action_id={self._action_id})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._action_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        action_id = reader.read("B")
        return cls(action_id=action_id)


class RobotDelocalized(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0xC2)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class RobotPoked(Packet):
    __slots__ = ()

    def __init__(self):
        super().__init__(PacketType.COMMAND, packet_id=0xC3)

    def __len__(self):
        return 0

    def __repr__(self):
        return f"{type(self).__name__}()"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        pass

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        del reader
        return cls()


class PathFollowingEvent(Packet):
    __slots__ = (
        "_event_id",  # uint16
        "_event_type",  # PathEventType
    )

    def __init__(self, event_id=0, event_type=0):
        super().__init__(PacketType.COMMAND, packet_id=0xC6)
        self.event_id = event_id
        self.event_type = PathEventType(event_type)

    @property
    def event_id(self):
        return self._event_id

    @event_id.setter
    def event_id(self, value):
        self._event_id = validate_integer("event_id", value, 0, 65535)

    @property
    def event_type(self) -> PathEventType:
        return self._event_type

    @event_type.setter
    def event_type(self, value: PathEventType) -> None:
        self._event_type = value
        validate_integer("event_type", value.value, 0, 255)

    def __len__(self):
        return 2 + 1

    def __repr__(self):
        return f"{type(self).__name__}(event_id={self._event_id}, event_type={self._event_type})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._event_id, "H")
        writer.write(self._event_type.value, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        event_id = reader.read("H")
        event_type = reader.read("B")
        return cls(event_id=event_id, event_type=event_type)


class HardwareInfo(Packet):
    __slots__ = (
        "_serial_number_head",  # uint32
        "_unknown1",  # uint8
        "_unknown2",  # uint8
    )

    def __init__(self, serial_number_head=0, unknown1=0, unknown2=0):
        super().__init__(PacketType.COMMAND, packet_id=0xC9)
        self.serial_number_head = serial_number_head
        self.unknown1 = unknown1
        self.unknown2 = unknown2

    @property
    def serial_number_head(self):
        return self._serial_number_head

    @serial_number_head.setter
    def serial_number_head(self, value):
        self._serial_number_head = validate_integer("serial_number_head", value, 0, 4294967295)

    @property
    def unknown1(self):
        return self._unknown1

    @unknown1.setter
    def unknown1(self, value):
        self._unknown1 = validate_integer("unknown1", value, 0, 255)

    @property
    def unknown2(self):
        return self._unknown2

    @unknown2.setter
    def unknown2(self, value):
        self._unknown2 = validate_integer("unknown2", value, 0, 255)

    def __len__(self):
        return 4 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"serial_number_head={self._serial_number_head}, "
            f"unknown1={self._unknown1}, "
            f"unknown2={self._unknown2})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._serial_number_head, "L")
        writer.write(self._unknown1, "B")
        writer.write(self._unknown2, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        serial_number_head = reader.read("L")
        unknown1 = reader.read("B")
        unknown2 = reader.read("B")
        return cls(serial_number_head=serial_number_head, unknown1=unknown1, unknown2=unknown2)


class AnimationStarted(Packet):
    __slots__ = (
        "_anim_id",  # uint8
    )

    def __init__(self, anim_id=0):
        super().__init__(PacketType.COMMAND, packet_id=0xCA)
        self.anim_id = anim_id

    @property
    def anim_id(self):
        return self._anim_id

    @anim_id.setter
    def anim_id(self, value):
        self._anim_id = validate_integer("anim_id", value, 0, 255)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(anim_id={self._anim_id})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._anim_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        anim_id = reader.read("B")
        return cls(anim_id=anim_id)


class AnimationEnded(Packet):
    __slots__ = (
        "_anim_id",  # uint8
    )

    def __init__(self, anim_id=0):
        super().__init__(PacketType.COMMAND, packet_id=0xCB)
        self.anim_id = anim_id

    @property
    def anim_id(self):
        return self._anim_id

    @anim_id.setter
    def anim_id(self, value):
        self._anim_id = validate_integer("anim_id", value, 0, 255)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(anim_id={self._anim_id})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._anim_id, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        anim_id = reader.read("B")
        return cls(anim_id=anim_id)


class NvStorageOpResult(Packet):
    __slots__ = (
        "_data",  # uint8[uint16]
        "_length",  # int32
        "_op",  # NvOperation
        "_result",  # NvResult
        "_tag",  # NvEntryTag
    )

    def __init__(self, tag=4294967295, length=0, op=0, result=0, data=()):
        super().__init__(PacketType.COMMAND, packet_id=0xCD)
        self.tag = NvEntryTag(tag)
        self.length = length
        self.op = NvOperation(op)
        self.result = NvResult(result)
        self.data = data

    @property
    def tag(self) -> NvEntryTag:
        return self._tag

    @tag.setter
    def tag(self, value: NvEntryTag) -> None:
        self._tag = value
        validate_integer("tag", value.value, 0, 4294967295)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = validate_integer("length", value, -2147483648, 2147483647)

    @property
    def op(self) -> NvOperation:
        return self._op

    @op.setter
    def op(self, value: NvOperation) -> None:
        self._op = value
        validate_integer("op", value.value, 0, 255)

    @property
    def result(self) -> NvResult:
        return self._result

    @result.setter
    def result(self, value: NvResult) -> None:
        self._result = value
        validate_integer("result", value.value, -128, 127)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_varray(
            "data",
            value,
            65535,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 4 + 4 + 1 + 1 + get_varray_size(self._data, "H", "B")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"tag={self._tag}, "
            f"length={self._length}, "
            f"op={self._op}, "
            f"result={self._result}, "
            f"data={self._data})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._tag.value, "L")
        writer.write(self._length, "l")
        writer.write(self._op.value, "B")
        writer.write(self._result.value, "b")
        writer.write_varray(self._data, "B", "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        tag = reader.read("L")
        length = reader.read("l")
        op = reader.read("B")
        result = reader.read("b")
        data = reader.read_varray("B", "H")
        return cls(tag=tag, length=length, op=op, result=result, data=data)


class ObjectPowerLevel(Packet):
    __slots__ = (
        "_battery_level",  # uint8
        "_missed_packets",  # uint32
        "_object_id",  # uint32
    )

    def __init__(self, object_id=0, missed_packets=0, battery_level=0):
        super().__init__(PacketType.COMMAND, packet_id=0xCE)
        self.object_id = object_id
        self.missed_packets = missed_packets
        self.battery_level = battery_level

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def missed_packets(self):
        return self._missed_packets

    @missed_packets.setter
    def missed_packets(self, value):
        self._missed_packets = validate_integer("missed_packets", value, 0, 4294967295)

    @property
    def battery_level(self):
        return self._battery_level

    @battery_level.setter
    def battery_level(self, value):
        self._battery_level = validate_integer("battery_level", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"object_id={self._object_id}, "
            f"missed_packets={self._missed_packets}, "
            f"battery_level={self._battery_level})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._object_id, "L")
        writer.write(self._missed_packets, "L")
        writer.write(self._battery_level, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        object_id = reader.read("L")
        missed_packets = reader.read("L")
        battery_level = reader.read("B")
        return cls(
            object_id=object_id,
            missed_packets=missed_packets,
            battery_level=battery_level,
        )


class ObjectConnectionState(Packet):
    __slots__ = (
        "_connected",  # bool
        "_factory_id",  # uint32
        "_object_id",  # uint32
        "_object_type",  # ObjectType
    )

    def __init__(self, object_id=0, factory_id=0, object_type=-1, connected=False):
        super().__init__(PacketType.COMMAND, packet_id=0xD0)
        self.object_id = object_id
        self.factory_id = factory_id
        self.object_type = ObjectType(object_type)
        self.connected = connected

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def factory_id(self):
        return self._factory_id

    @factory_id.setter
    def factory_id(self, value):
        self._factory_id = validate_integer("factory_id", value, 0, 4294967295)

    @property
    def object_type(self) -> ObjectType:
        return self._object_type

    @object_type.setter
    def object_type(self, value: ObjectType) -> None:
        self._object_type = value
        validate_integer("object_type", value.value, -2147483648, 2147483647)

    @property
    def connected(self):
        return self._connected

    @connected.setter
    def connected(self, value):
        self._connected = validate_bool("connected", value)

    def __len__(self):
        return 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"object_id={self._object_id}, "
            f"factory_id={self._factory_id}, "
            f"object_type={self._object_type}, "
            f"connected={self._connected})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._object_id, "L")
        writer.write(self._factory_id, "L")
        writer.write(self._object_type.value, "l")
        writer.write(int(self._connected), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        object_id = reader.read("L")
        factory_id = reader.read("L")
        object_type = reader.read("l")
        connected = bool(reader.read("b"))
        return cls(
            object_id=object_id,
            factory_id=factory_id,
            object_type=object_type,
            connected=connected,
        )


class MotorCalibration(Packet):
    __slots__ = (
        "_auto_started",  # bool
        "_calib_started",  # bool
        "_motor_id",  # MotorID
    )

    def __init__(self, motor_id=0, calib_started=False, auto_started=False):
        super().__init__(PacketType.COMMAND, packet_id=0xD1)
        self.motor_id = MotorID(motor_id)
        self.calib_started = calib_started
        self.auto_started = auto_started

    @property
    def motor_id(self) -> MotorID:
        return self._motor_id

    @motor_id.setter
    def motor_id(self, value: MotorID) -> None:
        self._motor_id = value
        validate_integer("motor_id", value.value, 0, 255)

    @property
    def calib_started(self):
        return self._calib_started

    @calib_started.setter
    def calib_started(self, value):
        self._calib_started = validate_bool("calib_started", value)

    @property
    def auto_started(self):
        return self._auto_started

    @auto_started.setter
    def auto_started(self, value):
        self._auto_started = validate_bool("auto_started", value)

    def __len__(self):
        return 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"motor_id={self._motor_id}, "
            f"calib_started={self._calib_started}, "
            f"auto_started={self._auto_started})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._motor_id.value, "B")
        writer.write(int(self._calib_started), "b")
        writer.write(int(self._auto_started), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        motor_id = reader.read("B")
        calib_started = bool(reader.read("b"))
        auto_started = bool(reader.read("b"))
        return cls(motor_id=motor_id, calib_started=calib_started, auto_started=auto_started)


class ObjectUpAxisChanged(Packet):
    __slots__ = (
        "_axis",  # UpAxis
        "_object_id",  # uint32
        "_timestamp",  # uint32
    )

    def __init__(self, timestamp=0, object_id=0, axis=7):
        super().__init__(PacketType.COMMAND, packet_id=0xD7)
        self.timestamp = timestamp
        self.object_id = object_id
        self.axis = UpAxis(axis)

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def axis(self) -> UpAxis:
        return self._axis

    @axis.setter
    def axis(self, value: UpAxis) -> None:
        self._axis = value
        validate_integer("axis", value.value, 0, 255)

    def __len__(self):
        return 4 + 4 + 1

    def __repr__(self):
        return f"{type(self).__name__}(timestamp={self._timestamp}, object_id={self._object_id}, axis={self._axis})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")
        writer.write(self._axis.value, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        axis = reader.read("B")
        return cls(timestamp=timestamp, object_id=object_id, axis=axis)


class ButtonPressed(Packet):
    __slots__ = (
        "_pressed",  # bool
    )

    def __init__(self, pressed=False):
        super().__init__(PacketType.COMMAND, packet_id=0xDB)
        self.pressed = pressed

    @property
    def pressed(self):
        return self._pressed

    @pressed.setter
    def pressed(self, value):
        self._pressed = validate_bool("pressed", value)

    def __len__(self):
        return 1

    def __repr__(self):
        return f"{type(self).__name__}(pressed={self._pressed})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(int(self._pressed), "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        pressed = bool(reader.read("b"))
        return cls(pressed=pressed)


class FallingStarted(Packet):
    __slots__ = (
        "_unknown",  # uint32
    )

    def __init__(self, unknown=0):
        super().__init__(PacketType.COMMAND, packet_id=0xDD)
        self.unknown = unknown

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 4294967295)

    def __len__(self):
        return 4

    def __repr__(self):
        return f"{type(self).__name__}(unknown={self._unknown})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._unknown, "L")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read("L")
        return cls(unknown=unknown)


class FallingStopped(Packet):
    __slots__ = (
        "_duration_ms",  # uint32
        "_impact_intensity",  # float
        "_unknown",  # uint32
    )

    def __init__(self, unknown=0, duration_ms=0, impact_intensity=0.0):
        super().__init__(PacketType.COMMAND, packet_id=0xDE)
        self.unknown = unknown
        self.duration_ms = duration_ms
        self.impact_intensity = impact_intensity

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 4294967295)

    @property
    def duration_ms(self):
        return self._duration_ms

    @duration_ms.setter
    def duration_ms(self, value):
        self._duration_ms = validate_integer("duration_ms", value, 0, 4294967295)

    @property
    def impact_intensity(self):
        return self._impact_intensity

    @impact_intensity.setter
    def impact_intensity(self, value):
        self._impact_intensity = validate_float("impact_intensity", value)

    def __len__(self):
        return 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"unknown={self._unknown}, "
            f"duration_ms={self._duration_ms}, "
            f"impact_intensity={self._impact_intensity})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._unknown, "L")
        writer.write(self._duration_ms, "L")
        writer.write(self._impact_intensity, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read("L")
        duration_ms = reader.read("L")
        impact_intensity = reader.read("f")
        return cls(unknown=unknown, duration_ms=duration_ms, impact_intensity=impact_intensity)


class BodyInfo(Packet):
    __slots__ = (
        "_body_color",  # BodyColor
        "_body_hw_version",  # uint32
        "_serial_number",  # uint32
    )

    def __init__(self, serial_number=0, body_hw_version=0, body_color=-1):
        super().__init__(PacketType.COMMAND, packet_id=0xED)
        self.serial_number = serial_number
        # Production units report 5. Development units report 7.
        self.body_hw_version = body_hw_version
        self.body_color = BodyColor(body_color)

    @property
    def serial_number(self):
        return self._serial_number

    @serial_number.setter
    def serial_number(self, value):
        self._serial_number = validate_integer("serial_number", value, 0, 4294967295)

    @property
    def body_hw_version(self):
        return self._body_hw_version

    @body_hw_version.setter
    def body_hw_version(self, value):
        self._body_hw_version = validate_integer("body_hw_version", value, 0, 4294967295)

    @property
    def body_color(self) -> BodyColor:
        return self._body_color

    @body_color.setter
    def body_color(self, value: BodyColor) -> None:
        self._body_color = value
        validate_integer("body_color", value.value, -2147483648, 2147483647)

    def __len__(self):
        return 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"serial_number={self._serial_number}, "
            f"body_hw_version={self._body_hw_version}, "
            f"body_color={self._body_color})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._serial_number, "L")
        writer.write(self._body_hw_version, "L")
        writer.write(self._body_color.value, "l")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        serial_number = reader.read("L")
        body_hw_version = reader.read("L")
        body_color = reader.read("l")
        return cls(
            serial_number=serial_number,
            body_hw_version=body_hw_version,
            body_color=body_color,
        )


class FirmwareSignature(Packet):
    __slots__ = (
        "_signature",  # str
        "_unknown",  # uint16
    )

    def __init__(self, unknown=0, signature=""):
        super().__init__(PacketType.COMMAND, packet_id=0xEE)
        # Last 2 bytes of head s/n?
        self.unknown = unknown
        self.signature = signature

    @property
    def unknown(self):
        return self._unknown

    @unknown.setter
    def unknown(self, value):
        self._unknown = validate_integer("unknown", value, 0, 65535)

    @property
    def signature(self):
        return self._signature

    @signature.setter
    def signature(self, value):
        self._signature = validate_string("signature", value, 65535)

    def __len__(self):
        return 2 + get_string_size(self._signature, "H")

    def __repr__(self):
        return f"{type(self).__name__}(unknown={self._unknown}, signature={self._signature})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._unknown, "H")
        writer.write_string(self._signature, "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        unknown = reader.read("H")
        signature = reader.read_string("H")
        return cls(unknown=unknown, signature=signature)


class FirmwareUpdateResult(Packet):
    __slots__ = (
        "_byte_count",  # uint32
        "_chunk_id",  # uint16
        "_status",  # uint8
    )

    def __init__(self, byte_count=0, chunk_id=0, status=0):
        super().__init__(PacketType.COMMAND, packet_id=0xEF)
        self.byte_count = byte_count
        self.chunk_id = chunk_id
        # 0=OK; 0x0a=complete?
        self.status = status

    @property
    def byte_count(self):
        return self._byte_count

    @byte_count.setter
    def byte_count(self, value):
        self._byte_count = validate_integer("byte_count", value, 0, 4294967295)

    @property
    def chunk_id(self):
        return self._chunk_id

    @chunk_id.setter
    def chunk_id(self, value):
        self._chunk_id = validate_integer("chunk_id", value, 0, 65535)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = validate_integer("status", value, 0, 255)

    def __len__(self):
        return 4 + 2 + 1

    def __repr__(self):
        return f"{type(self).__name__}(byte_count={self._byte_count}, chunk_id={self._chunk_id}, status={self._status})"

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._byte_count, "L")
        writer.write(self._chunk_id, "H")
        writer.write(self._status, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        byte_count = reader.read("L")
        chunk_id = reader.read("H")
        status = reader.read("B")
        return cls(byte_count=byte_count, chunk_id=chunk_id, status=status)


class RobotState(Packet):
    __slots__ = (
        "_accel_x",  # float
        "_accel_y",  # float
        "_accel_z",  # float
        "_backpack_touch_sensor_raw",  # uint16
        "_battery_voltage",  # float
        "_cliff_data_raw",  # uint16[4]
        "_curr_path_segment",  # uint8
        "_gyro_x",  # float
        "_gyro_y",  # float
        "_gyro_z",  # float
        "_head_angle_rad",  # float
        "_lift_height_mm",  # float
        "_lwheel_speed_mmps",  # float
        "_pose_angle_rad",  # float
        "_pose_frame_id",  # uint32
        "_pose_origin_id",  # uint32
        "_pose_pitch_rad",  # float
        "_pose_x",  # float
        "_pose_y",  # float
        "_pose_z",  # float
        "_rwheel_speed_mmps",  # float
        "_status",  # uint32
        "_timestamp",  # uint32
    )

    def __init__(
        self,
        timestamp=0,
        pose_frame_id=0,
        pose_origin_id=0,
        pose_x=0.0,
        pose_y=0.0,
        pose_z=0.0,
        pose_angle_rad=0.0,
        pose_pitch_rad=0.0,
        lwheel_speed_mmps=0.0,
        rwheel_speed_mmps=0.0,
        head_angle_rad=0.0,
        lift_height_mm=0.0,
        accel_x=0.0,
        accel_y=0.0,
        accel_z=0.0,
        gyro_x=0.0,
        gyro_y=0.0,
        gyro_z=0.0,
        battery_voltage=0.0,
        status=0,
        cliff_data_raw=(),
        backpack_touch_sensor_raw=0,
        curr_path_segment=0,
    ):
        super().__init__(PacketType.EVENT, packet_id=0xF0)
        self.timestamp = timestamp
        self.pose_frame_id = pose_frame_id
        self.pose_origin_id = pose_origin_id
        self.pose_x = pose_x
        self.pose_y = pose_y
        self.pose_z = pose_z
        self.pose_angle_rad = pose_angle_rad
        self.pose_pitch_rad = pose_pitch_rad
        self.lwheel_speed_mmps = lwheel_speed_mmps
        self.rwheel_speed_mmps = rwheel_speed_mmps
        self.head_angle_rad = head_angle_rad
        self.lift_height_mm = lift_height_mm
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z
        self.gyro_x = gyro_x
        self.gyro_y = gyro_y
        self.gyro_z = gyro_z
        self.battery_voltage = battery_voltage
        self.status = status
        self.cliff_data_raw = cliff_data_raw
        self.backpack_touch_sensor_raw = backpack_touch_sensor_raw
        self.curr_path_segment = curr_path_segment

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def pose_frame_id(self):
        return self._pose_frame_id

    @pose_frame_id.setter
    def pose_frame_id(self, value):
        self._pose_frame_id = validate_integer("pose_frame_id", value, 0, 4294967295)

    @property
    def pose_origin_id(self):
        return self._pose_origin_id

    @pose_origin_id.setter
    def pose_origin_id(self, value):
        self._pose_origin_id = validate_integer("pose_origin_id", value, 0, 4294967295)

    @property
    def pose_x(self):
        return self._pose_x

    @pose_x.setter
    def pose_x(self, value):
        self._pose_x = validate_float("pose_x", value)

    @property
    def pose_y(self):
        return self._pose_y

    @pose_y.setter
    def pose_y(self, value):
        self._pose_y = validate_float("pose_y", value)

    @property
    def pose_z(self):
        return self._pose_z

    @pose_z.setter
    def pose_z(self, value):
        self._pose_z = validate_float("pose_z", value)

    @property
    def pose_angle_rad(self):
        return self._pose_angle_rad

    @pose_angle_rad.setter
    def pose_angle_rad(self, value):
        self._pose_angle_rad = validate_float("pose_angle_rad", value)

    @property
    def pose_pitch_rad(self):
        return self._pose_pitch_rad

    @pose_pitch_rad.setter
    def pose_pitch_rad(self, value):
        self._pose_pitch_rad = validate_float("pose_pitch_rad", value)

    @property
    def lwheel_speed_mmps(self):
        return self._lwheel_speed_mmps

    @lwheel_speed_mmps.setter
    def lwheel_speed_mmps(self, value):
        self._lwheel_speed_mmps = validate_float("lwheel_speed_mmps", value)

    @property
    def rwheel_speed_mmps(self):
        return self._rwheel_speed_mmps

    @rwheel_speed_mmps.setter
    def rwheel_speed_mmps(self, value):
        self._rwheel_speed_mmps = validate_float("rwheel_speed_mmps", value)

    @property
    def head_angle_rad(self):
        return self._head_angle_rad

    @head_angle_rad.setter
    def head_angle_rad(self, value):
        self._head_angle_rad = validate_float("head_angle_rad", value)

    @property
    def lift_height_mm(self):
        return self._lift_height_mm

    @lift_height_mm.setter
    def lift_height_mm(self, value):
        self._lift_height_mm = validate_float("lift_height_mm", value)

    @property
    def accel_x(self):
        return self._accel_x

    @accel_x.setter
    def accel_x(self, value):
        self._accel_x = validate_float("accel_x", value)

    @property
    def accel_y(self):
        return self._accel_y

    @accel_y.setter
    def accel_y(self, value):
        self._accel_y = validate_float("accel_y", value)

    @property
    def accel_z(self):
        return self._accel_z

    @accel_z.setter
    def accel_z(self, value):
        self._accel_z = validate_float("accel_z", value)

    @property
    def gyro_x(self):
        return self._gyro_x

    @gyro_x.setter
    def gyro_x(self, value):
        self._gyro_x = validate_float("gyro_x", value)

    @property
    def gyro_y(self):
        return self._gyro_y

    @gyro_y.setter
    def gyro_y(self, value):
        self._gyro_y = validate_float("gyro_y", value)

    @property
    def gyro_z(self):
        return self._gyro_z

    @gyro_z.setter
    def gyro_z(self, value):
        self._gyro_z = validate_float("gyro_z", value)

    @property
    def battery_voltage(self):
        return self._battery_voltage

    @battery_voltage.setter
    def battery_voltage(self, value):
        self._battery_voltage = validate_float("battery_voltage", value)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = validate_integer("status", value, 0, 4294967295)

    @property
    def cliff_data_raw(self):
        return self._cliff_data_raw

    @cliff_data_raw.setter
    def cliff_data_raw(self, value):
        self._cliff_data_raw = validate_farray(
            "cliff_data_raw",
            value,
            4,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 65535),
        )

    @property
    def backpack_touch_sensor_raw(self):
        return self._backpack_touch_sensor_raw

    @backpack_touch_sensor_raw.setter
    def backpack_touch_sensor_raw(self, value):
        self._backpack_touch_sensor_raw = validate_integer("backpack_touch_sensor_raw", value, 0, 65535)

    @property
    def curr_path_segment(self):
        return self._curr_path_segment

    @curr_path_segment.setter
    def curr_path_segment(self, value):
        self._curr_path_segment = validate_integer("curr_path_segment", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 2 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"pose_frame_id={self._pose_frame_id}, "
            f"pose_origin_id={self._pose_origin_id}, "
            f"pose_x={self._pose_x}, "
            f"pose_y={self._pose_y}, "
            f"pose_z={self._pose_z}, "
            f"pose_angle_rad={self._pose_angle_rad}, "
            f"pose_pitch_rad={self._pose_pitch_rad}, "
            f"lwheel_speed_mmps={self._lwheel_speed_mmps}, "
            f"rwheel_speed_mmps={self._rwheel_speed_mmps}, "
            f"head_angle_rad={self._head_angle_rad}, "
            f"lift_height_mm={self._lift_height_mm}, "
            f"accel_x={self._accel_x}, "
            f"accel_y={self._accel_y}, "
            f"accel_z={self._accel_z}, "
            f"gyro_x={self._gyro_x}, "
            f"gyro_y={self._gyro_y}, "
            f"gyro_z={self._gyro_z}, "
            f"battery_voltage={self._battery_voltage}, "
            f"status={self._status}, "
            f"cliff_data_raw={self._cliff_data_raw}, "
            f"backpack_touch_sensor_raw={self._backpack_touch_sensor_raw}, "
            f"curr_path_segment={self._curr_path_segment})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._pose_frame_id, "L")
        writer.write(self._pose_origin_id, "L")
        writer.write(self._pose_x, "f")
        writer.write(self._pose_y, "f")
        writer.write(self._pose_z, "f")
        writer.write(self._pose_angle_rad, "f")
        writer.write(self._pose_pitch_rad, "f")
        writer.write(self._lwheel_speed_mmps, "f")
        writer.write(self._rwheel_speed_mmps, "f")
        writer.write(self._head_angle_rad, "f")
        writer.write(self._lift_height_mm, "f")
        writer.write(self._accel_x, "f")
        writer.write(self._accel_y, "f")
        writer.write(self._accel_z, "f")
        writer.write(self._gyro_x, "f")
        writer.write(self._gyro_y, "f")
        writer.write(self._gyro_z, "f")
        writer.write(self._battery_voltage, "f")
        writer.write(self._status, "L")
        writer.write_farray(self._cliff_data_raw, "H", 4)
        writer.write(self._backpack_touch_sensor_raw, "H")
        writer.write(self._curr_path_segment, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        pose_frame_id = reader.read("L")
        pose_origin_id = reader.read("L")
        pose_x = reader.read("f")
        pose_y = reader.read("f")
        pose_z = reader.read("f")
        pose_angle_rad = reader.read("f")
        pose_pitch_rad = reader.read("f")
        lwheel_speed_mmps = reader.read("f")
        rwheel_speed_mmps = reader.read("f")
        head_angle_rad = reader.read("f")
        lift_height_mm = reader.read("f")
        accel_x = reader.read("f")
        accel_y = reader.read("f")
        accel_z = reader.read("f")
        gyro_x = reader.read("f")
        gyro_y = reader.read("f")
        gyro_z = reader.read("f")
        battery_voltage = reader.read("f")
        status = reader.read("L")
        cliff_data_raw = reader.read_farray("H", 4)
        backpack_touch_sensor_raw = reader.read("H")
        curr_path_segment = reader.read("B")
        return cls(
            timestamp=timestamp,
            pose_frame_id=pose_frame_id,
            pose_origin_id=pose_origin_id,
            pose_x=pose_x,
            pose_y=pose_y,
            pose_z=pose_z,
            pose_angle_rad=pose_angle_rad,
            pose_pitch_rad=pose_pitch_rad,
            lwheel_speed_mmps=lwheel_speed_mmps,
            rwheel_speed_mmps=rwheel_speed_mmps,
            head_angle_rad=head_angle_rad,
            lift_height_mm=lift_height_mm,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            gyro_x=gyro_x,
            gyro_y=gyro_y,
            gyro_z=gyro_z,
            battery_voltage=battery_voltage,
            status=status,
            cliff_data_raw=cliff_data_raw,
            backpack_touch_sensor_raw=backpack_touch_sensor_raw,
            curr_path_segment=curr_path_segment,
        )


class AnimationState(Packet):
    __slots__ = (
        "_client_drop_count",  # uint8
        "_enabled_anim_tracks",  # uint8
        "_num_anim_bytes_played",  # int32
        "_num_audio_frames_played",  # int32
        "_tag",  # uint8
        "_timestamp",  # uint32
    )

    def __init__(
        self,
        timestamp=0,
        num_anim_bytes_played=0,
        num_audio_frames_played=0,
        enabled_anim_tracks=0,
        tag=0,
        client_drop_count=0,
    ):
        super().__init__(PacketType.EVENT, packet_id=0xF1)
        self.timestamp = timestamp
        self.num_anim_bytes_played = num_anim_bytes_played
        self.num_audio_frames_played = num_audio_frames_played
        self.enabled_anim_tracks = enabled_anim_tracks
        self.tag = tag
        # Not present in v2214 and older.
        self.client_drop_count = client_drop_count

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def num_anim_bytes_played(self):
        return self._num_anim_bytes_played

    @num_anim_bytes_played.setter
    def num_anim_bytes_played(self, value):
        self._num_anim_bytes_played = validate_integer("num_anim_bytes_played", value, -2147483648, 2147483647)

    @property
    def num_audio_frames_played(self):
        return self._num_audio_frames_played

    @num_audio_frames_played.setter
    def num_audio_frames_played(self, value):
        self._num_audio_frames_played = validate_integer("num_audio_frames_played", value, -2147483648, 2147483647)

    @property
    def enabled_anim_tracks(self):
        return self._enabled_anim_tracks

    @enabled_anim_tracks.setter
    def enabled_anim_tracks(self, value):
        self._enabled_anim_tracks = validate_integer("enabled_anim_tracks", value, 0, 255)

    @property
    def tag(self):
        return self._tag

    @tag.setter
    def tag(self, value):
        self._tag = validate_integer("tag", value, 0, 255)

    @property
    def client_drop_count(self):
        return self._client_drop_count

    @client_drop_count.setter
    def client_drop_count(self, value):
        self._client_drop_count = validate_integer("client_drop_count", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 1 + 1 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"num_anim_bytes_played={self._num_anim_bytes_played}, "
            f"num_audio_frames_played={self._num_audio_frames_played}, "
            f"enabled_anim_tracks={self._enabled_anim_tracks}, "
            f"tag={self._tag}, "
            f"client_drop_count={self._client_drop_count})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._num_anim_bytes_played, "l")
        writer.write(self._num_audio_frames_played, "l")
        writer.write(self._enabled_anim_tracks, "B")
        writer.write(self._tag, "B")
        writer.write(self._client_drop_count, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        num_anim_bytes_played = reader.read("l")
        num_audio_frames_played = reader.read("l")
        enabled_anim_tracks = reader.read("B")
        tag = reader.read("B")
        client_drop_count = reader.read("B")
        return cls(
            timestamp=timestamp,
            num_anim_bytes_played=num_anim_bytes_played,
            num_audio_frames_played=num_audio_frames_played,
            enabled_anim_tracks=enabled_anim_tracks,
            tag=tag,
            client_drop_count=client_drop_count,
        )


class ImageChunk(Packet):
    __slots__ = (
        "_chunk_debug",  # uint32
        "_chunk_id",  # uint8
        "_data",  # uint8[uint16]
        "_frame_timestamp",  # uint32
        "_image_chunk_count",  # uint8
        "_image_encoding",  # ImageEncoding
        "_image_id",  # uint32
        "_image_resolution",  # ImageResolution
        "_status",  # uint16
    )

    def __init__(
        self,
        frame_timestamp=0,
        image_id=0,
        chunk_debug=0,
        image_encoding=0,
        image_resolution=0,
        image_chunk_count=0,
        chunk_id=0,
        status=0,
        data=(),
    ):
        super().__init__(PacketType.EVENT, packet_id=0xF2)
        self.frame_timestamp = frame_timestamp
        self.image_id = image_id
        self.chunk_debug = chunk_debug
        self.image_encoding = ImageEncoding(image_encoding)
        self.image_resolution = ImageResolution(image_resolution)
        self.image_chunk_count = image_chunk_count
        self.chunk_id = chunk_id
        self.status = status
        self.data = data

    @property
    def frame_timestamp(self):
        return self._frame_timestamp

    @frame_timestamp.setter
    def frame_timestamp(self, value):
        self._frame_timestamp = validate_integer("frame_timestamp", value, 0, 4294967295)

    @property
    def image_id(self):
        return self._image_id

    @image_id.setter
    def image_id(self, value):
        self._image_id = validate_integer("image_id", value, 0, 4294967295)

    @property
    def chunk_debug(self):
        return self._chunk_debug

    @chunk_debug.setter
    def chunk_debug(self, value):
        self._chunk_debug = validate_integer("chunk_debug", value, 0, 4294967295)

    @property
    def image_encoding(self) -> ImageEncoding:
        return self._image_encoding

    @image_encoding.setter
    def image_encoding(self, value: ImageEncoding) -> None:
        self._image_encoding = value
        validate_integer("image_encoding", value.value, -128, 127)

    @property
    def image_resolution(self) -> ImageResolution:
        return self._image_resolution

    @image_resolution.setter
    def image_resolution(self, value: ImageResolution) -> None:
        self._image_resolution = value
        validate_integer("image_resolution", value.value, -128, 127)

    @property
    def image_chunk_count(self):
        return self._image_chunk_count

    @image_chunk_count.setter
    def image_chunk_count(self, value):
        self._image_chunk_count = validate_integer("image_chunk_count", value, 0, 255)

    @property
    def chunk_id(self):
        return self._chunk_id

    @chunk_id.setter
    def chunk_id(self, value):
        self._chunk_id = validate_integer("chunk_id", value, 0, 255)

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, value):
        self._status = validate_integer("status", value, 0, 65535)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = validate_varray(
            "data",
            value,
            65535,
            lambda name, value_inner: validate_integer(name, value_inner, 0, 255),
        )

    def __len__(self):
        return 4 + 4 + 4 + 1 + 1 + 1 + 1 + 2 + get_varray_size(self._data, "H", "B")

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"frame_timestamp={self._frame_timestamp}, "
            f"image_id={self._image_id}, "
            f"chunk_debug={self._chunk_debug}, "
            f"image_encoding={self._image_encoding}, "
            f"image_resolution={self._image_resolution}, "
            f"image_chunk_count={self._image_chunk_count}, "
            f"chunk_id={self._chunk_id}, "
            f"status={self._status}, "
            f"data={self._data})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._frame_timestamp, "L")
        writer.write(self._image_id, "L")
        writer.write(self._chunk_debug, "L")
        writer.write(self._image_encoding.value, "b")
        writer.write(self._image_resolution.value, "b")
        writer.write(self._image_chunk_count, "B")
        writer.write(self._chunk_id, "B")
        writer.write(self._status, "H")
        writer.write_varray(self._data, "B", "H")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        frame_timestamp = reader.read("L")
        image_id = reader.read("L")
        chunk_debug = reader.read("L")
        image_encoding = reader.read("b")
        image_resolution = reader.read("b")
        image_chunk_count = reader.read("B")
        chunk_id = reader.read("B")
        status = reader.read("H")
        data = reader.read_varray("B", "H")
        return cls(
            frame_timestamp=frame_timestamp,
            image_id=image_id,
            chunk_debug=chunk_debug,
            image_encoding=image_encoding,
            image_resolution=image_resolution,
            image_chunk_count=image_chunk_count,
            chunk_id=chunk_id,
            status=status,
            data=data,
        )


class ObjectAvailable(Packet):
    __slots__ = (
        "_factory_id",  # uint32
        "_object_type",  # ObjectType
        "_rssi",  # int8
    )

    def __init__(self, factory_id=0, object_type=-1, rssi=0):
        super().__init__(PacketType.EVENT, packet_id=0xF3)
        self.factory_id = factory_id
        self.object_type = ObjectType(object_type)
        self.rssi = rssi

    @property
    def factory_id(self):
        return self._factory_id

    @factory_id.setter
    def factory_id(self, value):
        self._factory_id = validate_integer("factory_id", value, 0, 4294967295)

    @property
    def object_type(self) -> ObjectType:
        return self._object_type

    @object_type.setter
    def object_type(self, value: ObjectType) -> None:
        self._object_type = value
        validate_integer("object_type", value.value, -2147483648, 2147483647)

    @property
    def rssi(self):
        return self._rssi

    @rssi.setter
    def rssi(self, value):
        self._rssi = validate_integer("rssi", value, -128, 127)

    def __len__(self):
        return 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}(factory_id={self._factory_id}, object_type={self._object_type}, rssi={self._rssi})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._factory_id, "L")
        writer.write(self._object_type.value, "l")
        writer.write(self._rssi, "b")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        factory_id = reader.read("L")
        object_type = reader.read("l")
        rssi = reader.read("b")
        return cls(factory_id=factory_id, object_type=object_type, rssi=rssi)


class ImageImuData(Packet):
    __slots__ = (
        "_image_id",  # uint32
        "_line_2_number",  # uint8
        "_rate_x",  # float
        "_rate_y",  # float
        "_rate_z",  # float
    )

    def __init__(self, image_id=0, rate_x=0.0, rate_y=0.0, rate_z=0.0, line_2_number=0):
        super().__init__(PacketType.EVENT, packet_id=0xF4)
        self.image_id = image_id
        self.rate_x = rate_x
        self.rate_y = rate_y
        self.rate_z = rate_z
        self.line_2_number = line_2_number

    @property
    def image_id(self):
        return self._image_id

    @image_id.setter
    def image_id(self, value):
        self._image_id = validate_integer("image_id", value, 0, 4294967295)

    @property
    def rate_x(self):
        return self._rate_x

    @rate_x.setter
    def rate_x(self, value):
        self._rate_x = validate_float("rate_x", value)

    @property
    def rate_y(self):
        return self._rate_y

    @rate_y.setter
    def rate_y(self, value):
        self._rate_y = validate_float("rate_y", value)

    @property
    def rate_z(self):
        return self._rate_z

    @rate_z.setter
    def rate_z(self, value):
        self._rate_z = validate_float("rate_z", value)

    @property
    def line_2_number(self):
        return self._line_2_number

    @line_2_number.setter
    def line_2_number(self, value):
        self._line_2_number = validate_integer("line_2_number", value, 0, 255)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 1

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"image_id={self._image_id}, "
            f"rate_x={self._rate_x}, "
            f"rate_y={self._rate_y}, "
            f"rate_z={self._rate_z}, "
            f"line_2_number={self._line_2_number})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._image_id, "L")
        writer.write(self._rate_x, "f")
        writer.write(self._rate_y, "f")
        writer.write(self._rate_z, "f")
        writer.write(self._line_2_number, "B")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        image_id = reader.read("L")
        rate_x = reader.read("f")
        rate_y = reader.read("f")
        rate_z = reader.read("f")
        line_2_number = reader.read("B")
        return cls(
            image_id=image_id,
            rate_x=rate_x,
            rate_y=rate_y,
            rate_z=rate_z,
            line_2_number=line_2_number,
        )


class ObjectAccel(Packet):
    __slots__ = (
        "_accel_x",  # float
        "_accel_y",  # float
        "_accel_z",  # float
        "_object_id",  # uint32
        "_timestamp",  # uint32
    )

    def __init__(self, timestamp=0, object_id=0, accel_x=0.0, accel_y=0.0, accel_z=0.0):
        super().__init__(PacketType.EVENT, packet_id=0xF5)
        self.timestamp = timestamp
        self.object_id = object_id
        self.accel_x = accel_x
        self.accel_y = accel_y
        self.accel_z = accel_z

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, value):
        self._timestamp = validate_integer("timestamp", value, 0, 4294967295)

    @property
    def object_id(self):
        return self._object_id

    @object_id.setter
    def object_id(self, value):
        self._object_id = validate_integer("object_id", value, 0, 4294967295)

    @property
    def accel_x(self):
        return self._accel_x

    @accel_x.setter
    def accel_x(self, value):
        self._accel_x = validate_float("accel_x", value)

    @property
    def accel_y(self):
        return self._accel_y

    @accel_y.setter
    def accel_y(self, value):
        self._accel_y = validate_float("accel_y", value)

    @property
    def accel_z(self):
        return self._accel_z

    @accel_z.setter
    def accel_z(self, value):
        self._accel_z = validate_float("accel_z", value)

    def __len__(self):
        return 4 + 4 + 4 + 4 + 4

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"timestamp={self._timestamp}, "
            f"object_id={self._object_id}, "
            f"accel_x={self._accel_x}, "
            f"accel_y={self._accel_y}, "
            f"accel_z={self._accel_z})"
        )

    def to_bytes(self):
        writer = BinaryWriter()
        self.to_writer(writer)
        return writer.dumps()

    def to_writer(self, writer):
        writer.write(self._timestamp, "L")
        writer.write(self._object_id, "L")
        writer.write(self._accel_x, "f")
        writer.write(self._accel_y, "f")
        writer.write(self._accel_z, "f")

    @classmethod
    def from_bytes(cls, buffer):
        reader = BinaryReader(buffer)
        obj = cls.from_reader(reader)
        return obj

    @classmethod
    def from_reader(cls, reader):
        timestamp = reader.read("L")
        object_id = reader.read("L")
        accel_x = reader.read("f")
        accel_y = reader.read("f")
        accel_z = reader.read("f")
        return cls(
            timestamp=timestamp,
            object_id=object_id,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
        )


PACKETS_BY_ID = {
    0x03: LightStateCenter,  # 3
    0x04: CubeLights,  # 4
    0x05: SetPropSlot,  # 5
    0x08: StreamObjectAccel,  # 8
    0x0A: SetAccessoryDiscovery,  # 10
    0x0B: SetHeadLight,  # 11
    0x10: CubeId,  # 16
    0x11: LightStateSide,  # 17
    0x25: Enable,  # 37
    0x32: DriveWheels,  # 50
    0x33: TurnInPlaceAtSpeed,  # 51
    0x34: MoveLift,  # 52
    0x35: MoveHead,  # 53
    0x36: SetLiftHeight,  # 54
    0x37: SetHeadAngle,  # 55
    0x39: TurnInPlace,  # 57
    0x3B: StopAllMotors,  # 59
    0x3C: ClearPath,  # 60
    0x3D: AppendPathSegLine,  # 61
    0x3E: AppendPathSegArc,  # 62
    0x3F: AppendPathSegPointTurn,  # 63
    0x40: TrimPath,  # 64
    0x41: ExecutePath,  # 65
    0x45: SetOrigin,  # 69
    0x4B: SyncTime,  # 75
    0x4C: EnableCamera,  # 76
    0x57: SetCameraParams,  # 87
    0x58: StartMotorCalibration,  # 88
    0x60: EnableStopOnCliff,  # 96
    0x64: SetRobotVolume,  # 100
    0x66: EnableColorImages,  # 102
    0x81: NvStorageOp,  # 129
    0x8D: AbortAnimation,  # 141
    0x8E: OutputAudio,  # 142
    0x8F: OutputSilence,  # 143
    0x91: RecordHeading,  # 145
    0x92: TurnToRecordedHeading,  # 146
    0x93: AnimHead,  # 147
    0x94: AnimLift,  # 148
    0x97: DisplayImage,  # 151
    0x98: AnimBackpackLights,  # 152
    0x99: AnimBody,  # 153
    0x9A: EndAnimation,  # 154
    0x9B: StartAnimation,  # 155
    0x9F: EnableAnimationState,  # 159
    0xA9: ShutdownRobot,  # 169
    0xAE: WifiOff,  # 174
    0xAF: FirmwareUpdate,  # 175
    0xB0: DebugData,  # 176
    0xB4: ObjectMoved,  # 180
    0xB5: ObjectStoppedMoving,  # 181
    0xB6: ObjectTapped,  # 182
    0xB9: ObjectTapFiltered,  # 185
    0xC2: RobotDelocalized,  # 194
    0xC3: RobotPoked,  # 195
    0xC4: AcknowledgeAction,  # 196
    0xC6: PathFollowingEvent,  # 198
    0xC9: HardwareInfo,  # 201
    0xCA: AnimationStarted,  # 202
    0xCB: AnimationEnded,  # 203
    0xCD: NvStorageOpResult,  # 205
    0xCE: ObjectPowerLevel,  # 206
    0xD0: ObjectConnectionState,  # 208
    0xD1: MotorCalibration,  # 209
    0xD7: ObjectUpAxisChanged,  # 215
    0xDB: ButtonPressed,  # 219
    0xDD: FallingStarted,  # 221
    0xDE: FallingStopped,  # 222
    0xED: BodyInfo,  # 237
    0xEE: FirmwareSignature,  # 238
    0xEF: FirmwareUpdateResult,  # 239
    0xF0: RobotState,  # 240
    0xF1: AnimationState,  # 241
    0xF2: ImageChunk,  # 242
    0xF3: ObjectAvailable,  # 243
    0xF4: ImageImuData,  # 244
    0xF5: ObjectAccel,  # 245
}


PACKETS_BY_GROUP = {
    "anim": {
        0x8D,  # AbortAnimation
        0x91,  # RecordHeading
        0x92,  # TurnToRecordedHeading
        0x93,  # AnimHead
        0x94,  # AnimLift
        0x98,  # AnimBackpackLights
        0x99,  # AnimBody
        0x9A,  # EndAnimation
        0x9B,  # StartAnimation
        0xCA,  # AnimationStarted
        0xCB,  # AnimationEnded
    },
    "audio": {
        0x64,  # SetRobotVolume
        0x8E,  # OutputAudio
        0x8F,  # OutputSilence
    },
    "camera": {
        0x0B,  # SetHeadLight
        0x4C,  # EnableCamera
        0x57,  # SetCameraParams
        0x66,  # EnableColorImages
    },
    "debug": {
        0xB0,  # DebugData
    },
    "display": {
        0x97,  # DisplayImage
    },
    "firmware": {
        0xAF,  # FirmwareUpdate
        0xEF,  # FirmwareUpdateResult
    },
    "lights": {
        0x03,  # LightStateCenter
        0x11,  # LightStateSide
    },
    "localization": {
        0x45,  # SetOrigin
        0xC2,  # RobotDelocalized
        0xC3,  # RobotPoked
        0xDD,  # FallingStarted
        0xDE,  # FallingStopped
    },
    "motors": {
        0x32,  # DriveWheels
        0x33,  # TurnInPlaceAtSpeed
        0x34,  # MoveLift
        0x35,  # MoveHead
        0x36,  # SetLiftHeight
        0x37,  # SetHeadAngle
        0x39,  # TurnInPlace
        0x3B,  # StopAllMotors
        0x3C,  # ClearPath
        0x3D,  # AppendPathSegLine
        0x3E,  # AppendPathSegArc
        0x3F,  # AppendPathSegPointTurn
        0x40,  # TrimPath
        0x41,  # ExecutePath
        0x58,  # StartMotorCalibration
        0x60,  # EnableStopOnCliff
        0xC4,  # AcknowledgeAction
        0xC6,  # PathFollowingEvent
        0xD1,  # MotorCalibration
    },
    "nv": {
        0x81,  # NvStorageOp
        0xCD,  # NvStorageOpResult
    },
    "objects": {
        0x04,  # CubeLights
        0x05,  # SetPropSlot
        0x08,  # StreamObjectAccel
        0x0A,  # SetAccessoryDiscovery
        0x10,  # CubeId
        0xB4,  # ObjectMoved
        0xB5,  # ObjectStoppedMoving
        0xB6,  # ObjectTapped
        0xB9,  # ObjectTapFiltered
        0xCE,  # ObjectPowerLevel
        0xD0,  # ObjectConnectionState
        0xD7,  # ObjectUpAxisChanged
    },
    "state": {
        0xF0,  # RobotState
        0xF1,  # AnimationState
        0xF2,  # ImageChunk
        0xF3,  # ObjectAvailable
        0xF4,  # ImageImuData
        0xF5,  # ObjectAccel
    },
    "system": {
        0x25,  # Enable
        0x4B,  # SyncTime
        0x9F,  # EnableAnimationState
        0xA9,  # ShutdownRobot
        0xAE,  # WifiOff
        0xC9,  # HardwareInfo
        0xDB,  # ButtonPressed
        0xED,  # BodyInfo
        0xEE,  # FirmwareSignature
    },
}
