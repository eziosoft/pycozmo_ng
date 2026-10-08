"""

Event declaration and dispatching.

"""

import collections
import threading
from collections.abc import Callable

from . import exception, robot

__all__ = [
    "STATUS_EVENTS",
    "Dispatcher",
    "Event",
    "EvtAnimationCompleted",
    "EvtAnnotatedCameraImage",
    "EvtArucoDetected",
    "EvtArucoLost",
    "EvtArucoObserved",
    "EvtAudioCompleted",
    "EvtBehaviorDone",
    "EvtChargerDetected",
    "EvtChargerLost",
    "EvtChargerOOSChange",
    "EvtChargerObserved",
    "EvtCliffDetectedChange",
    "EvtCubeDetected",
    "EvtCubeLost",
    "EvtCubeObserved",
    "EvtFaceDetected",
    "EvtFaceLost",
    "EvtFaceObserved",
    "EvtFloorProjectedImage",
    "EvtNewRawCameraImage",
    "EvtPacketReceived",
    "EvtReactionTrigger",
    "EvtRobotAnimBufferFullChange",
    "EvtRobotAnimatingChange",
    "EvtRobotAnimatingIdleChange",
    "EvtRobotBodyAccModeChange",
    "EvtRobotCarryingBlockChange",
    "EvtRobotChargingChange",
    "EvtRobotFallingChange",
    "EvtRobotFound",
    "EvtRobotHeadInPositionChange",
    "EvtRobotLiftInPositionChange",
    "EvtRobotMovingChange",
    "EvtRobotOnChargerChange",
    "EvtRobotOrientationChange",
    "EvtRobotPathingChange",
    "EvtRobotPickedUpChange",
    "EvtRobotPickingOrPlacingChange",
    "EvtRobotReady",
    "EvtRobotStateUpdated",
    "EvtRobotWheelsMovingChange",
    "Handler",
]


class Event:
    """Base class for events."""


class Handler:
    """Event handler class."""

    def __init__(self, f: Callable, one_shot: bool):
        self.f = f
        self.one_shot = one_shot


class EvtRobotFound(Event):
    """Triggered when the robot has been first connected."""


class EvtRobotReady(Event):
    """Triggered when the robot has been initialized and is ready for commands."""


class EvtPacketReceived(Event):
    """Triggered when a new packet has been received from the robot."""


class EvtNewRawCameraImage(Event):
    """Triggered when a new raw image is received from the robot's camera."""


class EvtAnnotatedCameraImage(Event):
    """
    Triggered when a new annotated camera image is available with cube detections drawn.

    Attributes:
        image: PIL Image with detection boxes and labels
        detections: List of CubeDetection objects
    """


class EvtFloorProjectedImage(Event):
    """
    Triggered when a new floor-projected (bird's-eye) image is available.

    Attributes:
        image: numpy array (BGR) of the floor-projected camera frame
        occupancy_grid: int8 array — 0 occupied, 1 free, -1 unknown
    """


class EvtRobotMovingChange(Event):
    pass


class EvtRobotCarryingBlockChange(Event):
    pass


class EvtRobotPickingOrPlacingChange(Event):
    pass


class EvtRobotPickedUpChange(Event):
    pass


class EvtRobotBodyAccModeChange(Event):
    pass


class EvtRobotFallingChange(Event):
    pass


class EvtRobotAnimatingChange(Event):
    pass


class EvtRobotPathingChange(Event):
    pass


class EvtRobotLiftInPositionChange(Event):
    pass


class EvtRobotHeadInPositionChange(Event):
    pass


class EvtRobotAnimBufferFullChange(Event):
    pass


class EvtRobotAnimatingIdleChange(Event):
    pass


class EvtRobotOnChargerChange(Event):
    pass


class EvtRobotChargingChange(Event):
    pass


class EvtCliffDetectedChange(Event):
    pass


class EvtRobotWheelsMovingChange(Event):
    pass


class EvtChargerOOSChange(Event):
    pass


STATUS_EVENTS = {
    robot.RobotStatusFlag.IS_MOVING: EvtRobotMovingChange,
    robot.RobotStatusFlag.IS_CARRYING_BLOCK: EvtRobotCarryingBlockChange,
    robot.RobotStatusFlag.IS_PICKING_OR_PLACING: EvtRobotPickingOrPlacingChange,
    robot.RobotStatusFlag.IS_PICKED_UP: EvtRobotPickedUpChange,
    robot.RobotStatusFlag.IS_BODY_ACC_MODE: EvtRobotBodyAccModeChange,
    robot.RobotStatusFlag.IS_FALLING: EvtRobotFallingChange,
    robot.RobotStatusFlag.IS_ANIMATING: EvtRobotAnimatingChange,
    robot.RobotStatusFlag.IS_PATHING: EvtRobotPathingChange,
    robot.RobotStatusFlag.LIFT_IN_POS: EvtRobotLiftInPositionChange,
    robot.RobotStatusFlag.HEAD_IN_POS: EvtRobotHeadInPositionChange,
    robot.RobotStatusFlag.IS_ANIM_BUFFER_FULL: EvtRobotAnimBufferFullChange,
    robot.RobotStatusFlag.IS_ANIMATING_IDLE: EvtRobotAnimatingChange,
    robot.RobotStatusFlag.IS_ON_CHARGER: EvtRobotOnChargerChange,
    robot.RobotStatusFlag.IS_CHARGING: EvtRobotChargingChange,
    robot.RobotStatusFlag.CLIFF_DETECTED: EvtCliffDetectedChange,
    robot.RobotStatusFlag.ARE_WHEELS_MOVING: EvtRobotWheelsMovingChange,
    robot.RobotStatusFlag.IS_CHARGER_OOS: EvtChargerOOSChange,
}


class EvtRobotStateUpdated(Event):
    """Triggered when a new robot state is received."""


class EvtRobotOrientationChange(Event):
    """Triggered when the robot orientation changes."""


class EvtAudioCompleted(Event):
    pass


class EvtAnimationCompleted(Event):
    pass


class EvtReactionTrigger(Event):
    pass


class EvtBehaviorDone(Event):
    pass


class EvtCubeDetected(Event):
    """
    Triggered when a cube is first detected in the camera image.

    Attributes:
        detections: List of CubeDetection objects (each carries
            ``cube_index``, the world-map index 0-2)
    """


class EvtCubeObserved(Event):
    """
    Triggered periodically while one or more cubes are visible.

    Attributes:
        detections: List of CubeDetection objects (each carries
            ``cube_index``, the world-map index 0-2)
    """


class EvtCubeLost(Event):
    """
    Triggered when a tracked cube is no longer visible.

    Attributes:
        cube_id: Public ID of the lost cube (1-3)
    """


class EvtFaceDetected(Event):
    """
    Triggered when one or more faces are first detected.

    Attributes:
        faces: List of FaceDetection objects (each carries the ``face_id``
            assigned for this presence run, 1+)
    """


class EvtFaceObserved(Event):
    """
    Triggered periodically while faces are visible.

    Attributes:
        faces: List of FaceDetection objects
    """


class EvtFaceLost(Event):
    """
    Triggered when a tracked face is no longer visible.

    Attributes:
        face_id: ID of the lost face presence run (1+; never reused)
    """


class EvtChargerDetected(Event):
    """
    Triggered when the charger marker is first detected.

    Attributes:
        detections: List of ChargerDetection objects (the charger is a
            single static object — no per-object ID, COZMO-65)
    """


class EvtChargerObserved(Event):
    """
    Triggered periodically while the charger marker is visible.

    Attributes:
        detections: List of ChargerDetection objects
    """


class EvtChargerLost(Event):
    """
    Triggered when the charger marker is no longer visible.
    """


class EvtArucoDetected(Event):
    """
    Triggered when one or more ArUco markers enter the frame.

    Attributes:
        detections: List of ArucoDetection objects — newly visible markers
            only (marker ID is the identity; diffed against the last frame)
    """


class EvtArucoObserved(Event):
    """
    Triggered periodically while ArUco markers are visible.

    Attributes:
        detections: List of ArucoDetection objects (all markers this frame)
    """


class EvtArucoLost(Event):
    """
    Triggered when ArUco markers leave the frame.

    Attributes:
        marker_ids: List of marker IDs no longer visible
    """


class Dispatcher:
    """Event dispatcher class."""

    def __init__(self):
        super().__init__()
        self.dispatch_children = []
        self.dispatch_handlers = collections.defaultdict(list)

    def add_child_dispatcher(self, child):
        self.dispatch_children.append(child)

    def del_child_dispatcher(self, child):
        try:
            self.dispatch_children.remove(child)
        except ValueError:
            pass

    def add_handler(self, event, f, one_shot=False):
        handler = Handler(f, one_shot=one_shot)
        self.dispatch_handlers[event].append(handler)
        return handler

    def del_handler(self, event, handler):
        for i, _handler in enumerate(self.dispatch_handlers[event]):
            if _handler == handler:
                del self.dispatch_handlers[event][i]
                return

    def del_all_handlers(self):
        self.dispatch_handlers = collections.defaultdict(list)

    def dispatch(self, event, *args, **kwargs):
        # Snapshot the handler list before iterating: handlers may be added or
        # removed from other threads (e.g. Client.go_to_pose registering its
        # path-completion waiter) while a packet is being dispatched.  The old
        # in-place iteration/deletion could raise "list changed size during
        # iteration", silently dropping the packet and leaving callers blocked
        # forever (COZMO-53).
        handlers = list(self.dispatch_handlers[event])
        if any(h.one_shot for h in handlers):
            self.dispatch_handlers[event] = [h for h in handlers if not h.one_shot]
        for handler in handlers:
            handler.f(*args, **kwargs)
        # Dispatch to child dispatchers.
        for child in self.dispatch_children:
            child.dispatch(event, *args, **kwargs)

    def wait_for(self, evt, timeout: float | None = None) -> None:
        e = threading.Event()
        self.add_handler(evt, lambda *args: e.set(), one_shot=True)
        if not e.wait(timeout):
            raise exception.Timeout("Failed to receive event in time.")
