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
    "EvtAudioCompleted",
    "EvtBehaviorDone",
    "EvtChargerOOSChange",
    "EvtCliffDetectedChange",
    "EvtCubeDetected",
    "EvtCubeDetectorDebugImages",
    "EvtCubeLost",
    "EvtCubeObserved",
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


class EvtCubeDetectorDebugImages(Event):
    """
    Triggered when debug images from cube detection pipeline are available.

    Attributes:
        debug_images: Dict of {step_name: image_array} containing intermediate images
        detections: List of CubeDetection objects
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
        detections: List of CubeDetection objects
    """


class EvtCubeObserved(Event):
    """
    Triggered periodically while one or more cubes are visible.

    Attributes:
        detections: List of CubeDetection objects
    """


class EvtCubeLost(Event):
    """
    Triggered when a tracked cube is no longer visible.

    Attributes:
        cube_id: ID of the lost cube
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
        # Dispatch to handlers.
        handlers = []
        for i, handler in enumerate(self.dispatch_handlers[event]):
            if handler.one_shot:
                # Delete one-shot handlers prior to actual dispatch
                del self.dispatch_handlers[event][i]
            handlers.append(handler)
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
