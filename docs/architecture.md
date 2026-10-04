
PyCozmo-NG Architecture
====================


Overview
--------

PyCozmo-NG is designed as a multithreaded library.

It is organized in three layers with each higher layer building on the ones below it: 
- low-level connection layer
- client or SDK layer
- application layer

Each layer provides it's own API and can be used independently.

The following diagram illustrates the library architecture.

```
                                                                             ^
                                                                             |
                    +-----------------------------------------+              |
                    |                                         |              |
                    |                  Brain                  |              |
                    |                                         |              |
                    |                                         |              |
                    +-----+--------------+--------------+-----+              |
                          ^              |              ^                    |
                          |              |              |                    |
     +-----------+  +-----+-----+        |              |                    |  Application
     |           |  |           |        |              |                    |  Layer
     | Heartbeat |  |  Reaction |        |              |                    |
     |  Thread   |  |   Thread  |        |              |                    |
     |           |  |           |        |              |                    |
     +-----+-----+  +-----+-----+        |              |                    |
           |              ^              |  Commands    |  Events            |
           |              |              |              |                    |
           |             +-+             |              |                    |
           |             |-|  Event      |              |                    |
           |             |-|  Queue      |              |                    |
           |             +-+             |              |                    |
           |              ^              |              |                    |
           |              |              |              |                    v
           +------------->+  Reactions   |              |
                          |              v              |                    ^
                    +-----+--------------+--------------+-----+              |
                    |                                         |              |
                    |                  Client                 |              |
                    |                                         |              |
                    |                                         |              |
                    +-----+--------------+--------------+-----+              |
                          |              |              ^                    |  SDK
                          v              |              |                    |  Layer
                         +-+             |              |                    |
              Animation  |-|             |  Commands    |  Events            |
                Queue    |-|             |              |                    |
                         +-+             |              |                    |
                          |              |              |                    |
                          v              v              |                    |
+-----------+       +-----+-----+  +-----+--------------+-----+              |
|           |       |           |  |                          |              |
|   Face    |       | Animation |  |        Connection        |              v
| Generator +------>+  Thread   |  |          Thread          |
|           |       |           |  |                          |              ^
+-----------+       +-----+-----+  +-----+--------------+-----+              |
                          |              |              ^                    |
                          +------------->+              |                    |
                                         |              |                    |
                                         v              |                    |
                                        +-+            +-+                   |
                              Outgoing  |-|            |-|  Incoming         |
                               Message  |-|            |-|   Message         |
                                Queue   +-+            +-+    Queue          |
                                         |              ^                    |
                                         v              |                    |  Connection
                                   +-----+-----+  +-----+-----+              |  Layer
                                   |           |  |           |              |
                                   |   Send    |  |  Receive  |              |
                                   |  Thread   |  |  Thread   |              |
                                   |           |  |           |              |
                                   +---------+-+  +--+--------+              |
                                             |       ^                       |
                                             |       |                       |
                                             |       |                       |
                                             v       |                       |
                                           +-+-------+-+                     |
                                           |           |                     |
                                           |    UDP    |                     |
                                           |   Socket  |                     |
                                           |           |                     |
                                           +-----------+                     |
                                                                             v
```

Connection Layer
----------------

The connection layer implements the Cozmo communication protocol.

The receive thread reads Cozmo protocol frames, encapsulated in UDP datagrams, from the UDP socket. It maintains
a receive window for incoming packets and sends a stream of incoming packets in the correct order over the incoming
message queue to the connection thread.

The send thread reads a stream of outgoing packets from the outgoing message queue, builds Cozmo protocol frames
and sends them over the UDP socket. It maintains a send window and resends packets that are not acknowledged in time.

The connection thread reads a stream of incoming packets from the incoming message queue and dispatches them to
registered handler functions. It sends ping packets on a regular basis to maintain connection with the robot. 


Client Layer (SDK)
------------------

The client layer provides access to robot on-board functions.

It allows sending commands and registering handler function for incoming packets and events.

It performs:
- camera image reconstruction
- display image encoding
- audio encoding
- animation and audio playback
- procedural face generation 

The animation controller synchronizes animations, audio playback, and image display. It works as a separate thread
that aims to send images and audio to the robot at 30 frames per second. All on-board function of the robot are
synchronized to this framerate, including images, audio playback, backpack and cube LED animations.


Application Layer
-----------------

The application layer implements high-level off-board functions:
- reactions and behaviors
- personality engine
- computer vision (CV) camera image processing

Events from the client layer are converted to reactions. The reaction thread reads events from its incoming event
queue and handles them appropriately. Reactions normally trigger behaviors.

The heartbeat thread drives the personality engine and timers for activities and behaviors.


### Computer Vision Module

The CV module processes camera images from the robot to detect objects (cubes, faces, etc.) in the environment.

**Recommended Architecture for Cube Detection:**

The video processing and cube detection should be implemented at the **Application Layer** as a separate module:

1. **Vision Module** (`pycozmo_ng/vision.py`) - Core computer vision functionality
   - Cube detection algorithms
   - Image processing utilities
   - Object tracking
   - Can run in a separate thread for async processing

2. **Integration with Brain** - The Brain class should:
   - Subscribe to camera image events from the Client layer
   - Pass images to the vision module for processing
   - Convert detected objects into reactions/events
   - Trigger appropriate behaviors based on detections

3. **Event-driven Design** - New events should be added:
   - `EvtCubeDetected` - Fired when a cube is detected
   - `EvtCubeObserved` - Fired periodically while cube is visible
   - `EvtCubeLost` - Fired when a tracked cube is no longer visible

**Data Flow:**
```
Camera (Client Layer) -> EvtNewRawCameraImage -> Brain/Vision Thread -> 
Process Image -> Detect Cubes -> Generate EvtCubeDetected -> 
Reaction Queue -> Trigger Behavior
```

This design follows the existing architecture pattern where:
- Client layer handles low-level robot communication and camera image reconstruction
- Application layer (Brain + Vision) handles high-level processing
- Events are used to communicate between components
- Processing can be done asynchronously without blocking robot control

