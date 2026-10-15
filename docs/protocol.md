Cozmo Protocol
==============

Full reference for the wire protocol between the Cozmo app (the **engine**) and the
robot (the **body**), plus the cube/object subsystem.

**Scope.** Two firmware revisions are described here, and they disagree — that
disagreement is the point of this document:

| Firmware | Released | What it is |
|---|---|---|
| **2381** | 2019-01-08 | the revision `pycozmo_ng/protocol/protocol_declaration.py` implements |
| **2458** | 2025-07-07 | the revision a current Cozmo runs |

Behaviour described below as *verified* was confirmed against a live robot.

> **The protocol changed.** See [Versioning](#versioning) and
[Cubes and objects](#cubes-and-objects) for the two consequences that matter most.


Overview
--------

The Cozmo protocol is a UDP-based variant of the
[selective repeat automatic-repeat request (ARQ) protocol](https://en.wikipedia.org/wiki/Selective_Repeat_ARQ).

The engine acts as a client and the robot acts as a server. The two exchange **frames**,
encapsulated in UDP packets. Each frame can contain 0, 1, or more **packets**.

Every message and its payload fields are listed in the two message tables below: one for
each direction, with the fields in order.


Network setup
-------------

The robot acts as a Wi-Fi access point, always using an SSID of the form `Cozmo_XXXXXX`
(upper-case hex). It runs a DHCP server and assigns clients an address in
`172.31.1.0/24`.

The robot is always `172.31.1.1` and expects UDP on port **5551**. It only accepts
packets from `172.31.1.0/24`. The engine only accepts packets from `172.31.1.1`.

```
+--------------------+                        +--------------------+
|     Cozmo app      |                        |       Cozmo        |
|      "engine"      |        UDP/Wi-Fi       |      "robot"       |
|    Wi-Fi client    | ---------------------> |      Wi-Fi AP      |
|     UDP client     |                        |     UDP Server     |
+--------------------+                        +--------------------+
    172.31.1.0/24                                 172.31.1.1:5551
```


Frames
------

```
Field           Length      Description
---------------------------------------------------------------------------------
id              7           Always "COZ\x03RE\x01"
type            1           Frame type
first_seq       2           First packet sequence number in the frame or 0
seq             2           Last packet sequence number in the frame or 0
ack             2           Peer packet sequence number acknowledgement
packets         -           0 or more encapsulated packets
```

Frame types:

```
Type            Source      Description
---------------------------------------------------------------------------------
0x01            engine      Reset
0x02            robot       Reset ACK
0x03            engine      Disconnect
0x04            engine      Engine packet - single
0x07            engine      Engine packets - zero or more
0x09            robot       Robot packets - zero or more
0x0b            engine      Out-of-band engine ping
```

`COZ\x03RE\x01` and the frame types above describe **2381**. Firmware 2458 uses a
different transport — see [Transport in 2458](#transport-in-2458).


Packets
-------

```
Type    OOB     Source      Description
---------------------------------------------------------------------------------
0x02    n       robot       Connect
0x03    n       engine      Disconnect
0x04    n       both        Command
0x05    y       robot       Event
0x0a    y       engine      Keyframe
0x0b    y       engine      Ping
```

Out-of-band packets do not get assigned sequence IDs.

Packet content is firmware-version specific. Commands and events are identified by an
8-bit ID: **0x00–0xAF are engine→robot, 0xB0+ are robot→engine**. Out-of-band updates use
`0xF0–0xFF`; they are not sequenced and are not retransmitted.


Connection establishment
------------------------

The engine sends a reset frame (0x01) with `first_seq`/`seq` set to 1 and `ack` 0.

The robot replies with a robot packets frame (0x09), `first_seq`/`seq` = 1, `ack` = 1,
containing a connect packet (0x02). That establishes the connection.

The engine keeps the connection alive with periodic ping frames (0x0b); the robot echoes
the ping back so the engine can measure RTT. If the robot sees no ping for >5 s it
disconnects and shows "COZMO 01".

The engine can close gracefully with either a disconnect frame (0x03) or an engine
packets frame (0x07) carrying a disconnect packet (0x03).


Versioning
----------

Each firmware image carries a JSON manifest with a **hash of its message definitions**
per direction. These are the authoritative statement of "did the protocol change":

| Field | firmware **2381** (pycozmo) | firmware **2458** (current) |
|---|---|---|
| `messageEngineToRobotHash` | `9e4a965ace4e09d86997b87ba14235d5` | `fedb4b12f1b5b45456aec1629cd0b8cc` |
| `messageRobotToEngineHash` | `a259247f16231db440957215baba12ab` | `5a7211095fc4961407e96b9126658317` |
| `wifiSig` | `69ca03352e42143d340f0f7fac02ed8ff96ef10b` | `6e81ae00a5b7e046c9d8924ccb664bcba7ca6957` |
| `rtipSig` | `36574986d76144a70e9252ab633be4617a4bc661` | `2891c2f25e3afd2c3be6750f40c1f78b2508de88` |
| `bodySig` | `695b59eff43664acd1a5a956d08c682b3f8bd2c8` | `32d1903533faf12411dac6ad6d393660ab489d68` |

**Both message hashes differ**, so the message set changed in both directions. The body
and Wi-Fi signatures differ too, so the two revisions are not interchangeable.

2458 also versions messages individually rather than only per direction.

A robot can run different firmware revisions, so which one it runs must be checked, not
assumed. `pycozmo_ng` pins `FIRMWARE_VERSION = 2381`.


Transport in 2458
-----------------

The framing above is the 2381 transport. In 2458 messages are carried by a
reliable-transport layer instead:

- messages carry a reliable-message header, and the transport **header prefix is
  settable** rather than the fixed literal above, with the prefix validated on receive
- there is a protocol message to negotiate the transport mode (tag `0x89`, `setRTTO`)

Corollary: do not assume `COZ\x03RE\x01` on a 2458 robot.


Data types
----------

Structs and enums referenced by the messages below.

### Structs

- **`LightState`** — on_color:u16, off_color:u16, on_frames:u8, off_frames:u8, transition_on_frames:u8, transition_off_frames:u8, offset:i16
- **`PathSegmentSpeed`** — speed_mmps:f32, accel_mmps2:f32, decel_mmps2:f32

### Enums

- **`BodyColor`** — `UNKNOWN`=-1, `WHITE_v10`=0, `RESERVED`=1, `WHITE_v15`=2, `CE_LM_v15`=3, `LE_BL_v16`=4, `DEV`=5
- **`NvEntryTag`** — `NVEntry_Invalid`=0xFFFFFFFF, `NVEntry_GameSkillLevels`=0x180000, `NVEntry_OnboardingData`=0x181000, `NVEntry_GameUnlocks`=0x182000, `NVEntry_FaceEnrollData`=0x183000, `NVEntry_FaceAlbumData`=0x184000, `NVEntry_NurtureGameData`=0x194000, `NVEntry_InventoryData`=0x195000, `NVEntry_LabAssignments`=0x196000, `NVEntry_SavedCubeIDs`=0x197000, `NVEntry_NEXT_SLOT`=0x198000, `NVEntry_FACTORY_RESERVED1`=0x1C0000, `NVEntry_FACTORY_RESERVED2`=0x1DE000, `NVEntry_BirthCertificate`=0x80000000, `NVEntry_CameraCalib`=0x80000001, `NVEntry_ToolCodeInfo`=0x80000002, `NVEntry_CalibPose`=0x80000003, `NVEntry_CalibMetaInfo`=0x80000004, `NVEntry_ObservedCubePose`=0x80000005, `NVEntry_IMUInfo`=0x80000006, `NVEntry_CliffValOnDrop`=0x80000007, `NVEntry_CliffValOnGround`=0x80000008, `NVEntry_PlaypenTestResults`=0x80000010, `NVEntry_FactoryLock`=0x80000011, `NVEntry_VersionMagic`=0x80000012, `NVEntry_CalibImage1`=0x80010000, `NVEntry_CalibImage2`=0x80020000, `NVEntry_CalibImage3`=0x80030000, `NVEntry_CalibImage4`=0x80040000, `NVEntry_CalibImage5`=0x80050000, `NVEntry_CalibImage6`=0x80060000, `NVEntry_ToolCodeImageLeft`=0x80100000, `NVEntry_ToolCodeImageRight`=0x80110000, `NVEntry_PrePlaypenResults`=0xC0000000, `NVEntry_PrePlaypenCentroids`=0xC0000001, `NVEntry_IMUAverages`=0xC0000004, `NVEntry_FactoryBaseTag`=0xDE000, `NVEntry_FactoryBaseTagWithBCOffset`=0xDE030
- **`NvOperation`** — `NVOP_READ`=0, `NVOP_WRITE`=1, `NVOP_ERASE`=2, `NVOP_WIPEALL`=3
- **`NvResult`** — `NV_OKAY`=0, `NV_SCHEDULED`=1, `NV_NO_DO`=2, `NV_MORE`=3, `NV_UNKNOWN_4`=4, `NV_UNKNOWN_5`=5, `NV_UNKNOWN_6`=6, `NV_UNKNOWN_7`=7, `NV_UNKNOWN_8`=8, `NV_NOT_FOUND`=-1, `NV_NO_ROOM`=-2, `NV_ERROR`=-3, `NV_TIMEOUT`=-4, `NV_BUSY`=-5, `NV_BAD_ARGS`=-6, `NV_NO_MEM`=-7, `NV_LOOP`=-8, `NV_CORRUPT`=-9
- **`UpAxis`** — `XNegative`=0, `XPositive`=1, `YNegative`=2, `YPositive`=3, `ZNegative`=4, `ZPositive`=5, `NumAxes`=6, `UnknownAxis`=7
- **`ObjectType`** — `InvalidObject`=-1, `UnknownObject`=0, `Block_LIGHTCUBE1`=1, `Block_LIGHTCUBE2`=2, `Block_LIGHTCUBE3`=3, `Block_LIGHTCUBE_GHOST`=4, `FlatMat_GEARS_4x4`=5, `FlatMat_LETTERS_4x4`=6, `FlatMat_ANKI_LOGO_8BIT`=7, `FlatMat_LAVA_PLAYTEST`=8, `Platform_LARGE`=9, `Bridge_LONG`=10, `Bridge_SHORT`=11, `Ramp_Basic`=12, `Charger_Basic`=13, `ProxObstacle`=14, `CliffDetection`=15, `CollisionObstacle`=16, `CustomType00`=17, `CustomType01`=18, `CustomType02`=19, `CustomType03`=20, `CustomType04`=21, `CustomType05`=22, `CustomType06`=23, `CustomType07`=24, `CustomType08`=25, `CustomType09`=26, `CustomType10`=27, `CustomType11`=28, `CustomType12`=29, `CustomType13`=30, `CustomType14`=31, `CustomType15`=32, `CustomType16`=33, `CustomType17`=34, `CustomType18`=35, `CustomType19`=36, `CustomFixedObstacle`=37
- **`ImageEncoding`** — `NoneImageEncoding`=0, `RawGray`=1, `RawRGB`=2, `YUYV`=3, `BAYER`=4, `JPEGGray`=5, `JPEGColor`=6, `JPEGColorHalfWidth`=7, `JPEGMinimizedGray`=8, `JPEGMinimizedColor`=9
- **`ImageResolution`** — `VerificationSnapshot`=0, `QQQQVGA`=1, `QQQVGA`=2, `QQVGA`=3, `QVGA`=4, `CVGA`=5, `VGA`=6, `SVGA`=7, `XGA`=8, `SXGA`=9, `UXGA`=10, `QXGA`=11, `QUXGA`=12, `ImageResolutionCount`=13, `ImageResolutionNone`=14
- **`ImageSendMode`** — `Off`=0, `Stream`=1, `SingleShot`=2
- **`MotorID`** — `MOTOR_LEFT_WHEEL`=0, `MOTOR_RIGHT_WHEEL`=1, `MOTOR_LIFT`=2, `MOTOR_HEAD`=3
- **`PathEventType`** — `PATH_STARTED`=0, `PATH_INTERRUPTED`=1, `PATH_COMPLETED`=2


Engine → robot commands
-----------------------

105 live IDs in `0x00–0xAF`. `bytes` is the payload length in bytes,
excluding the 1-byte message ID; `variable` marks a message that carries a
variable-length field. `payload` gives the fields in order — `name:type`, with
a non-zero default after `=` — and is shown where the layout is known.

| ID | message | bytes | pycozmo class | payload |
|---|---|---|---|---|
| `0x01` | `adjustTimestamp` | 4 | — | — |
| `0x02` | `setBackpackLayer` | 1 | — | — |
| `0x03` | `setBackpackLightsMiddle` | variable | `LightStateCenter` | states:LightState[3], unknown:u8 |
| `0x04` | `setCubeLights` | variable | `CubeLights` | states:LightState[4] |
| `0x05` | `setPropSlot` | 5 | `SetPropSlot` | factory_id:u32, slot:u8 |
| `0x06` | `killBodyCode` | 0 | — | — |
| `0x07` | `setBodyRadioMode` | 2 | — | — |
| `0x08` | `streamObjectAccel` | 5 | `StreamObjectAccel` | object_id:u32, enable:bool |
| `0x0a` | `setAccessoryDiscovery` | 1 | `SetAccessoryDiscovery` | enable:bool |
| `0x0b` | `setHeadlight` | 1 | `SetHeadLight` | enable:bool |
| `0x0c` | `setCubeGamma` | 1 | — | — |
| `0x0d` | `bodyEnterOTA` | 0 | — | — |
| `0x0e` | `readBodyStorage` | 2 | — | — |
| `0x0f` | `writeBodyStorage` | variable | — | — |
| `0x10` | `setCubeID` | 5 | `CubeId` | object_id:u32, rotation_period_frames:u8 |
| `0x11` | `setBackpackLightsTurnSignals` | variable | `LightStateSide` | states:LightState[2], unknown:u8 |
| `0x12` | `sendDTMCommand` | 16 | — | — |
| `0x24` | `getBodySerialNumber` | 0 | — | — |
| `0x25` | `getMfgInfo` | 0 | `Enable` | — |
| `0x26` | `bleSendData` | 17 | — | — |
| `0x27` | `bleDisconnect` | 4 | — | — |
| `0x30` | `enterRecoveryMode` | 1 | — | — |
| `0x32` | `drive` | 16 | `DriveWheels` | lwheel_speed_mmps:f32, rwheel_speed_mmps:f32, lwheel_accel_mmps2:f32, rwheel_accel_mmps2:f32 |
| `0x33` | `driveCurvature` | 10 | `TurnInPlaceAtSpeed` | wheel_speed_mmps:f32, wheel_accel_mmps2:f32, direction:i16 |
| `0x34` | `moveLift` | 4 | `MoveLift` | speed_rad_per_sec:f32 |
| `0x35` | `moveHead` | 4 | `MoveHead` | speed_rad_per_sec:f32 |
| `0x36` | `liftHeight` | 17 | `SetLiftHeight` | height_mm:f32, max_speed_rad_per_sec:f32=3.0, accel_rad_per_sec2:f32=20.0, duration_sec:f32, action_id:u8 |
| `0x37` | `headAngle` | 17 | `SetHeadAngle` | angle_rad:f32, max_speed_rad_per_sec:f32=15.0, accel_rad_per_sec2:f32=20.0, duration_sec:f32, action_id:u8 |
| `0x38` | `headAngleUpdate` | 4 | — | — |
| `0x39` | `setBodyAngle` | 20 | `TurnInPlace` | angle_rad:f32, speed_rad_per_sec:f32, accel_rad_per_sec2:f32, angle_tolerance_rad:f32, unknown4:u8, unknown5:u8, is_absolute:bool, action_id:u8 |
| `0x3a` | `turnInPlaceAtSpeed` | 8 | — | — |
| `0x3b` | `stop` | 0 | `StopAllMotors` | — |
| `0x3c` | `clearPath` | 2 | `ClearPath` | unknown:u16 |
| `0x3d` | `appendPathSegLine` | 28 | `AppendPathSegLine` | from_x:f32, from_y:f32, to_x:f32, to_y:f32, speed_mmps:f32, accel_mmps2:f32, decel_mmps2:f32 |
| `0x3e` | `appendPathSegArc` | 32 | `AppendPathSegArc` | center_x:f32, center_y:f32, radius_mm:f32, start_angle_rad:f32, sweep_rad:f32, speed_mmps:f32, accel_mmps2:f32, decel_mmps2:f32 |
| `0x3f` | `appendPathSegPointTurn` | 29 | `AppendPathSegPointTurn` | x:f32, y:f32, angle_rad:f32, angle_tolerance_rad:f32, speed_mmps:f32, accel_mmps2:f32, decel_mmps2:f32, unknown:bool |
| `0x40` | `trimPath` | 2 | `TrimPath` | head:u8, tail:u8 |
| `0x41` | `executePath` | 3 | `ExecutePath` | event_id:u16, unknown:bool |
| `0x42` | `dockWithObject` | 21 | — | — |
| `0x43` | `abortDocking` | 0 | — | — |
| `0x44` | `placeObjectOnGround` | 25 | — | — |
| `0x45` | `absLocalizationUpdate` | 24 | `SetOrigin` | unknown0:u32, pose_frame_id:u32, pose_origin_id:u32=1, pose_x:f32, pose_y:f32, unknown5:u32=0x80000000 |
| `0x46` | `startControllerTestMode` | 13 | — | — |
| `0x47` | `setControllerGains` | 17 | — | — |
| `0x48` | `dockingErrorSignal` | 22 | — | — |
| `0x49` | `setCarryState` | 1 | — | — |
| `0x4a` | `imuRequest` | 4 | — | — |
| `0x4b` | `syncTime` | 8 | `SyncTime` | timestamp:u32, unknown:u32 |
| `0x4c` | `imageRequest` | 2 | `EnableCamera` | image_send_mode:ImageSendMode(i8)=1, image_resolution:ImageResolution(i8)=4 |
| `0x4d` | `flashObjectIDs` | 4 | — | — |
| `0x4e` | `setObjectBeingCarried` | 5 | — | — |
| `0x4f` | `checkLiftLoad` | 0 | — | — |
| `0x50` | `enableMotorPower` | 2 | — | — |
| `0x51` | `setMotionModelParams` | 4 | — | — |
| `0x52` | `enterSleepMode` | 0 | — | — |
| `0x53` | `powerState` | 17 | — | — |
| `0x54` | `setCliffDetectThreshold` | 2 | — | — |
| `0x55` | `enableReadToolCodeMode` | 9 | — | — |
| `0x57` | `setCameraParams` | 7 | `SetCameraParams` | gain:f32, exposure_ms:u16, auto_exposure_enabled:bool |
| `0x58` | `startMotorCalibration` | 2 | `StartMotorCalibration` | head:bool, lift:bool |
| `0x59` | `rollActionParams` | 20 | — | — |
| `0x5a` | `cameraFOVInfo` | 8 | — | — |
| `0x60` | `enableStopOnCliff` | 1 | `EnableStopOnCliff` | enable:bool |
| `0x61` | `forceDelocalizeSimulatedRobot` | 0 | — | — |
| `0x62` | `objectConnectionStateToRobot` | 13 | — | — |
| `0x63` | `enableBraceWhenFalling` | 1 | — | — |
| `0x64` | `setAudioVolume` | 2 | `SetRobotVolume` | level:u16 |
| `0x65` | `generateTestTone` | 0 | — | — |
| `0x66` | `enableColorImages` | 1 | `EnableColorImages` | enable:bool |
| `0x80` | `requestCrashReports` | 4 | — | — |
| `0x81` | `commandNV` | variable | `NvStorageOp` | tag:NvEntryTag(u32)=0xFFFFFFFF, length:i32, op:NvOperation(u8), unknown:u8, data:u8[] |
| `0x82` | `enableWiFiTelemetry` | 0 | — | — |
| `0x86` | `bleConnectionState` | 4 | — | — |
| `0x87` | `bleDataReceived` | 17 | — | — |
| `0x89` | `setRTTO` | 2 | — | — |
| `0x8d` | `abortAnimation` | 0 | `AbortAnimation` | — |
| `0x8e` | `animAudioSample` | 744 | `OutputAudio` | samples:u8[744] |
| `0x8f` | `animAudioSilence` | 0 | `OutputSilence` | — |
| `0x91` | `animRecordHeading` | 0 | `RecordHeading` | — |
| `0x92` | `animTurnToRecordedHeading` | 13 | `TurnToRecordedHeading` | unknown:i16[6], unknown2:u8 |
| `0x93` | `animHeadAngle` | 3 | `AnimHead` | duration_ms:u8, variability_deg:i8, angle_deg:i8 |
| `0x94` | `animLiftHeight` | 3 | `AnimLift` | duration_ms:u8, variability_mm:u8, height_mm:u8 |
| `0x96` | `animEventToRTIP` | 2 | — | — |
| `0x97` | `animFaceImage` | variable | `DisplayImage` | image:u8[] |
| `0x98` | `animBackpackLights` | 10 | `AnimBackpackLights` | colors:u16[5] |
| `0x99` | `animBodyMotion` | 4 | `AnimBody` | speed:i16, unknown:i16 |
| `0x9a` | `animEndOfAnimation` | 0 | `EndAnimation` | — |
| `0x9b` | `animStartOfAnimation` | 1 | `StartAnimation` | anim_id:u8 |
| `0x9d` | `disableAnimTracks` | 1 | — | — |
| `0x9e` | `enableAnimTracks` | 1 | — | — |
| `0x9f` | `initAnimController` | 0 | `EnableAnimationState` | — |
| `0xa0` | `appRunID` | 16 | — | — |
| `0xa1` | `testState` | 49 | — | — |
| `0xa2` | `enterTestMode` | 8 | — | — |
| `0xa3` | `oledDisplayNumber` | 7 | — | — |
| `0xa4` | `bodyStorageContents` | variable | — | — |
| `0xa5` | `bodySerialNum` | 4 | — | — |
| `0xa9` | `shutdownRobot` | 32 | `ShutdownRobot` | unknown:u8[32] |
| `0xaa` | `appConCfgString` | 17 | — | — |
| `0xab` | `appConCfgFlags` | 17 | — | — |
| `0xac` | `appConCfgIPInfo` | 13 | — | — |
| `0xad` | `appConGetRobotIP` | 1 | — | — |
| `0xae` | `wifiOff` | 1 | `WifiOff` | enable:bool |
| `0xaf` | `otaWrite` | 1026 | `FirmwareUpdate` | chunk_id:u16, data:u8[1024] |
| `0xd5` | `animEvent` | 1 | — | — |


Robot → engine events
---------------------

56 live IDs in `0xB0–0xF5`, same conventions. Out-of-band messages
(`0xF0–0xFF` in the older scheme) are not sequenced and not retransmitted.

| ID | message | bytes | pycozmo class | payload |
|---|---|---|---|---|
| `0xb0` | `trace` | variable | `DebugData` | format_id:u16, unused:u16, name_id:u16, level:i8, args:u32[] |
| `0xb1` | `printText` | variable | — | — |
| `0xb2` | `mainCycleTimeError` | 16 | — | — |
| `0xb3` | `goalPose` | variable | — | — |
| `0xb4` | `activeObjectMoved` | variable | `ObjectMoved` | timestamp:u32, object_id:u32, active_accel_x:f32, active_accel_y:f32, active_accel_z:f32, axis_of_accel:UpAxis(u8)=7 |
| `0xb5` | `activeObjectStopped` | 8 | `ObjectStoppedMoving` | timestamp:u32, object_id:u32 |
| `0xb6` | `activeObjectTapped` | 12 | `ObjectTapped` | timestamp:u32, object_id:u32, num_taps:u8, tap_time:u8, tap_neg:i8, tap_pos:i8 |
| `0xb7` | `dataDump` | variable | — | — |
| `0xb8` | `pickAndPlaceResult` | 7 | — | — |
| `0xb9` | `objectTappedFiltered` | 10 | `ObjectTapFiltered` | timestamp:u32, object_id:u32, time:u8, intensity:u8 |
| `0xba` | `rampTraverseStarted` | 4 | — | — |
| `0xbb` | `rampTraverseCompleted` | 5 | — | — |
| `0xbc` | `bridgeTraverseStarted` | 4 | — | — |
| `0xbd` | `bridgeTraverseCompleted` | 5 | — | — |
| `0xbe` | `timeProfStat` | variable | — | — |
| `0xbf` | `imuDataChunk` | 195 | — | — |
| `0xc0` | `cliffEvent` | 6 | — | — |
| `0xc1` | `potentialCliff` | 0 | — | — |
| `0xc2` | `syncTimeAck` | 0 | `RobotDelocalized` | — |
| `0xc3` | `robotPoked` | 0 | `RobotPoked` | — |
| `0xc4` | `motorActionAck` | 1 | `AcknowledgeAction` | action_id:u8 |
| `0xc5` | `movingLiftPostDock` | 1 | — | — |
| `0xc6` | `pathFollowingEvent` | 3 | `PathFollowingEvent` | event_id:u16, event_type:PathEventType(u8) |
| `0xc7` | `imuRawDataChunk` | 14 | — | — |
| `0xc8` | `defaultCameraParams` | 29 | — | — |
| `0xc9` | `robotAvailable` | 6 | `HardwareInfo` | serial_number_head:u32, unknown1:u8, unknown2:u8 |
| `0xca` | `animStarted` | 1 | `AnimationStarted` | anim_id:u8 |
| `0xcb` | `animEnded` | 1 | `AnimationEnded` | anim_id:u8 |
| `0xcd` | `nvOpResult` | variable | `NvStorageOpResult` | tag:NvEntryTag(u32)=0xFFFFFFFF, length:i32, op:NvOperation(u8), result:NvResult(i8), data:u8[] |
| `0xce` | `objectPowerLevel` | 9 | `ObjectPowerLevel` | object_id:u32, missed_packets:u32, battery_level:u8 |
| `0xcf` | `crashReport` | variable | — | — |
| `0xd0` | `activeObjectConnectionState` | 13 | `ObjectConnectionState` | object_id:u32, factory_id:u32, object_type:ObjectType(i32)=-1, connected:bool |
| `0xd1` | `motorCalibration` | 3 | `MotorCalibration` | motor_id:MotorID(u8), calib_started:bool, auto_started:bool |
| `0xd2` | `factoryFirmwareVersion` | 44 | — | — |
| `0xd3` | `dockingStatus` | 5 | — | — |
| `0xd4` | `robotStopped` | 1 | — | — |
| `0xd5` | `animEvent` | 6 | — | — |
| `0xd6` | `factoryTestParam` | 4 | — | — |
| `0xd7` | `activeObjectUpAxisChanged` | 9 | `ObjectUpAxisChanged` | timestamp:u32, object_id:u32, axis:UpAxis(u8)=7 |
| `0xd8` | `motorAutoEnabled` | 2 | — | — |
| `0xd9` | `robotError` | 5 | — | — |
| `0xda` | `liftLoad` | 1 | — | — |
| `0xdb` | `backpackButton` | 1 | `ButtonPressed` | pressed:bool |
| `0xdc` | `imuTemperature` | 4 | — | — |
| `0xdd` | `fallingStarted` | 4 | `FallingStarted` | unknown:u32 |
| `0xde` | `fallingStopped` | 12 | `FallingStopped` | unknown:u32, duration_ms:u32, impact_intensity:f32 |
| `0xec` | `wifiFlashID` | 4 | — | — |
| `0xed` | `mfgId` | 12 | `BodyInfo` | serial_number:u32, body_hw_version:u32, body_color:BodyColor(i32)=-1 |
| `0xee` | `firmwareVersion` | variable | `FirmwareSignature` | unknown:u16, signature:string |
| `0xef` | `otaAck` | 7 | `FirmwareUpdateResult` | byte_count:u32, chunk_id:u16, status:u8 |
| `0xf0` | `state` | variable | `RobotState` | timestamp:u32, pose_frame_id:u32, pose_origin_id:u32, pose_x:f32, pose_y:f32, pose_z:f32, pose_angle_rad:f32, pose_pitch_rad:f32, lwheel_speed_mmps:f32, rwheel_speed_mmps:f32, head_angle_rad:f32, lift_height_mm:f32, accel_x:f32, accel_y:f32, accel_z:f32, gyro_x:f32, gyro_y:f32, gyro_z:f32, battery_voltage:f32, status:u32, cliff_data_raw:u16[4], backpack_touch_sensor_raw:u16, curr_path_segment:u8 |
| `0xf1` | `animState` | 15 | `AnimationState` | timestamp:u32, num_anim_bytes_played:i32, num_audio_frames_played:i32, enabled_anim_tracks:u8, tag:u8, client_drop_count:u8 |
| `0xf2` | `image` | variable | `ImageChunk` | frame_timestamp:u32, image_id:u32, chunk_debug:u32, image_encoding:ImageEncoding(i8), image_resolution:ImageResolution(i8), image_chunk_count:u8, chunk_id:u8, status:u16, data:u8[] |
| `0xf3` | `activeObjectAvailable` | 9 | `ObjectAvailable` | factory_id:u32, object_type:ObjectType(i32)=-1, rssi:i8 |
| `0xf4` | `imageGyro` | 17 | `ImageImuData` | image_id:u32, rate_x:f32, rate_y:f32, rate_z:f32, line_2_number:u8 |
| `0xf5` | `objectAccel` | variable | `ObjectAccel` | timestamp:u32, object_id:u32, accel_x:f32, accel_y:f32, accel_z:f32 |


Revisions in practice
---------------------

`pycozmo_ng` implements the 77 messages whose payload is specified above; firmware 2458
defines 161 in total. The tables above put each 2458 message beside the pycozmo class
that carries the same ID.

**65 of the 77 declared messages have a byte-identical payload to their 2458
counterpart,** and none disagree on size. For a further 12 the 2458 message carries a
variable-length field, so only the name differs — the layout is unchanged.

Two renames are worth knowing because the names suggest a different meaning:

- `0x33` — pycozmo's `TurnInPlaceAtSpeed` is `driveCurvature` (curvature drive), while
  `0x3A turnInPlaceAtSpeed` is a separate message the library does not implement.
- `0x11`/`0x03` — the backpack light messages were reshaped (`LightStateCenter`,
  `LightStateSide` → `setBackpackLightsMiddle`, `setBackpackLightsTurnSignals`).

### Messages with no pycozmo class

**84 of the 161 messages are not implemented by this library** (57 engine → robot,
27 robot → engine). Their payload lengths are listed above.

The ones most likely to be worth adding later:

- **Camera intrinsics** — `defaultCameraParams` (`0xC8`, 29 B) and `cameraFOVInfo`
  (`0x5A`, 8 B), so the robot can report its own calibration.
- **IMU** — `imuDataChunk` (`0xBF`, 195 B), `imuRawDataChunk` (`0xC7`, 14 B),
  `imuTemperature` (`0xDC`, 4 B), `imuRequest` (`0x4A`, 4 B).
- **Carry / lift** — `setCarryState` (`0x49`), `checkLiftLoad` (`0x4F`), `liftLoad`
  (`0xDA`), `placeObjectOnGround` (`0x44`), `setObjectBeingCarried` (`0x4E`).
- **Cliff** — `setCliffDetectThreshold` (`0x54`), `cliffEvent` (`0xC0`),
  `potentialCliff` (`0xC1`).
- **Transport** — `setRTTO` (`0x89`), the mode-negotiation message from
  [Transport in 2458](#transport-in-2458).

### Keeping this document honest

The payload lengths that had to be corrected for 2458 are pinned by tests —
`pycozmo_ng/tests/test_protocol_set_prop_slot.py` and
`pycozmo_ng/tests/test_protocol_2458_recovered.py`. A declaration change that alters a
payload length fails them.
Cubes and objects
-----------------

### Connection model

The robot keeps **5 connection slots**, one per connectable object. Connecting is a
two-step handshake on message `0x05` (`setPropSlot`):

1. The engine sends `setPropSlot{factory_id, slot}` — the cube's factory id and the slot
   to place it in (`0`–`4`).
2. The robot answers with `activeObjectConnectionState` (`0xD0`), carrying the assigned
   `object_id`, the `factory_id`, the object type and `connected`.

Releasing a slot is the same message with `factory_id = 0`.

Every connected cube gets its own `object_id`, which is what the per-object messages
address: `setCubeID` (`0x10`), `setCubeLights` (`0x04`), `activeObjectMoved` (`0xB4`),
`activeObjectTapped` (`0xB6`), `objectPowerLevel` (`0xCE`), `activeObjectAvailable`
(`0xF3`) and so on.

### One object of each type

The robot refuses a second object whose type is already connected: one object *of each
type* may be held at once. The three light cubes are three distinct types — `LIGHTCUBE1`,
`LIGHTCUBE2`, `LIGHTCUBE3` (see the `ObjectType` enum) — so **all three cubes can be
connected simultaneously, one per slot.** Multi-cube operation is a supported design, not
a hardware limit.

### Why firmware 2381 could only light one cube

`setPropSlot` and the older `ObjectConnect` occupy the **same ID (`0x05`)** and are the
**same size (5 bytes)**, but the second field changed meaning:

| | field 1 | field 2 | meaning of field 2 |
|---|---|---|---|
| 2381 (`ObjectConnect`) | `factory_id` (u32) | `connect` (**bool**) | connect / disconnect flag |
| 2458 (`setPropSlot`) | `factory_id` (u32) | `slot` (**u8**) | **which of the 5 slots** |

Sending `ObjectConnect(factory_id, True)` for every cube is read as "put this cube in
**slot 1**". Every cube lands in the same slot, each new connect evicts the previous one,
and the surviving cube's `object_id` is always `1` — because slot 1 is always the slot
being written. That is the whole of the "only one cube lights up" behaviour; it is not a
BLE or per-cube limit.

### Verified on hardware (2026-10-15)

Both grammars in one session, three cubes in view:

| Phase | Sent per cube | Result |
|---|---|---|
| A | `ObjectConnect{factory_id, connect=True}` | **1 of 3** connected, `object_id=1` |
| B | `setPropSlot{factory_id, slot=0,1,2}` | **3 of 3** connected **simultaneously** |

Phase B, verbatim:

```
sent setPropSlot{factory_id=0x3b3958b8, slot=0}
sent setPropSlot{factory_id=0xb3660b44, slot=1}
sent setPropSlot{factory_id=0x5e62bca3, slot=2}
  [evt] factory_id=0x3b3958b8 object_id=0 connected=True
  [evt] factory_id=0xb3660b44 object_id=1 connected=True
  [evt] factory_id=0x5e62bca3 object_id=2 connected=True
```

Slot maps 1:1 to `object_id` (slot 0 → 0, slot 1 → 1, slot 2 → 2) and all three are still
connected five seconds later. No per-slot restriction is visible; the robot keeps five
slots and this uses three.

Through the public API, end to end, on the same robot:

```
discovered cubes: {1: '0x3b3958b8', 2: '0xb3660b44', 3: '0x5e62bca3'}
connected after connect(): [1, 2, 3]
still connected after 6 s:  [1, 2, 3]
```

### What this library does

`CubeManager` assigns a distinct slot per cube (`slot = cube_id - 1`, so 0/1/2) and sends
one `setPropSlot` per cube; `disconnect_all_cubes` releases every slot with
`factory_id = 0`. The keepalive thread reconnects a dropped cube into its own slot and
refreshes `setCubeID` + `setCubeLights` so LEDs persist.

The slot meaning is firmware-specific: on a robot running 2381 the second byte is still a
connect flag. This library targets current firmware.


References
----------

- `CONTEXT.md` — frames and coordinate transforms used by the vision/navigation stack
- `pycozmo_ng/protocol/protocol_declaration.py` — the machine-readable form of this document
- `pycozmo_ng/protocol/protocol_encoder.py` — encoder/decoder generated from it
- `pycozmo_ng/tests/` — payload lengths and wire layout pinned by tests
