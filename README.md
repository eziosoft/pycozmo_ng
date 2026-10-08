
PyCozmo-NG
=======

`PyCozmo-NG` is a pure-Python communication library, alternative SDK, and application for the
[Cozmo robot](https://www.digitaldreamlabs.com/pages/cozmo) . It allows controlling a Cozmo robot directly, without
having to go through a mobile device, running the Cozmo app.

The library is loosely based on the [Anki Cozmo Python SDK](https://github.com/anki/cozmo-python-sdk) and the
[cozmoclad](https://pypi.org/project/cozmoclad/) ("C-Like Abstract Data") library.

This project is a tool for exploring the hardware and software of the Digital Dream Labs (originally Anki) Cozmo robot.
It is unstable and heavily under development.


Usage
-----

Basic:

```python
import time
import pycozmo_ng

with pycozmo_ng.connect() as cli:
    cli.set_head_angle(angle=0.6)
    time.sleep(1)
```

Advanced:

```python
import pycozmo_ng

cli = pycozmo_ng.Client()
cli.start()
cli.connect()
cli.wait_for_robot()

cli.drive_wheels(lwheel_speed=50.0, rwheel_speed=50.0, duration=2.0)

cli.disconnect()
cli.stop()
```


Documentation
-------------

[https://pycozmo.readthedocs.io/](https://pycozmo.readthedocs.io/)


Robot Support
-------------

Sensors:
- [x] Camera
- [x] Cliff sensor
- [x] Accelerometers
- [x] Gyro
- [x] Battery voltage
- [x] Cube battery voltage
- [x] Cube accelerometers
- [x] Backpack button (v1.5 hardware and newer)

Actuators:
- [x] Wheel motors
- [x] Head motor
- [x] Lift motor
- [x] OLED display
- [x] Speaker
- [x] Backpack LEDs
- [x] IR LED
- [x] Cube LEDs
- [x] Platform LEDs (when available)

On-board functions:
- [x] Wi-Fi AP
- [x] Bluetooth LE
- [x] Localization
- [x] Path tracking
- [x] NV RAM storage
- [x] Over-the-air (OTA) firmware updates

Off-board functions:
- [x] Procedural face generation
- [x] Cozmo animations from FlatBuffers .bin files
- [ ] Personality engine - work in progress
- [ ] Cozmo behaviors - work in progress
- [ ] Motion detection
- [ ] Object (cube and platform) detection
- [ ] Cube marker recognition
- [ ] Face detection
- [ ] Face recognition
- [ ] Facial expression estimation
- [ ] Pet detection
- [ ] Camera calibration
- [ ] Navigation map building
- [ ] Text-to-speech
- [ ] Songs

Extra off-board functions:
- [ ] Vector animations from FlatBuffers .bin files
- [ ] Vector behaviors
- [x] ArUco marker recognition
- [ ] Cozmo and Vector robot detection
- [ ] Drivable area estimation
- [ ] Voice commands

If you have ideas for other functionality [share them via GitHub](https://github.com/zayfod/pycozmo/issues).


PyCozmo-NG In The Wild
-------------------

- [Expressive Eyes](https://git.brl.ac.uk/ca2-chambers/expressive-eyes) - rendering various facial expressions using
    PyCozmo-NG's procedural_face module
- an [application](https://github.com/paulbaumgarten/paulbaumgarten/blob/6d9025ec715e0b5f133ce9b7e39e74a5389c5334/myp/cozmo/assets/02-camera.py)
    that recognizes [ArUco markers](https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html), using OpenCV
- a [ROS2 driver](https://github.com/solosito/cozmo_ros2_ws/)
- another [ROS2 driver](https://github.com/brean/cozmo_ros)


Connecting to Cozmo over Wi-Fi
------------------------------

A Wi-Fi connection needs to be established with Cozmo before using PyCozmo-NG applications.

1. Wake up Cozmo by placing it on the charging platform
2. Make Cozmo display it's Wi-Fi PSK by rising and lowering its lift
3. Scan for Cozmo's Wi-Fi SSID (depends on the OS)
4. Connect using Cozmo's Wi-Fi PSK (depends on the OS)

[This video](https://www.youtube.com/watch?v=-k_oiQhBa5o) summarizes the connection process.


PyCozmo-NG vs. the Cozmo SDK
-------------------------

A Cozmo SDK application (aka "game") acts as a client to the Cozmo app (aka "engine") that runs on a mobile device.
The low-level communication happens over USB and is handled by the `cozmoclad` library.

In contrast, an application using PyCozmo-NG basically replaces the Cozmo app and acts as the "engine". PyCozmo-NG handles
the low-level UDP communication with Cozmo.
   
```
+------------------+                   +------------------+                   +------------------+
|     SDK app      |     Cozmo SDK     |    Cozmo app     |       Cozmo       |      Cozmo       |
|      "game"      |     cozmoclad     |     "engine"     |      protocol     |     "robot"      |
|                  | ----------------> |   Wi-Fi client   | ----------------> |     Wi-Fi AP     |
|                  |        USB        |    UDP client    |     UDP/Wi-Fi     |    UDP Server    |
+------------------+                   +------------------+                   +------------------+
```


Requirements
------------

- [Python](https://www.python.org/downloads/) 3.6.0 or newer
- [Pillow](https://github.com/python-pillow/Pillow) 6.0.0 - Python image library
- [FlatBuffers](https://github.com/google/flatbuffers) - serialization library
- [dpkt](https://github.com/kbandla/dpkt) - TCP/IP packet parsing library 


Installation
------------

Using pip:

```
pip install --user pycozmo

pycozmo_resources.py download
```

From source:

```
git clone https://github.com/zayfod/pycozmo.git
cd pycozmo
python setup.py install --user

pycozmo_resources.py download
```

From source, for development:

```
git clone git@github.com:zayfod/pycozmo.git
cd pycozmo
python setup.py develop --user
pip install --user -r requirements-dev.txt

pycozmo_resources.py download
```

 
Support
-------

Bug reports and changes should be sent via GitHub:

[https://github.com/zayfod/pycozmo](https://github.com/zayfod/pycozmo)

DDL Robot Discord server, channel #development-cozmo:

[https://discord.gg/ew92haS](https://discord.gg/ew92haS)


Disclaimer
----------

This project is not affiliated with [Digital Dream Labs](https://www.digitaldreamlabs.com/) or
[Anki](https://anki.com/).
