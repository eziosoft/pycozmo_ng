#!/usr/bin/env python

import time

import pycozmo

CUBE_TYPES = {
    pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE1,
    pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE2,
    pycozmo.protocol_encoder.ObjectType.Block_LIGHTCUBE3,
}


with pycozmo.connect() as cli:
    print("Waiting for cubes...")
    while True:
        available_objects = dict(cli.available_objects)
        cubes = [
            (fid, obj)
            for fid, obj in available_objects.items()
            if obj.object_type in CUBE_TYPES
        ]
        if cubes:
            break
        time.sleep(0.1)

    print(f"Found {len(cubes)} cube(s).")

    # Connect to all cubes.
    cube_ids = []
    for factory_id, obj in cubes:
        print(f"Connecting to cube S/N 0x{factory_id:08x} ({obj.object_type.name})...")
        pkt = pycozmo.protocol_encoder.ObjectConnect(
            factory_id=factory_id, connect=True
        )
        cli.conn.send(pkt)
        cli.conn.wait_for(pycozmo.protocol_encoder.ObjectConnectionState)
        object_id = list(cli.connected_objects.keys())[-1]
        cube_ids.append(object_id)
        print(f"  Connected — object_id={object_id}")

    color = pycozmo.lights.Color(int_color=0x00FF00FF)
    light = pycozmo.protocol_encoder.LightState(
        on_color=color.to_int16(),
        off_color=pycozmo.lights.off.to_int16(),
        on_frames=5,
        off_frames=20,
        transition_on_frames=5,
        transition_off_frames=10,
    )

    for cube_id in cube_ids:
        pkt = pycozmo.protocol_encoder.CubeId(
            object_id=cube_id, rotation_period_frames=40
        )
        cli.conn.send(pkt)
        pkt = pycozmo.protocol_encoder.CubeLights(
            states=(
                light,
                pycozmo.lights.off_light,
                pycozmo.lights.off_light,
                pycozmo.lights.off_light,
            )
        )
        cli.conn.send(pkt)

    time.sleep(30.0)
