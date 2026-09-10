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

    lights = [
        pycozmo.lights.red_light,
        pycozmo.lights.green_light,
        pycozmo.lights.blue_light,
        pycozmo.lights.off_light,
    ]
    for light in lights:
        for cube_id in cube_ids:
            pkt = pycozmo.protocol_encoder.CubeId(object_id=cube_id)
            cli.conn.send(pkt)
            pkt = pycozmo.protocol_encoder.CubeLights(
                states=(light, light, light, light)
            )
            cli.conn.send(pkt)
        time.sleep(2)
