"""Live cube tests — discovery, BLE keepalive cycler, connected property.

Covers CubesAPI/CubeManager paths that need a real robot and real cubes:
factory mapping from ObjectAvailable events and ``connected_cubes`` fed by
ObjectConnectionState events while the cycler runs.
"""

import pytest

from pycozmo_ng.tests_live.conftest import wait_for


@pytest.fixture(scope="session")
def cubes_started(cozmo):
    cozmo.cubes.discover()
    cozmo.cubes.start_lights()
    yield cozmo.cubes
    cozmo.cubes.stop_lights()


def _factory_map(cozmo):
    return cozmo._brain.cubes.ct.cube_factory_id_map


def test_discovery_finds_cubes(cozmo, cubes_started):
    if not _factory_map(cozmo):
        pytest.skip("robot sees no cubes (place cubes in view / check batteries)")
    assert all(1 <= cube_id <= 3 for cube_id in _factory_map(cozmo))


def test_connected_returns_ble_ids(cozmo, cubes_started):
    if not _factory_map(cozmo):
        pytest.skip("no cubes discovered")
    assert wait_for(lambda: len(cubes_started.connected) > 0, timeout=8.0), (
        "no cube connected over BLE within 8 s"
    )
    assert all(
        isinstance(cube_id, int) and 1 <= cube_id <= 3
        for cube_id in cubes_started.connected
    )
