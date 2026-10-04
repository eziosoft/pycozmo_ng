"""Live-robot fixtures for the opt-in suite in ``pycozmo_ng/tests_live``.

These tests talk to a real Cozmo and are deliberately **not** part of the
default offline suite (``pytest pycozmo_ng/tests/``). Run them on the Pi
(robot network)::

    pytest pycozmo_ng/tests_live/ -v

The whole suite skips when the robot is unreachable. Tests that physically
move the robot are additionally gated behind ``PYCOZMO_NG_LIVE_MOTION=1`` —
clear the floor space in front of the robot before enabling that.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterator

import pytest

from pycozmo_ng import Cozmo
from pycozmo_ng.exception import PyCozmoNGConnectionError
from pycozmo_ng.logger import setup_basic_logging


def wait_for(fn: Callable[[], object], timeout: float, interval: float = 0.1) -> bool:
    """Poll ``fn`` until it returns truthy or ``timeout`` seconds elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(interval)
    return bool(fn())


@pytest.fixture(scope="session")
def cozmo() -> Iterator[Cozmo]:
    """One robot connection for the whole run; skips if unreachable."""
    setup_basic_logging()
    c = Cozmo()
    try:
        c.connect()
    except (PyCozmoNGConnectionError, OSError) as exc:
        # Unreachable robot: connection timeout, refused or no route to host.
        pytest.skip(f"Cozmo unreachable: {exc}")
    yield c
    c.disconnect()


@pytest.fixture(scope="session")
def vision(cozmo: Cozmo):
    """Vision pipeline started once — calibration + handler registration."""
    cozmo.vision.start()
    yield cozmo.vision
    cozmo.vision.stop()


@pytest.fixture
def motion_ok() -> None:
    """Gate for tests that physically move the robot."""
    if os.environ.get("PYCOZMO_NG_LIVE_MOTION") != "1":
        pytest.skip(
            "set PYCOZMO_NG_LIVE_MOTION=1 (and clear floor space) to allow motion"
        )
