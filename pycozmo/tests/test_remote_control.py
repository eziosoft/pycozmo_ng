"""Tests for RemoteControl's place on the motion seam (driver lease).

The browser poll loop is an adapter: it may only command the wheels while
it holds the **driver lease** on the shared ``MotionController``. These
tests pin the contract navigation relies on:

* a *fresh* drive-key edge takes the lease and starts sending commands;
* a key merely held never takes the lease back — a motion job that
  preempts the keyboard keeps the wheels (the stop-doesn't-stop bug);
* losing the lease goes silent immediately (the holder commands alone);
* releasing every drive key sends one zero and gives the lease up;
* head/lift keys are ungated, and ``stop()`` releases the lease.
"""

import time
from unittest.mock import MagicMock

from pycozmo.navigation.motion_controller import MotionController
from pycozmo.navigation.remote_control import RemoteControl


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _make(
    motion: MotionController | None = None, **kwargs
) -> tuple[RemoteControl, MagicMock]:
    cli = MagicMock()
    kwargs.setdefault("tick_hz", 100.0)
    kwargs.setdefault("idle_stop_sec", 1.0)
    rc = RemoteControl(
        cli, motion=motion if motion is not None else MotionController(), **kwargs
    )
    return rc, cli


class TestLeaseGating:
    def test_silent_until_a_drive_key_appears(self):
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            rc.update_keys({"r"})  # head key only
            assert _wait_until(lambda: cli.move_head.call_count > 0)
            assert cli.drive_wheels.call_count == 0
            assert motion.lease_owner is None
        finally:
            rc.stop()

    def test_drive_edge_takes_lease_and_drives(self):
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: motion.lease_owner == "remote")
            assert _wait_until(lambda: cli.drive_wheels.call_count > 0)
            assert cli.drive_wheels.call_args.kwargs["lwheel_speed"] > 0
        finally:
            rc.stop()

    def test_held_key_does_not_take_the_lease_back(self):
        """A sequence that starts while a key is held keeps the wheels."""
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: motion.lease_owner == "remote")
            assert _wait_until(lambda: cli.drive_wheels.call_count > 0)

            # A motion job takes the lease (navigate/dock/approach start).
            motion.acquire_lease("navigate")
            # The browser keeps re-sending the same held key every 50 ms.
            rc.update_keys({"w"})
            time.sleep(0.05)  # let any in-flight tick land
            count = cli.drive_wheels.call_count
            time.sleep(0.1)  # several ticks
            assert cli.drive_wheels.call_count == count
            assert motion.lease_owner == "navigate"
        finally:
            rc.stop()

    def test_losing_the_lease_goes_silent(self):
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: cli.drive_wheels.call_count > 0)

            with motion.begin("dock"):  # a root job takes the lease
                assert motion.lease_owner == "dock"
                time.sleep(0.05)
                count = cli.drive_wheels.call_count
                time.sleep(0.1)
                assert cli.drive_wheels.call_count == count
        finally:
            rc.stop()

    def test_release_of_every_drive_key_zeroes_once_then_yields(self):
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: motion.lease_owner == "remote")

            rc.update_keys(set())
            assert _wait_until(lambda: motion.lease_owner is None)
            last = cli.drive_wheels.call_args
            assert last.kwargs == {"lwheel_speed": 0, "rwheel_speed": 0}

            count = cli.drive_wheels.call_count
            time.sleep(0.1)
            assert cli.drive_wheels.call_count == count  # silent again
        finally:
            rc.stop()

    def test_stale_input_releases_the_lease(self):
        """Browser gone (no refresh within idle window) → give the wheels up."""
        motion = MotionController()
        rc, cli = _make(motion, idle_stop_sec=0.1)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: motion.lease_owner == "remote")
            assert _wait_until(lambda: motion.lease_owner is None, timeout=2.0)
            # The last thing it sent was the zeroing, then silence.
            assert cli.drive_wheels.call_args.kwargs == {
                "lwheel_speed": 0,
                "rwheel_speed": 0,
            }
        finally:
            rc.stop()

    def test_stop_releases_the_lease(self):
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        rc.update_keys({"w"})
        assert _wait_until(lambda: motion.lease_owner == "remote")
        rc.stop()
        assert motion.lease_owner is None
        assert cli.stop_all_motors.called

    def test_head_keys_stay_available_without_the_lease(self):
        """Head/lift don't contend: they work while a job holds the wheels."""
        motion = MotionController()
        rc, cli = _make(motion)
        rc.start()
        try:
            with motion.begin("navigate"):
                rc.update_keys({"r"})
                assert _wait_until(lambda: cli.move_head.call_count > 0)
                assert cli.drive_wheels.call_count == 0
        finally:
            rc.stop()

    def test_standalone_instance_works_without_injected_motion(self):
        """Library users can still construct RemoteControl(cli) alone."""
        cli = MagicMock()
        rc = RemoteControl(cli, tick_hz=100.0, idle_stop_sec=1.0)
        rc.start()
        try:
            rc.update_keys({"w"})
            assert _wait_until(lambda: cli.drive_wheels.call_count > 0)
            assert rc._motion.lease_owner == RemoteControl.LEASE_OWNER
        finally:
            rc.stop()
