"""Tests for the motion seam: MotionController + MotionToken.

The controller is the single place that knows which motions are running,
which is what makes ``Navigator.stop()`` one call. The token is what each
job polls. These tests cover the contract the navigation code relies on:

* a token is cancelled by ``cancel(name)``, by ``cancel_all()``, or by an
  enclosing job being cancelled (jobs nest);
* a token begun *after* ``cancel_all()`` is not cancelled (re-arming);
* ``active_names`` lists exactly the jobs whose scope is open.
"""

import threading
import time

from pycozmo.navigation.motion_controller import MotionController


def _wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class TestTokenLifecycle:
    def test_fresh_token_is_not_cancelled(self):
        motion = MotionController()
        with motion.begin("drive") as job:
            assert not job.cancelled

    def test_active_names_tracks_open_scopes(self):
        motion = MotionController()
        with motion.begin("drive") as job:
            del job
            assert motion.active_names == ("drive",)
        assert motion.active_names == ()

    def test_active_names_lists_each_job_once(self):
        motion = MotionController()
        # Keep these nested (SIM117 suppressed below): the inner token takes
        # its parent from the enclosing scope when begin() is called.
        with motion.begin("drive"):  # noqa: SIM117
            with motion.begin("align"):
                assert sorted(motion.active_names) == ["align", "drive"]

    def test_scope_can_nest_on_one_thread(self):
        motion = MotionController()
        with motion.begin("drive"):
            with motion.begin("align") as inner:
                assert not inner.cancelled
            # Inner scope popped; outer one is still current.
            assert motion.active_names == ("drive",)


class TestCancellation:
    def test_cancel_by_name_hits_only_that_job(self):
        motion = MotionController()
        with motion.begin("navigate") as nav_job:  # noqa: SIM117 - see nesting note above
            with motion.begin("drive") as drive_job:
                assert motion.cancel("navigate") is True
                assert nav_job.cancelled
                assert drive_job.cancelled  # nested inside navigate
        assert motion.cancel("navigate") is False  # nothing live

    def test_cancel_all_cancels_running_tokens(self):
        motion = MotionController()
        with motion.begin("navigate") as old:
            assert motion.cancel_all() == ("navigate",)
            assert old.cancelled

    def test_token_begun_after_cancel_all_is_running(self):
        """Stop() re-arms: only what was already running is cancelled."""
        motion = MotionController()
        with motion.begin("navigate") as old:
            motion.cancel_all()
            assert old.cancelled
        with motion.begin("drive") as fresh:
            assert not fresh.cancelled

    def test_job_begun_inside_a_cancelled_job_still_dies_with_it(self):
        """Nesting outranks the generation: a dying parent takes its children."""
        motion = MotionController()
        with motion.begin("navigate") as old:
            motion.cancel_all()
            with motion.begin("drive") as child:
                assert old.cancelled
                assert child.cancelled

    def test_child_job_is_cancelled_with_its_parent(self):
        motion = MotionController()
        with motion.begin("approach") as parent:
            child = motion.begin("drive")  # begun on the parent's thread
            with child:
                motion.cancel("approach")
                assert parent.cancelled
                assert child.cancelled

    def test_child_job_survives_cancel_of_a_sibling_name(self):
        motion = MotionController()
        with motion.begin("approach"):  # noqa: SIM117 - see nesting note above
            with motion.begin("drive") as child:
                motion.cancel("dock")
                assert not child.cancelled


class TestSpawn:
    def test_spawn_runs_target_with_the_token_on_another_thread(self):
        motion = MotionController()
        seen: dict[str, object] = {}

        def target(job):
            seen["name"] = job.name
            seen["thread"] = threading.get_ident()
            seen["cancelled"] = job.cancelled

        job = motion.spawn("navigate", target)
        assert job.name == "navigate"
        assert _wait_until(lambda: "name" in seen)
        assert seen["name"] == "navigate"
        assert seen["cancelled"] is False
        assert seen["thread"] != threading.get_ident()
        assert _wait_until(lambda: motion.active_names == ())

    def test_spawn_registers_before_the_job_finishes(self):
        motion = MotionController()
        entered = threading.Event()
        release = threading.Event()

        def target(job):
            del job
            entered.set()
            release.wait(2.0)

        motion.spawn("dock", target)
        assert entered.wait(2.0)
        assert "dock" in motion.active_names

        release.set()
        assert _wait_until(lambda: motion.active_names == ())

    def test_spawned_job_sees_cancel_all(self):
        motion = MotionController()
        entered = threading.Event()
        release = threading.Event()
        result: dict[str, bool] = {}

        def target(job):
            entered.set()
            release.wait(2.0)
            result["cancelled"] = job.cancelled

        motion.spawn("navigate", target)
        assert entered.wait(2.0)
        motion.cancel_all()
        release.set()

        assert _wait_until(lambda: "cancelled" in result)
        assert result["cancelled"] is True

    def test_job_spawned_from_a_job_inherits_cancellation(self):
        """A drive started by an approach dies with the approach."""
        motion = MotionController()
        outer_entered = threading.Event()
        inner_entered = threading.Event()
        release = threading.Event()
        result: dict[str, bool] = {}

        def inner(job):
            inner_entered.set()
            release.wait(2.0)
            result["cancelled"] = job.cancelled

        def outer(job):
            del job
            outer_entered.set()
            motion.spawn("drive", inner)  # parented to the outer job
            release.wait(2.0)

        motion.spawn("approach", outer)
        assert outer_entered.wait(2.0)
        assert inner_entered.wait(2.0)

        assert motion.cancel("approach") is True
        release.set()

        assert _wait_until(lambda: "cancelled" in result)
        assert result["cancelled"] is True
        assert _wait_until(lambda: motion.active_names == ())
