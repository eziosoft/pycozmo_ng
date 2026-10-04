"""Tests for the event dispatcher (pycozmo_ng/event.py).

Covers the snapshot-based dispatch (COZMO-53): handlers may be added or
removed from other threads while a packet is being dispatched without
raising "list changed size during iteration".
"""

from pycozmo_ng.event import Dispatcher


class _Evt:
    pass


class TestDispatcher:
    def test_persistent_handler_fires_every_time(self):
        d = Dispatcher()
        calls = []

        def handler():
            calls.append(1)

        d.add_handler(_Evt, handler)
        d.dispatch(_Evt)
        d.dispatch(_Evt)
        assert calls == [1, 1]

    def test_one_shot_handler_removed_after_dispatch(self):
        d = Dispatcher()
        calls = []

        def handler():
            calls.append(1)

        d.add_handler(_Evt, handler, one_shot=True)
        d.dispatch(_Evt)
        d.dispatch(_Evt)
        assert calls == [1]

    def test_add_handler_during_dispatch_is_safe(self):
        """Mutating the handler list mid-dispatch must not raise."""
        d = Dispatcher()
        ran = []

        def add_and_run():
            ran.append("existing")
            d.add_handler(_Evt, added)

        def added():
            ran.append("added-later")

        d.add_handler(_Evt, add_and_run)
        # Must not raise RuntimeError (list changed size during iteration).
        d.dispatch(_Evt)
        d.dispatch(_Evt)
        assert ran == ["existing", "existing", "added-later"]

    def test_del_handler_during_dispatch_is_safe(self):
        d = Dispatcher()
        ran = []

        def remover():
            ran.append("remover")
            d.del_handler(_Evt, removed_handler)

        def removed():
            ran.append("removed")

        d.add_handler(_Evt, remover)
        removed_handler = d.add_handler(_Evt, removed)
        d.dispatch(_Evt)
        # "removed" was in the dispatch snapshot, so it still fires once this
        # round even though remover unregistered it mid-dispatch.
        assert ran == ["remover", "removed"]
        d.dispatch(_Evt)
        # After removal, a fresh dispatch no longer sees it.
        assert ran == ["remover", "removed", "remover"]

    def test_handler_kwargs_passed(self):
        d = Dispatcher()
        seen = []

        def handler(a, *, flag):
            seen.append((a, flag))

        d.add_handler(_Evt, handler)
        d.dispatch(_Evt, "x", flag=True)
        assert seen == [("x", True)]
