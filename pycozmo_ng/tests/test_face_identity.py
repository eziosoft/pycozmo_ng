"""Tests for face identity — presence epochs and the events they drive.

One face ID identifies one uninterrupted run of measured frames with a
face in view. The critical distinction: a skipped frame is "not
measured", never "face gone".
"""

from types import SimpleNamespace
from unittest.mock import Mock

from pycozmo_ng import event
from pycozmo_ng.vision.events import DetectionEventDispatcher
from pycozmo_ng.vision.face_identity import FaceFrame, FaceIdentity


def make_det():
    return SimpleNamespace(x=1, y=1, width=4, height=4, confidence=0.9, face_id=None)


def make_cli():
    return SimpleNamespace(dispatch=Mock())


class TestFaceIdentity:
    def test_starts_at_none(self):
        ident = FaceIdentity()
        assert ident.current_id is None

    def test_first_face_starts_epoch_one(self):
        ident = FaceIdentity()
        det = make_det()
        frame = ident.update([det])
        assert frame == FaceFrame(face_id=1, started=1, ended=None)
        assert det.face_id == 1
        assert ident.current_id == 1

    def test_sustained_presence_keeps_id(self):
        ident = FaceIdentity()
        ident.update([make_det()])
        frame = ident.update([make_det()])
        assert frame == FaceFrame(face_id=1, started=None, ended=None)

    def test_absence_ends_epoch(self):
        ident = FaceIdentity()
        ident.update([make_det()])
        frame = ident.update([])
        assert frame == FaceFrame(face_id=None, started=None, ended=1)
        assert ident.current_id is None

    def test_reappearance_gets_new_id(self):
        ident = FaceIdentity()
        ident.update([make_det()])
        ident.update([])
        det = make_det()
        frame = ident.update([det])
        assert frame.started == 2
        assert det.face_id == 2

    def test_frame_skip_holds_state(self):
        """A skipped frame must not end presence (no spurious EvtFaceLost)."""
        ident = FaceIdentity()
        ident.update([make_det()])
        frame = ident.update([], measured=False)
        assert frame == FaceFrame(face_id=1, started=None, ended=None)
        assert ident.current_id == 1
        # …and the next measured frame continues the same epoch
        frame = ident.update([make_det()])
        assert frame == FaceFrame(face_id=1, started=None, ended=None)

    def test_frame_skip_without_face_stays_empty(self):
        ident = FaceIdentity()
        frame = ident.update([], measured=False)
        assert frame == FaceFrame(face_id=None)
        assert ident.current_id is None

    def test_reset(self):
        ident = FaceIdentity()
        ident.update([make_det()])
        ident.reset()
        assert ident.current_id is None


class TestFaceEvents:
    """DetectionEventDispatcher turns FaceFrames into events."""

    def test_detected_then_lost(self):
        ident = FaceIdentity()
        dispatcher = DetectionEventDispatcher()
        cli = make_cli()

        det = make_det()
        frame = ident.update([det])
        dispatcher.dispatch_face_events(cli, [det], frame)
        assert cli.dispatch.call_args.args[0] is event.EvtFaceDetected
        assert cli.dispatch.call_args.args[2] == [det]

        cli2 = make_cli()
        lost = ident.update([])
        dispatcher.dispatch_face_events(cli2, [], lost)
        assert cli2.dispatch.call_args.args[0] is event.EvtFaceLost
        assert cli2.dispatch.call_args.args[2] == 1

    def test_observed_throttles(self):
        ident = FaceIdentity()
        dispatcher = DetectionEventDispatcher(obs_throttle_frames=3)
        cli = make_cli()

        det = make_det()
        for _ in range(3):
            frame = ident.update([det])
            dispatcher.dispatch_face_events(cli, [det], frame)

        observed = [
            c for c in cli.dispatch.call_args_list if c.args[0] is event.EvtFaceObserved
        ]
        assert len(observed) == 1

    def test_reset_clears_throttle(self):
        dispatcher = DetectionEventDispatcher()
        dispatcher._obs_frame_counter = 4
        dispatcher.reset()
        assert dispatcher._obs_frame_counter == 0


class TestDispatcherPublicIds:
    """The dispatcher converts internal indices to public IDs at the boundary."""

    def test_lost_carries_cube_id(self):
        dispatcher = DetectionEventDispatcher()
        cli = make_cli()

        det = SimpleNamespace(cube_index=0)
        dispatcher.dispatch_cube_events(cli, [det])
        assert cli.dispatch.call_args.args[0] is event.EvtCubeDetected

        cli2 = make_cli()
        dispatcher.dispatch_cube_events(cli2, [])
        assert cli2.dispatch.call_args.args[0] is event.EvtCubeLost
        assert cli2.dispatch.call_args.args[2] == 1  # index 0 -> public ID 1
