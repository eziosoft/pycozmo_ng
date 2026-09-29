"""Face identity — owns which face is which across frames.

The detection pipeline reports at most one face per frame and cannot
re-identify a person who leaves and comes back. Face identity is therefore
**presence**: a face ID identifies one uninterrupted run of processed
frames with a face in view. Each new run gets the next ID (1, 2, 3, …),
so IDs are unique per session and ``WorldMap`` keys faces by them.

The module also owns the distinction between *"no face"* and *"not
measured"*: when face detection is skipped for a frame
(``face_detect_every_n_frames``), :meth:`FaceIdentity.update` is called
with ``measured=False`` and holds state — a frame skip must not fire
``EvtFaceLost``.

Feeding one frame through :meth:`update` assigns ``detection.face_id``
(before annotation reads it) and returns a :class:`FaceFrame` describing
what changed: nothing, a face that ``started``, or a face that ``ended``.
``FrameStage`` passes that result to the event dispatcher and uses
``ended`` to call ``WorldMap.forget_face``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceFrame:
    """Identity outcome for one processed frame.

    Attributes:
        face_id: The face present this frame (``None`` if no face, or if
            this frame was not measured).
        started: Set when this frame began a new presence — the new ID.
        ended: Set when a presence ended on this frame — the lost ID.
    """

    face_id: int | None = None
    started: int | None = None
    ended: int | None = None


class FaceIdentity:
    """Assigns face IDs and detects presence transitions across frames.

    Stateful: one instance per vision session (``Brain.setup_vision``
    constructs it, ``FrameStage.configure`` injects it).
    """

    def __init__(self) -> None:
        self._next_id = 1
        self._current: int | None = None

    @property
    def current_id(self) -> int | None:
        """ID of the face currently in view, or ``None``."""
        return self._current

    def update(self, detections: list, *, measured: bool = True) -> FaceFrame:
        """Advance identity with this frame's detections.

        Args:
            detections: Faces detected this frame. Non-empty only when
                ``measured`` is True (face detection ran).
            measured: Whether face detection actually ran this frame.
                ``False`` (frame skip) holds state and returns no
                transitions — "not measured" is not "gone".

        Returns:
            The :class:`FaceFrame` describing what changed. Assigns
            ``face_id`` on every detection when a face is present.
        """
        if not measured:
            # Frame skip: presence (or absence) is unchanged.
            return FaceFrame(face_id=self._current)

        if detections:
            started = None
            if self._current is None:
                self._current = self._next_id
                self._next_id += 1
                started = self._current
            for det in detections:
                det.face_id = self._current
            return FaceFrame(face_id=self._current, started=started)

        ended = self._current
        self._current = None
        return FaceFrame(ended=ended)

    def reset(self) -> None:
        """Forget all identity state (ID numbering restarts)."""
        self._current = None


__all__ = ["FaceFrame", "FaceIdentity"]
