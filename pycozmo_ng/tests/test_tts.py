"""Tests for the offline TTS generator (pyttsx3).

pyttsx3's espeak driver synthesizes asynchronously: ``runAndWait()`` can
return before the WAV file exists - always so on the first utterance of a
fresh engine, where the queued endLoop stops the driver loop before the
synthesis it started has finished. These tests pin ``generate()`` to the
contract the audio pipeline relies on: it never returns until the file is
on disk and fully written, and it raises a clear error (leaving no partial
file behind) when synthesis stalls.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pycozmo_ng.sound import TTS, TTSGenerationError
from pycozmo_ng.sound import tts as tts_module


class AsyncFakeEngine:
    """Stand-in for pyttsx3's espeak driver.

    Mirrors the real driver's timing: ``runAndWait()`` returns immediately
    while the file shows up (empty first, then fully written) afterwards.
    """

    def __init__(self, delay_s: float = 0.05, completes: bool = True) -> None:
        self.delay_s = delay_s
        self.completes = completes
        self.syntheses: list[str] = []
        self._pending: tuple[str, str] | None = None

    def save_to_file(self, text: str, filename: str) -> None:
        self.syntheses.append(text)
        self._pending = (text, filename)

    def runAndWait(self) -> None:
        if self._pending is None:
            return
        _, filename = self._pending
        Path(filename).touch()  # the real driver creates the file before filling it
        if self.completes:
            threading.Thread(target=self._finish, args=(filename,), daemon=True).start()

    def _finish(self, filename: str) -> None:
        time.sleep(self.delay_s)
        Path(filename).write_bytes(b"RIFF....WAVE" * 8)


def _tts(monkeypatch, tmp_path, engine) -> TTS:
    """TTS wired to a fake engine and a per-test cache directory."""
    monkeypatch.setattr(tts_module, "_TTS_CACHE_DIR", tmp_path)
    tts = TTS()
    monkeypatch.setattr(tts, "_engine", engine)
    return tts


class TestGenerate:
    def test_waits_for_the_file_to_be_written(self, monkeypatch, tmp_path):
        """runAndWait returns early - generate must still wait for the bytes."""
        engine = AsyncFakeEngine(delay_s=0.1)
        tts = _tts(monkeypatch, tmp_path, engine)
        start = time.monotonic()
        path = tts.generate("hello there")
        elapsed = time.monotonic() - start
        assert Path(path).read_bytes()  # fully written, not the empty placeholder
        assert elapsed >= engine.delay_s

    def test_reuses_the_cached_file(self, monkeypatch, tmp_path):
        engine = AsyncFakeEngine(delay_s=0.01)
        tts = _tts(monkeypatch, tmp_path, engine)
        path = tts.generate("same words")
        assert engine.syntheses == ["same words"]
        assert tts.generate("same words") == path
        assert engine.syntheses == ["same words"]  # no second synthesis

    def test_reports_a_timeout_and_removes_the_partial_file(self, monkeypatch, tmp_path):
        """A stalled synthesis is a refusal, never a missing/partial file."""
        monkeypatch.setattr(tts_module, "_WAV_TIMEOUT_S", 0.05)
        monkeypatch.setattr(tts_module, "_WAV_POLL_INTERVAL_S", 0.005)
        engine = AsyncFakeEngine(completes=False)
        tts = _tts(monkeypatch, tmp_path, engine)
        with pytest.raises(TTSGenerationError, match="timed out"):
            tts.generate("never finishes")
        assert list(tmp_path.glob("*.wav")) == []  # leftover dropped, a retry re-synthesizes


class TestSay:
    def test_plays_the_generated_file(self, monkeypatch, tmp_path):
        engine = AsyncFakeEngine(delay_s=0.01)
        tts = _tts(monkeypatch, tmp_path, engine)
        client = MagicMock()
        tts.say("hi there", client)
        (path,) = client.play_audio.call_args.args
        assert Path(path).read_bytes()
