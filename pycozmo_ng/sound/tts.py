"""
Text-to-Speech module using pyttsx3 (offline, lightweight).

Generates WAV audio files from text and feeds them into the existing
audio playback pipeline via Client.play_audio().
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from pycozmo_ng.logger import logger
from pycozmo_ng.sound.exception import TTSGenerationError, TTSNotInstalledError

if TYPE_CHECKING:
    import pyttsx3

__all__ = ["TTS", "TTSGenerationError", "TTSNotInstalledError"]

_TTS_CACHE_DIR = None


def _get_cache_dir() -> Path:
    """Get the directory for cached TTS audio files."""
    global _TTS_CACHE_DIR
    if _TTS_CACHE_DIR is None:
        _TTS_CACHE_DIR = Path(tempfile.gettempdir()) / "pycozmo_tts"
        _TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _TTS_CACHE_DIR


class TTS:
    """Offline text-to-speech engine using pyttsx3.

    Converts text to WAV audio files that can be played through the
    Cozmo robot's existing audio pipeline. Generated audio is cached
    to avoid re-synthesizing the same text.

    Usage:
        tts = TTS(rate=150)
        tts.say("Hello", client)  # plays through client.play_audio()
    """

    def __init__(self, rate: int = 150) -> None:
        self._cache: dict[str, str] = {}
        self._lock = threading.Lock()
        self._engine: pyttsx3.Engine | None = None
        self._rate = rate

    @property
    def engine(self) -> pyttsx3.Engine:
        """Lazy-init the pyttsx3 engine."""
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError:
                raise TTSNotInstalledError("pyttsx3 is not installed. Install it with: pip install pyttsx3")
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
        return self._engine

    def _text_to_path(self, text: str) -> str:
        """Convert text to a deterministic cache file path."""
        h = hashlib.md5(text.encode()).hexdigest()
        return str(_get_cache_dir() / f"{h}.wav")

    def generate(self, text: str) -> str:
        """Generate a WAV file from text and return its path.

        Caches the result so repeated calls for the same text
        reuse the existing file.

        Args:
            text: The text to synthesize.

        Returns:
            Path to the generated WAV file.
        """
        path = self._text_to_path(text)
        with self._lock:
            if path not in self._cache and not os.path.exists(path):
                try:
                    self.engine.save_to_file(text, path)
                    self.engine.runAndWait()
                except Exception as exc:  # noqa: BLE001
                    raise TTSGenerationError(f"Failed to synthesize '{text[:40]}...': {exc}")
                self._cache[text] = path
                logger.debug(f"TTS: generated '{path}' for '{text[:40]}'")
        return path

    def say(self, text: str, client) -> None:
        """Say text through the Cozmo robot.

        Generates a WAV file (or uses cached version) and plays it
        through the client's audio pipeline.

        Args:
            text: The text to speak.
            client: The Client instance with a working anim_controller.
        """
        path = self.generate(text)
        client.play_audio(path)
