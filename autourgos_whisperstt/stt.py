"""
WhisperSTT -- local, offline speech-to-text using faster-whisper (a
CTranslate2 reimplementation of OpenAI's Whisper). No API key, no per-call
network cost -- runs entirely on-device (CPU or GPU) after a one-time model
download on first use. Cross-platform (unlike the sibling
autourgos-windowstt, which is Windows-only via SAPI).

Gated behind the `whisper` extra (`pip install autourgos-whisperstt[whisper]`,
which pulls in `faster-whisper`) so importing this module never requires it
-- only constructing a `WhisperSTT` and calling `.transcribe()` does. The
underlying `WhisperModel` itself is lazily created on first `.transcribe()`
call (not in `__init__`), so constructing a `WhisperSTT()` is always cheap
and never triggers a model download by itself.

Live-verified (2026-09-04): a real speech WAV ("Testing one two three, this
is a speech recognition check.", generated via Windows SAPI text-to-speech)
transcribed as "Testing 1, 2, 3, this is a speech recognition check." --
correct engine wiring, real recognition, and noticeably higher accuracy
than autourgos-windowstt's SAPI-based recognition of the identical audio
("Testing 1 to 3 this is a speech recognition check").
"""

from __future__ import annotations

import os
import tempfile
import wave
from typing import Any, Optional, Tuple


class WhisperSTTError(Exception):
    """Base error for autourgos-whisperstt."""


class WhisperSTTUnavailableError(WhisperSTTError):
    """Raised when `faster-whisper` isn't installed."""


def _load_faster_whisper() -> Tuple[bool, Any, Optional[str]]:
    """Try to import faster_whisper. Returns (available, WhisperModel class, error)."""
    try:
        from faster_whisper import WhisperModel
        return True, WhisperModel, None
    except ImportError as exc:
        return False, None, str(exc)


class WhisperSTT:
    """
    Speech-to-text via a local Whisper model (faster-whisper).

    Usage::

        from autourgos_micinput import MicrophoneStream
        from autourgos_whisperstt import WhisperSTT

        stt = WhisperSTT(model_size="base")  # tiny/base/small/medium/large-v3

        async with MicrophoneStream(sample_rate=16000) as mic:
            chunks = [chunk async for chunk in mic][:20]  # ~2s at chunk_ms=100
        text = stt.transcribe(b"".join(chunks), sample_rate=16000)

    The model is downloaded (once, cached by `faster-whisper`/huggingface_hub
    under the user's cache directory) and loaded lazily on the first
    `.transcribe()`/`.atranscribe()` call, not in the constructor -- so
    creating a `WhisperSTT()` instance is always cheap.
    """

    def __init__(
        self,
        *,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ) -> None:
        """
        model_size: any faster-whisper model name/size -- "tiny", "base",
            "small", "medium", "large-v3", etc. Bigger = more accurate, slower,
            larger download. "base" is a reasonable default for short utterances.
        device: "cpu" or "cuda" (GPU, if available and ctranslate2 was built
            with CUDA support).
        compute_type: quantization -- "int8" (fastest/smallest, default,
            good for CPU), "float16" (typical for GPU), "float32".
        language: ISO 639-1 code (e.g. "en") to skip language auto-detection,
            or None to auto-detect per call.
        """
        self._available, self._WhisperModel, self._import_error = _load_faster_whisper()
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Any = None

    def _require_available(self) -> None:
        if not self._available:
            raise WhisperSTTUnavailableError(
                "The 'faster-whisper' package is required "
                f"(pip install autourgos-whisperstt[whisper]). Import error: {self._import_error}"
            )

    def _get_model(self) -> Any:
        if self._model is None:
            self._require_available()
            self._model = self._WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe raw 16-bit PCM audio (as produced by
        `autourgos_micinput.MicrophoneStream`) to text. Blocking -- runs
        Whisper inference on the calling thread. Use `atranscribe()` from
        async code instead of calling this directly inside an event loop.

        The audio is written to a temporary WAV file and handed to
        faster-whisper as a file path (rather than a raw numpy array) so its
        own audio decoding (via `av`/ffmpeg) resamples correctly to the
        16kHz Whisper expects regardless of `sample_rate` -- passing an
        array directly would skip that resampling and degrade accuracy for
        any rate other than 16000.
        """
        self._require_available()
        model = self._get_model()

        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(channels)
                w.setsampwidth(2)  # 16-bit PCM
                w.setframerate(sample_rate)
                w.writeframes(pcm_bytes)

            segments, _info = model.transcribe(wav_path, language=language or self.language)
            return " ".join(seg.text.strip() for seg in segments).strip()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    async def atranscribe(
        self,
        pcm_bytes: bytes,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        language: Optional[str] = None,
    ) -> str:
        """Async-safe equivalent of `transcribe()` -- runs the blocking
        Whisper inference in a worker thread instead of stalling the event
        loop."""
        import asyncio

        return await asyncio.to_thread(
            self.transcribe, pcm_bytes, sample_rate=sample_rate, channels=channels, language=language
        )
