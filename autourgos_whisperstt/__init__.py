"""
autourgos-whisperstt
======================
Local, offline speech-to-text for the Autourgos framework using
faster-whisper (a CTranslate2 reimplementation of OpenAI's Whisper). No API
key, no per-call network cost -- runs entirely on-device after a one-time
model download. Cross-platform.

Zero dependency to import; `faster-whisper` is required only to actually
call `WhisperSTT.transcribe()` (`pip install autourgos-whisperstt[whisper]`),
and the model itself loads lazily on first use, not at construction.

For zero-download, zero-setup (but noticeably less accurate) transcription
on Windows specifically, see the sibling package `autourgos-windowstt`
(Windows' own built-in SAPI recognizer).

Quick start::

    from autourgos_micinput import MicrophoneStream
    from autourgos_whisperstt import WhisperSTT

    stt = WhisperSTT(model_size="base")

    async def listen():
        async with MicrophoneStream(sample_rate=16000) as mic:
            chunks = [chunk async for chunk in mic][:20]  # ~2s
        return stt.transcribe(b"".join(chunks), sample_rate=16000)
"""

from .stt import WhisperSTT, WhisperSTTError, WhisperSTTUnavailableError

from autourgos_core import package_version

__version__ = package_version("autourgos-whisperstt", fallback="0.1.2")

__all__ = [
    "WhisperSTT",
    "WhisperSTTError",
    "WhisperSTTUnavailableError",
]
