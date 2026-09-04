# Changelog

## 0.1.2

- Internal: `__version__` resolution moved to `autourgos_core.package_version()` (bumped `autourgos-core>=0.3.0`). No functional change.

## 0.1.1

- Internal: `_load_faster_whisper()`'s import-probing logic moved to `autourgos_core.try_import()` (new `autourgos-core>=0.2.0` dependency), and `_require_available()`'s conditional-raise moved to `autourgos_core.require_available()`. No behavior change -- error messages stay identical.

## 0.1.0

- Initial release: `WhisperSTT` -- local, offline speech-to-text via `faster-whisper`. No API key, no per-call network cost. Takes raw 16-bit PCM (matches `autourgos-micinput`'s `MicrophoneStream` output) via `transcribe()`/`atranscribe()`; model loads lazily on first call, not at construction. Audio is handed to faster-whisper as a WAV file path (not a raw array) so its own resampling to 16kHz happens correctly regardless of input `sample_rate`.
- Live-verified 2026-09-04: the exact same test WAV used to verify `autourgos-windowstt` ("Testing one two three, this is a speech recognition check.") transcribed as "Testing 1, 2, 3, this is a speech recognition check." -- correct engine wiring, and noticeably higher accuracy than SAPI on identical audio. Also verified end-to-end: real `autourgos-micinput` mic capture piped into `atranscribe()` in the same event loop, no threading issues (unlike `autourgos-windowstt`'s COM-based engine, faster-whisper needed no special per-thread setup).
- 7 tests (mocked `faster_whisper.WhisperModel`, no real model download or inference required to run the suite).
