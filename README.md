# autourgos-whisperstt

[![Framework: Autourgos](https://img.shields.io/badge/Framework-Autourgos-orange.svg)](https://github.com/devxjitin)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://pypi.org/project/autourgos-whisperstt/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/devxjitin/autourgos-whisperstt/blob/main/LICENSE)
[![Author](https://img.shields.io/badge/Author-Jitin%20Kumar%20Sengar-blue.svg)](https://github.com/devxjitin)

Local, offline speech-to-text for the Autourgos framework using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (a CTranslate2 reimplementation of OpenAI's Whisper). No API key, no per-call network cost — runs entirely on-device (CPU or GPU) after a one-time model download. Cross-platform.

For zero-download, zero-setup transcription on Windows specifically (at the cost of noticeably lower accuracy), see the sibling package [autourgos-windowstt](https://github.com/devxjitin/autourgos-windowstt).

```python
from autourgos_micinput import MicrophoneStream
from autourgos_whisperstt import WhisperSTT

stt = WhisperSTT(model_size="base")  # tiny/base/small/medium/large-v3

async def listen(seconds: float = 2.0):
    frames_needed = int(seconds * 1000 / 100)  # default chunk_ms=100
    chunks = []
    async with MicrophoneStream(sample_rate=16000) as mic:
        async for chunk in mic:
            chunks.append(chunk)
            if len(chunks) >= frames_needed:
                break
    return stt.transcribe(b"".join(chunks), sample_rate=16000)

# text = asyncio.run(listen())
```

---

## Install

```bash
pip install "autourgos-whisperstt[whisper]"
```

`faster-whisper` is required to actually call `.transcribe()`/`.atranscribe()` and is gated behind the `whisper` extra — `import autourgos_whisperstt` alone never requires it. The model itself downloads (once, cached by `faster-whisper`/`huggingface_hub`) and loads lazily on first use, not at `WhisperSTT()` construction. Requires Python 3.10+.

---

## Usage

`WhisperSTT` takes raw 16-bit PCM bytes (exactly what [autourgos-micinput](https://github.com/devxjitin/autourgos-micinput)'s `MicrophoneStream` yields) and returns transcribed text:

```python
stt = WhisperSTT(model_size="base")
text = stt.transcribe(pcm_bytes, sample_rate=16000)          # sync, blocking
text = await stt.atranscribe(pcm_bytes, sample_rate=16000)   # async, offloads to a worker thread
```

`transcribe()` is blocking — it runs Whisper inference on the calling thread. Use `atranscribe()` from inside an event loop (e.g. right after capturing from `MicrophoneStream`) instead of calling `transcribe()` directly, same reasoning as `SpeakerPlayer.awrite()` in `autourgos-live`.

Audio is written to a temporary WAV file and handed to faster-whisper as a file path rather than a raw array — its own decoding (via `av`/ffmpeg) resamples correctly to the 16kHz Whisper expects regardless of your `sample_rate`, so you don't need to resample yourself.

### With autourgos-openaichat / autourgos-responses / autourgos-agent

None of those packages depend on this one (same reasoning as `autourgos-micinput` — no forced dependency on callers who don't need local speech). Wire them together yourself:

```python
from autourgos_micinput import MicrophoneStream
from autourgos_whisperstt import WhisperSTT
from autourgos_openaichat import OpenAIChatModel

stt = WhisperSTT(model_size="base")
llm = OpenAIChatModel(model="gpt-4o")

async def voice_turn():
    chunks = []
    async with MicrophoneStream(sample_rate=16000) as mic:
        async for chunk in mic:
            chunks.append(chunk)
            if len(chunks) >= 20:  # ~2s
                break
    text = await stt.atranscribe(b"".join(chunks), sample_rate=16000)
    return llm.invoke(text)
```

---

## API Reference

### `WhisperSTT(*, model_size="base", device="cpu", compute_type="int8", language=None)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_size` | `str` | `"base"` | Any faster-whisper model name — `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`, etc. Bigger = more accurate, slower, larger download |
| `device` | `str` | `"cpu"` | `"cpu"` or `"cuda"` (GPU, if available and ctranslate2 has CUDA support) |
| `compute_type` | `str` | `"int8"` | Quantization — `"int8"` (fastest/smallest, good for CPU), `"float16"` (typical for GPU), `"float32"` |
| `language` | `str` | `None` | ISO 639-1 code (e.g. `"en"`) to skip auto-detection; `None` auto-detects per call. Overridable per-call via `transcribe(..., language=...)` |

| Method | Description |
|---|---|
| `transcribe(pcm_bytes, *, sample_rate=16000, channels=1, language=None) -> str` | Blocking. Lazily creates the model on first call, then reuses it. Returns `""` if nothing was recognized (e.g. silence). |
| `atranscribe(pcm_bytes, *, sample_rate=16000, channels=1, language=None) -> str` | Async-safe equivalent — runs `transcribe()` in a worker thread. |

### Errors (`autourgos_whisperstt`)

| Name | Raised when |
|---|---|
| `WhisperSTTError` | Base class |
| `WhisperSTTUnavailableError` | `faster-whisper` isn't installed |

---

## Accuracy note

Live-tested round-trip (Windows SAPI text-to-speech generating a WAV, fed back through this package's recognition): spoken "Testing one two three, this is a speech recognition check." came back as `"Testing 1, 2, 3, this is a speech recognition check."` — near-perfect, including punctuation. `autourgos-windowstt`'s SAPI-based recognition of the identical audio returned `"Testing 1 to 3 this is a speech recognition check"` — noticeably rougher. Use this package when accuracy matters more than the one-time model download; use `autourgos-windowstt` when you want zero setup / zero download / instant offline results on Windows specifically and can tolerate lower accuracy.

---

## License

Apache License 2.0, Copyright (c) 2026 Jitin Kumar Sengar
