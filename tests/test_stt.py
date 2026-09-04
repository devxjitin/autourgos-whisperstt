"""
Tests for autourgos_whisperstt.WhisperSTT against a fake faster_whisper
module - no real model download or inference involved.
"""

import sys
import types

import pytest

from autourgos_whisperstt.stt import WhisperSTT, WhisperSTTUnavailableError


def _make_pcm(seconds: float = 0.1, sample_rate: int = 16000) -> bytes:
    n = int(seconds * sample_rate)
    return (b"\x00\x01" * n)[: n * 2]


class FakeSegment:
    def __init__(self, text):
        self.text = text


class FakeInfo:
    language = "en"
    language_probability = 1.0


class FakeWhisperModel:
    """Stand-in for faster_whisper.WhisperModel."""

    instances = []

    def __init__(self, model_size, device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.transcribe_calls = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio_path, language=None):
        self.transcribe_calls.append((audio_path, language))
        return [FakeSegment("hello"), FakeSegment("world")], FakeInfo()


@pytest.fixture(autouse=True)
def reset_instances():
    FakeWhisperModel.instances.clear()
    yield
    FakeWhisperModel.instances.clear()


@pytest.fixture
def fake_faster_whisper(monkeypatch):
    fake_module = types.ModuleType("faster_whisper")
    fake_module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)
    return fake_module


@pytest.fixture
def no_faster_whisper(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # forces ImportError


def test_constructor_does_not_load_model(fake_faster_whisper):
    WhisperSTT(model_size="tiny")
    assert FakeWhisperModel.instances == []  # lazy: no model created yet


def test_transcribe_lazily_loads_model_once(fake_faster_whisper):
    stt = WhisperSTT(model_size="tiny", device="cpu", compute_type="int8")
    text1 = stt.transcribe(_make_pcm(), sample_rate=16000)
    text2 = stt.transcribe(_make_pcm(), sample_rate=16000)

    assert text1 == "hello world"
    assert text2 == "hello world"
    assert len(FakeWhisperModel.instances) == 1  # model created once, reused
    model = FakeWhisperModel.instances[0]
    assert model.model_size == "tiny"
    assert len(model.transcribe_calls) == 2


def test_transcribe_writes_a_wav_file_path(fake_faster_whisper):
    stt = WhisperSTT()
    stt.transcribe(_make_pcm(), sample_rate=16000)
    model = FakeWhisperModel.instances[0]
    audio_path, _language = model.transcribe_calls[0]
    assert audio_path.endswith(".wav")


def test_transcribe_passes_language_override(fake_faster_whisper):
    stt = WhisperSTT(language="en")
    stt.transcribe(_make_pcm(), language="fr")
    model = FakeWhisperModel.instances[0]
    _audio_path, language = model.transcribe_calls[0]
    assert language == "fr"  # per-call override wins over the constructor default


def test_transcribe_uses_constructor_language_by_default(fake_faster_whisper):
    stt = WhisperSTT(language="en")
    stt.transcribe(_make_pcm())
    model = FakeWhisperModel.instances[0]
    _audio_path, language = model.transcribe_calls[0]
    assert language == "en"


def test_without_faster_whisper_raises_unavailable_error(no_faster_whisper):
    stt = WhisperSTT()
    with pytest.raises(WhisperSTTUnavailableError, match="autourgos-whisperstt\\[whisper\\]"):
        stt.transcribe(_make_pcm())


@pytest.mark.asyncio
async def test_atranscribe_offloads_to_thread(fake_faster_whisper):
    stt = WhisperSTT()
    text = await stt.atranscribe(_make_pcm(), sample_rate=16000)
    assert text == "hello world"
