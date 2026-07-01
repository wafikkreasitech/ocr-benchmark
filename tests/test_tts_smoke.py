"""Smoke test — TTS engine sanity.

Skipped when the Piper voice isn't downloaded (scripts/download_voice.py), so a
CI box without the ~63MB model still passes. When present, asserts synthesis
produces non-empty audio at a sane real-time factor.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ocr_bench.config import get_settings
from ocr_bench.tts_engine import TTSEngine, pcm_to_wav

_VOICE = Path(get_settings().piper_voice_path)
_needs_voice = pytest.mark.skipif(
    not _VOICE.exists(),
    reason=f"Piper voice not downloaded at {_VOICE} — run scripts/download_voice.py",
)


@_needs_voice
def test_synthesize_produces_audio():
    engine = TTSEngine(str(_VOICE))
    pcm, r = engine.synthesize("Halo, ini uji coba benchmark.")
    assert pcm, "no PCM produced"
    assert r.n_chars > 0
    assert r.audio_seconds > 0
    assert 0.0 < r.rtf < 5.0, f"RTF out of sane range: {r.rtf}"
    assert r.first_chunk_ms > 0


@_needs_voice
def test_wav_header():
    engine = TTSEngine(str(_VOICE))
    pcm, r = engine.synthesize("Halo.")
    wav = pcm_to_wav(pcm, r.sample_rate)
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE", "not a valid WAV container"
    assert len(wav) > len(pcm), "WAV should wrap PCM with a header"
