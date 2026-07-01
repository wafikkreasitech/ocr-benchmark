"""Standalone Piper TTS wrapper — mirrors ai4db's tts/piper_voice.py.

Same package (`piper`), same voice ai4db runs (id_ID-news_tts-medium), but
times the synthesis and returns metrics + raw PCM. We deliberately do NOT
import ai4db (see docs/plan-tts.md §4) — this is a ~40-line copy so the repo
stays installable with a plain `uv sync`.

The one measurable, honest question this answers: is synthesis faster than
real-time (RTF < 1.0), i.e. can the platform speak a page live.
"""
from __future__ import annotations

import time
import wave
from dataclasses import dataclass, field
from io import BytesIO

from piper import PiperVoice


@dataclass
class TTSResult:
    text: str
    synth_ms: float          # wall-clock for the whole synthesize() call
    first_chunk_ms: float    # wall-clock to the first audio chunk (perceived lag)
    audio_seconds: float     # pcm_len / (sample_width * channels * sample_rate)
    sample_rate: int
    n_chars: int

    @property
    def rtf(self) -> float:
        """Real-time factor. < 1.0 == synthesized faster than it plays."""
        return (self.synth_ms / 1000) / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.n_chars / (self.synth_ms / 1000) if self.synth_ms else 0.0


@dataclass
class TTSEngine:
    """Loads the Piper voice once; synthesize() returns (pcm_bytes, TTSResult).

    Raises at construction if the .onnx / .onnx.json is missing — callers
    (api.py) turn that into a 503 with a "run download_voice" message.
    """
    voice_path: str
    voice: PiperVoice = field(init=False)

    def __post_init__(self) -> None:
        self.voice = PiperVoice.load(self.voice_path)

    def synthesize(self, text: str) -> tuple[bytes, TTSResult]:
        t0 = time.perf_counter()
        parts: list[bytes] = []
        sample_rate = 22050
        sample_width = 2   # 16-bit
        channels = 1
        first_chunk_ms = 0.0
        for i, chunk in enumerate(self.voice.synthesize(text)):
            if i == 0:
                first_chunk_ms = (time.perf_counter() - t0) * 1000
            parts.append(chunk.audio_int16_bytes)
            sample_rate = chunk.sample_rate
            sample_width = chunk.sample_width
            channels = chunk.sample_channels
        pcm = b"".join(parts)
        synth_ms = (time.perf_counter() - t0) * 1000
        bytes_per_sec = sample_width * channels * sample_rate
        return pcm, TTSResult(
            text=text,
            synth_ms=synth_ms,
            first_chunk_ms=first_chunk_ms,
            audio_seconds=len(pcm) / bytes_per_sec if bytes_per_sec else 0.0,
            sample_rate=sample_rate,
            n_chars=len(text),
        )


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM in a WAV container. Same helper as ai4db."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


if __name__ == "__main__":
    # ponytail: runnable self-check — proves the wrapper + voice actually speak.
    # Needs the voice downloaded (scripts/download_voice.py).
    import sys
    from .config import get_settings

    engine = TTSEngine(get_settings().piper_voice_path)
    pcm, r = engine.synthesize("Halo, ini uji coba benchmark.")
    assert pcm, "no audio produced"
    assert 0.0 < r.rtf < 5.0, f"RTF out of sane range: {r.rtf}"
    print(f"OK  chars={r.n_chars}  synth={r.synth_ms:.0f}ms  "
          f"audio={r.audio_seconds:.2f}s  RTF={r.rtf:.3f}  "
          f"first_chunk={r.first_chunk_ms:.0f}ms  {r.chars_per_sec:.0f} chars/s",
          file=sys.stderr)
