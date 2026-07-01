# TTS Benchmark — Plan

> **Beauty-design, lazy-where-it-doesn't-hurt, full-where-it-must.**
> Adds the **second stage** of the ai4db pipeline to this repo: after OCR reads
> the document, **TTS speaks it aloud**. This plan measures that TTS stage on
> the exact text the OCR stage produces, and lets a reviewer *listen* to any
> result in the dashboard.
>
> Sibling of [`plan.md`](./plan.md). Same repo, same standalone philosophy,
> same design tokens. Read `plan.md` §4 first — the model-identity / standalone
> rationale there applies here verbatim, just for Piper instead of RapidOCR.

---

## 1 · Why this exists

ai4db is a deafblind assistive platform. Its pipeline is:

```
camera → OCR (RapidOCR) → text → TTS (Piper) → audio  →  blind user hears the page
                                → Braille                →  deafblind user reads it
```

`plan.md` benchmarks the **OCR** box. This plan benchmarks the **TTS** box —
the same Piper voice ai4db actually runs (`id_ID-news_tts-medium`), fed the
same kind of text (OCR output on Indonesian documents). Two honest questions:

1. **Is it fast enough to feel live?** A blind user waits between "point camera"
   and "hear text." We measure that wait: latency, real-time factor, throughput.
2. **Is what it says intelligible?** Optional closed-loop check (§7): speak the
   OCR text, transcribe it back with Whisper, compare. Off by default (pulls a
   heavy dep) — it's the upgrade path, not the baseline.

There is no ground-truth audio and no local MOS panel, so we do **not** claim to
measure "voice quality." We measure what a local machine can measure honestly:
**speed** (always) and **intelligibility drift** (opt-in). Stated plainly in the
UI, same as the OCR plan states "directional; n=55."

---

## 2 · Scope

| In | Out |
|---|---|
| Mirror ai4db's Piper wrapper (30 lines, no ai4db import) | Depending on the `ai4db` package |
| `/api/tts?text=…` → WAV — "🔊 Listen" on any GT/PR line in the OCR drill-down | Streaming/websocket audio |
| TTS speed benchmark over the OCR result corpus → `reports/tts_summary.json` | Training/fine-tuning a voice |
| Metrics: synth latency, **RTF**, chars/sec, audio seconds, first-chunk latency | MOS / subjective quality scoring |
| Dedicated TTS dashboard page (`ui/tts.html`) | Multi-page SPA, build step |
| Optional TTS→STT→CER intelligibility loop (flagged off) | Multi-engine TTS comparison |

### What the benchmark actually measures

Per text sample (one OCR-predicted page, joined):

1. **Synthesis latency** — wall-clock `PiperVoice.synthesize()` for the whole text.
2. **Audio duration** — `pcm_bytes / (2 · sample_rate)` seconds (16-bit mono).
3. **Real-Time Factor (RTF)** — `synth_latency / audio_duration`. **< 1.0 = faster
   than real-time** (can speak live). This is the headline number.
4. **Throughput** — characters synthesized per wall-second.
5. **First-chunk latency** — wall-clock to the *first* streamed chunk (what the
   user actually perceives as "lag before it starts talking").

That's the honest set the setup supports. Everything else is opt-in (§7).

---

## 3 · Architecture (delta over `plan.md`)

Only new/changed files listed — everything else in the repo is untouched.

```
ocr-benchmark/
├── pyproject.toml            # + piper-tts>=1.4  (see §9)
├── .env.example              # + PIPER_VOICE_PATH, TTS_* flags (see §8)
├── models/piper-voices/id/   # id_ID-news_tts-medium.onnx (+ .onnx.json)  ← gitignored, downloaded
├── scripts/
│   ├── download_voice.py     # NEW — fetch the Piper voice (mirrors ai4db's model path)
│   └── run_tts_benchmark.py  # NEW — CLI: uv run python -m scripts.run_tts_benchmark
├── src/ocr_bench/
│   ├── tts_engine.py         # NEW — Piper wrapper, mirrors ai4db/tts/piper_voice.py
│   ├── tts_runner.py         # NEW — iterate OCR results → reports/tts_summary.json
│   └── api.py                # + /api/tts, /api/tts/summary, /api/tts/run, serve tts.html
└── ui/
    ├── tts.html              # NEW — TTS benchmark dashboard
    └── app.js                # + tiny "🔊 Listen" hook in the per-image drill-down
```

Ponytail rule (same as `plan.md`): **one** engine wrapper, **one** runner,
**one** new page. No audio-format abstraction (WAV only), no voice registry
(one voice — the one ai4db ships), no queue/worker pool (a `ThreadPoolExecutor(1)`
mirrors ai4db and is enough for a batch of 55).

---

## 4 · Engine wrapper — standalone, mirrors ai4db's Piper backend

### Decision: mirror, don't import (same as `plan.md` §4)

ai4db's TTS is a thin wrapper over the `piper` package
(`src/ai4db/tts/piper_voice.py` → `PiperVoice.load()` + `SynthesisConfig`). We
copy that ~30-line wrapper rather than importing `ai4db` (which would force
Python 3.13, sounddevice, scipy, whisper, the whole tree). The wrapper is small,
stable, and owning it costs a 5-line diff to sync.

### Model identity guarantee

ai4db's Indonesian profile uses `models/piper-voices/id/id_ID-news_tts-medium.onnx`
(from `config.py` `LANGUAGE_PROFILES["id"]`). We benchmark **that exact voice
file** — download it to the same relative path and pin the wrapper to the same
`piper-tts` major. Unlike RapidOCR, the Piper voice is **not** bundled in the
wheel: `scripts/download_voice.py` fetches it (from the huggingface
`rhasspy/piper-voices` mirror) into `models/piper-voices/id/`. Both the `.onnx`
and its `.onnx.json` config are required — Piper loads the JSON for sample rate,
phoneme map, and defaults.

### Sync contract

If ai4db swaps the voice (e.g. `id_ID-news_tts-medium` → a new one) or bumps
`piper-tts` across a major, mirror it here: change one path constant + one pin,
re-download, re-run. That re-baselines "what ai4db sounds like today."

### Implementation (copied from ai4db, trimmed to what the benchmark needs)

```python
# src/ocr_bench/tts_engine.py
import time
from dataclasses import dataclass, field
from pathlib import Path

from piper import PiperVoice   # ponytail: same package + wrapper shape as ai4db

@dataclass
class TTSResult:
    text: str
    synth_ms: float        # wall-clock for the whole synthesize() call
    first_chunk_ms: float  # wall-clock to the first audio chunk (perceived lag)
    audio_seconds: float   # pcm_len / (2 * sample_rate)  — 16-bit mono
    sample_rate: int
    n_chars: int

    @property
    def rtf(self) -> float:                       # < 1.0 == faster than real time
        return self.synth_ms / 1000 / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def chars_per_sec(self) -> float:
        return self.n_chars / (self.synth_ms / 1000) if self.synth_ms else 0.0

@dataclass
class TTSEngine:
    """Standalone Piper wrapper. Same backend + voice as ai4db, but times the
    synthesis and returns metrics + raw PCM (no ai4db import)."""
    voice_path: str
    voice: PiperVoice = field(init=False)

    def __post_init__(self):
        self.voice = PiperVoice.load(self.voice_path)   # raises if .onnx/.json missing

    def synthesize(self, text: str) -> tuple[bytes, TTSResult]:
        t0 = time.perf_counter()
        parts: list[bytes] = []
        sample_rate = 22050
        first_chunk_ms = 0.0
        for i, chunk in enumerate(self.voice.synthesize(text)):
            if i == 0:
                first_chunk_ms = (time.perf_counter() - t0) * 1000
            parts.append(chunk.audio_int16_bytes)
            sample_rate = chunk.sample_rate
        pcm = b"".join(parts)
        synth_ms = (time.perf_counter() - t0) * 1000
        return pcm, TTSResult(
            text=text, synth_ms=synth_ms, first_chunk_ms=first_chunk_ms,
            audio_seconds=len(pcm) / (2 * sample_rate),
            sample_rate=sample_rate, n_chars=len(text),
        )
```

WAV wrapping for the HTTP response is the same 8-line `wave` helper ai4db uses
(`routers/tts.py::_pcm_to_wav`) — copy it into `api.py`. No new dep.

---

## 5 · What text do we speak?

The **OCR predicted text**, joined per page — because that is *exactly* what
ai4db's TTS receives in production (OCR output, warts and all). Source of truth:
the already-generated `reports/per_category/*.json` from the OCR benchmark. Each
image's overlays carry `pr_text` (and `gt_text`). The TTS runner joins predicted
lines per page and speaks that.

This makes the two benchmarks compose: **OCR CER tells you how wrong the words
are; TTS RTF tells you how fast they get spoken.** Together they characterize the
whole ai4db reading experience on Indonesian docs.

Toggle `TTS_SOURCE=gt` to speak ground-truth instead (isolates TTS speed from OCR
errors — useful when OCR text is garbage and skews character counts).

---

## 6 · Metrics & aggregation

| Metric | What it tells you |
|---|---|
| **RTF (mean/median)** | Faster-than-real-time? The headline. < 1.0 = live-capable. |
| **Synth latency ms** | Absolute wait to synthesize a full page. |
| **First-chunk latency ms** | Perceived lag before audio starts (streaming). |
| **Chars/sec** | Synthesis throughput, length-normalized. |
| **Audio seconds** | How long the page takes to *hear* (independent of synth speed). |
| **Failure rate** | % samples that raised (empty text, phoneme errors). |

Aggregated per OCR category + overall, one row each, written to
`reports/tts_summary.json` (+ a flat `reports/tts_summary.csv`, 6-line writer,
no pandas — same as `plan.md` §7). Empty predicted pages are skipped and counted.

---

## 7 · Optional intelligibility loop (opt-in, OFF by default)

The honest "did it say the right thing" check, if you want it:

```
OCR pr_text  →  Piper TTS  →  WAV  →  Whisper STT  →  transcript  →  CER vs pr_text
```

Low CER ⇒ the synthesized speech is intelligible enough that an ASR recovers the
words. It's a **proxy**, not human MOS, and it pulls in a Whisper dependency
(`openai-whisper` or `faster-whisper`) that breaks the "just `uv sync`"
promise — so it lives behind `TTS_INTELLIGIBILITY_CHECK=false` and its dep is an
optional extra (`[project.optional-dependencies] intelligibility = [...]`).

```
ponytail: intelligibility loop is the upgrade path, not the baseline.
          Build it only when someone asks "but does it sound right?" — the
          speed benchmark answers the shipping question on its own.
```

Do **not** build this in the first pass. Ship §1–§6, note this exists.

---

## 8 · Config (.env additions)

```bash
# ── TTS ─────────────────────────────────────────────────────────────
# Piper voice model — same path ai4db uses (config.py LANGUAGE_PROFILES["id"]).
# Download with: uv run python -m scripts.download_voice
PIPER_VOICE_PATH=models/piper-voices/id/id_ID-news_tts-medium.onnx

# Which text to speak in the benchmark: pred (OCR output, realistic) | gt (clean)
TTS_SOURCE=pred

# Intelligibility loop (TTS→STT→CER). Off — needs the `intelligibility` extra.
TTS_INTELLIGIBILITY_CHECK=false
WHISPER_MODEL_PATH=models/whisper/ggml-tiny.id.bin
```

One flag added to `config.py`'s existing pydantic-settings block. No new config
system.

---

## 9 · Dependencies (one line added)

```toml
dependencies = [
  # … existing OCR deps unchanged …
  "piper-tts>=1.4",              # same TTS backend ai4db pins; ships onnxruntime-compatible
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-timeout>=2.0"]
intelligibility = ["faster-whisper>=1.0"]   # only for §7; not installed by default
```

`onnxruntime` is already a dep (RapidOCR uses it) — Piper reuses it. So the net
addition to the base install is **one package**. No sounddevice (we don't *play*
audio server-side — the browser plays the WAV), no scipy.

---

## 10 · API surface (additions)

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/tts?text=…` | Synthesize arbitrary text → `audio/wav`. Powers "🔊 Listen". |
| `POST` | `/api/tts/run?dataset=…` | Run the TTS benchmark over OCR results; writes `tts_summary.json`. |
| `GET`  | `/api/tts/summary` | Per-category + overall TTS metrics JSON. |
| `GET`  | `/tts` | The TTS dashboard page (`ui/tts.html`). |

`GET /api/tts` is synchronous (a single line synthesizes in tens of ms). The
batch `POST /api/tts/run` reuses the existing background-task + `/api/progress`
pattern already in `api.py` — no new progress machinery.

`/api/tts` returns **503** with a clear message if the voice model is missing
(mirrors ai4db's `routers/tts.py`), so the button degrades gracefully to "voice
model not downloaded — run `download_voice`" instead of a stack trace.

---

## 11 · UI

### 11a · "🔊 Listen" in the existing OCR drill-down (tiny hook, `app.js`)

Each GT/PR line in the per-image "SAMPLE COMPARISON" panel gets a small speaker
button. Click → `new Audio('/api/tts?text=' + encodeURIComponent(line)).play()`.
~5 lines, native `<audio>`, no library. This is the single most demo-friendly
feature: a reviewer clicks a KTP line and *hears the platform read it*.

### 11b · TTS dashboard (`ui/tts.html`) — matches OCR dashboard tokens

```
┌──────────────────────────────────────────────────────────────────┐
│  TTS BENCHMARK · ai4db Piper · Indonesian voice (news_tts-medium) │
│  ──────────────────────────────────────────────────────────────  │
│   ▌ RUN TTS BENCHMARK              speaks OCR output · 55 pages    │
│                                                                  │
│   OVERALL                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │   RTF    │  │ SYNTH ms │  │FIRST CHUNK│  │ CHARS/s  │         │
│   │  0.18    │  │   340    │  │   45 ms  │  │  4,200   │         │
│   │ ✔ faster │  │ per page │  │  to talk │  │          │         │
│   │ than live│  │          │  │          │  │          │         │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                                                  │
│   RTF BY CATEGORY            SYNTH LATENCY BY CATEGORY           │
│   ▇▇▇▇▇▇▇▇▇▇▇                ▇▇▇▇▇▇▇▇▇▇▇                       │
│                                                                  │
│   PER-CATEGORY TABLE                                              │
│   category         pages  chars   RTF   synth  1st-chunk  audio-s │
│   NEWSPAPERS          5    9,120  0.21   1180ms   52ms      42.1   │
│   IDENTITY CARDS      5      840  0.14    120ms   38ms       5.9   │
│   ...                                                            │
│                                                                  │
│   ─────  🔊 CLICK ANY ROW TO HEAR A SAMPLE PAGE  ─────           │
└──────────────────────────────────────────────────────────────────┘
```

Same palette, glass cards, tabular-nums, hand-rolled SVG bars as the OCR page
(reuse `style.css` — zero new tokens). A header link toggles OCR ⇄ TTS
dashboards. The RTF card shows a "✔ faster than live" / "⚠ slower than live"
badge off the 1.0 threshold — the one number a non-ML reviewer needs.

---

## 12 · Phased tasks

| # | Task | Files | Done when |
|---|---|---|---|
| 1 | Add `piper-tts` dep + voice download script | `pyproject.toml`, `scripts/download_voice.py` | `uv run python -m scripts.download_voice` lands `.onnx`+`.json` |
| 2 | Piper wrapper (mirror ai4db, add timing) | `tts_engine.py` | `synthesize("halo")` returns PCM + `TTSResult` with RTF |
| 3 | `GET /api/tts` → WAV (+ 503 if no model) | `api.py` | browser plays `/api/tts?text=halo` |
| 4 | "🔊 Listen" buttons in OCR drill-down | `ui/app.js` | clicking a GT/PR line speaks it |
| 5 | TTS runner over OCR results → `tts_summary.json`+`.csv` | `tts_runner.py`, `scripts/run_tts_benchmark.py` | `reports/tts_summary.json` exists |
| 6 | `POST /api/tts/run` + `GET /api/tts/summary` | `api.py` | curl summary returns per-category metrics |
| 7 | TTS dashboard page | `ui/tts.html`, small `app.js` additions | `/tts` renders table + RTF cards + row-click plays a page |
| 8 | README: download voice + run TTS bench | `README.md` | copy-paste works on a clean checkout |
| 9 | Smoke test | `tests/test_tts_smoke.py` | asserts RTF in `(0, 5)` and PCM non-empty on one short string |
| 10 | *(optional, deferred)* intelligibility loop | `tts_runner.py` `+ whisper` | only if requested — see §7 |

---

## 13 · Risks & deliberate skips

* **Voice model isn't in the wheel** — must be downloaded (unlike RapidOCR's
  bundled pack). Mitigated by `download_voice.py` + a 503 that tells you to run
  it. This is the one break from "just `uv sync`"; documented in README.
* **No objective voice-quality metric** — we measure speed, not MOS. Stated in
  the UI. Intelligibility loop (§7) is the honest-but-heavy upgrade, opt-in.
* **RTF depends on the machine** — a Pi vs a laptop differ wildly. The number is
  "what *this device* does," same caveat as OCR latency. Surface the host in the
  report footer.
* **`piper-tts` version drift from ai4db** — mirror the pin + voice path in the
  same commit ai4db changes them (sync contract §4). 5-line diff.
* **Speaking garbage OCR text** — long strings of misrecognized characters can
  produce odd phonemes / inflated char counts. `TTS_SOURCE=gt` isolates TTS
  speed from OCR quality when needed.
* **No server-side playback** — deliberately. The browser plays the WAV; the
  server never touches `sounddevice`. Keeps the dep tree and the deploy small.

---

## 14 · Definition of done

* `uv run python -m scripts.download_voice` fetches the Indonesian Piper voice.
* `uv run python -m scripts.run_tts_benchmark` produces `reports/tts_summary.json`
  and `reports/tts_summary.csv` in under a minute on this machine.
* `/tts` dashboard shows overall RTF / synth latency / first-chunk / chars-sec,
  a per-category table, and row-click plays a sample page.
* In the OCR per-image view, "🔊 Listen" speaks any GT/PR line.
* `/api/tts?text=…` returns valid WAV; returns 503 with a clear message if the
  voice model is missing.
* `uv run pytest tests/test_tts_smoke.py` exits 0.
* README documents the voice download + run steps.

Then ship it. Intelligibility loop (§7) stays on the shelf until someone asks
"but does it *sound* right?"
