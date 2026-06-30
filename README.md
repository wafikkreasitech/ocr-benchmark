# OCR Benchmark

Standalone OCR benchmark for the **ai4db backend** on Indonesian documents.
Runs the same OCR model ai4db runs, against a real Indonesian OCR dataset,
and shows the results in a single-page dashboard.

See `docs/plan.md` for the full design rationale.

---

## What it measures

| Metric | Meaning |
|---|---|
| **Detection F1** | How well OCR localizes text lines vs labelme GT polygons (IoU ≥ 0.5) |
| **CER (matched)** | Character error rate on matched line pairs |
| **WER (matched)** | Word error rate on matched line pairs |
| **Joined-page CER** | Coarse full-page CER (joined GT vs joined prediction) |
| **Throughput** | Images/sec, mean ms/image |
| **Confidence** | OCR score on matched lines |
| **Empty-output rate** | % images where OCR returned zero lines |

## Quick start

```bash
# 1. Install deps
uv sync

# 2. Run benchmark (~2 minutes for 55 images)
uv run python scripts/run_benchmark.py

# 3. Open dashboard
uv run ocr-bench-serve
# → http://127.0.0.1:8765
```

You can also re-run from the dashboard (top-right "Run benchmark" button) or
restrict to specific categories:

```bash
uv run python scripts/run_benchmark.py --category "IDENTITY CARDS" --category NEWSPAPERS
```

## Datasets

The benchmark supports **multiple datasets** behind a single registry. Pick one
from the Run-panel dropdown in the UI, or via the CLI / `.env`.

| Key | Root | Format | Categories |
|---|---|---|---|
| `ind_cn` *(default)* | `IMG_OCR_IND_CN/` | labelme JSON sidecars | 11 (BADGES, BILLS, BOOK CONTENTS, …) |
| `new` | `dataset/dataset/{testing,training}_data/` | FUNSD-form | 2 (`TESTING DATA`, `TRAINING DATA`) |

**From the UI**: top-nav → **Datasets** page lists both with image/line counts
and one-click Run / View-last-results. The Run-panel **Dataset** dropdown
repopulates the Category dropdown live as you switch.

**From the CLI**:
```bash
uv run python scripts/run_benchmark.py --dataset ind_cn
uv run python scripts/run_benchmark.py --dataset new
# --dataset <path> still works as a literal-path override
```

**From `.env`**:
```env
OCR_BENCH_DATASET=new
```

## OCR Models

The benchmark uses **`rapidocr` v3.9.0+** (replaces the old `rapidocr-onnxruntime`).
Models are selectable from the dashboard or via `.env`.

### Available models

| OCR Version | model_type | Speed | Accuracy | Notes |
|---|---|---|---|---|
| **PP-OCRv6** | `tiny` | fastest | good | no japan support |
| | `small` | fast | better | **default** |
| | `medium` | slow | best | recommended for accuracy |
| **PP-OCRv5** | `mobile` | fast | good | per-language rec models |
| | `server` | slow | better | larger models |
| **PP-OCRv4** | `mobile` | fast | good | legacy, stable |
| | `server` | slow | better | legacy, larger |

### Configuring the model

**Option 1 — `.env` file:**
```env
OCR_VERSION=PP-OCRv6
MODEL_TYPE=small
```

**Option 2 — Dashboard:**
Use the model card selector at the top of the page. Click a version card,
then click a size chip (tiny/small/medium). The selection is sent with each
benchmark run.

**Option 3 — CLI:**
```bash
uv run python scripts/run_benchmark.py
# Uses OCR_VERSION and MODEL_TYPE from .env
```

### Model auto-download

Models are downloaded automatically on first use from ModelScope and cached
in `.venv/lib/.../rapidocr/models/`. First run with a new model may take
~10s extra for the download.

### Switching models mid-session

Change the model in the dashboard and click "Run benchmark". The engine
reinitializes with the new model. Previous results remain in `reports/`
until overwritten.

## Outputs

After a run, the following are written (gitignored):

```
reports/
├── summary.csv              ← one row per category + OVERALL row
├── summary.json             ← same data as JSON (includes ocr_version, model_type)
└── per_category/
    ├── identity_cards.json  ← per-image detail + overlay polygons
    ├── newspapers.json
    └── ...
```

`per_category/*.json` includes each line's GT polygon, predicted polygon,
predicted text, confidence, IoU, and status (`matched` / `missed` /
`spurious`) — the data backing the drill-down overlay in the dashboard.

## Why standalone

The whole point of this repo is **moving OCR benchmarking to a different
device with zero ai4db footprint**. We pin the same OCR backend:

| | ai4db | ocr-benchmark |
|---|---|---|
| OCR library | `rapidocr-onnxruntime>=1.4` | `rapidocr>=3.9.0` |
| Init | `RapidOCR()` no-args | `RapidOCR(params={...})` with model selection |
| Default model | PP-OCRv4 (old default) | PP-OCRv6 small (v3.9.0 default) |
| Backend deps | fastapi, piper-tts, sounddevice, scipy, … | minimal: cv2, numpy, pillow, fastapi |

> **Note**: ai4db still uses `rapidocr-onnxruntime` (PP-OCRv4). This benchmark
> upgraded to `rapidocr` v3 (PP-OCRv6) for better accuracy. If you need to
> mirror ai4db exactly, set `OCR_VERSION=PP-OCRv4` and `MODEL_TYPE=mobile`
> in `.env`.

## Deploy on a new device

```bash
git clone <this repo>
cd ocr-benchmark
uv sync
uv run python scripts/run_benchmark.py
uv run ocr-bench-serve
```

No ai4db checkout, no Python 3.13 required. Models download automatically
on first run.

## Version drift — when ai4db updates its OCR backend

1. `cd …/ai4db && uv pip show rapidocr-onnxruntime` → note new version
2. If ai4db upgrades to `rapidocr` v3+: update `OCR_VERSION` / `MODEL_TYPE` in `.env`
3. If ai4db stays on `rapidocr-onnxruntime`: set `OCR_VERSION=PP-OCRv4` here
4. `uv run python scripts/run_benchmark.py` → re-baseline numbers
5. Commit the new config + refreshed `reports/summary.csv`

## Dataset

`./IMG_OCR_IND_CN/` — Appen-style Indonesian OCR corpus.

* 55 images across 11 categories (KTP, koran, kontrak, invoice, whiteboard, novel, etc.)
* Each image has a labelme JSON sidecar with polygon + transcript per text line
* 1,674 GT lines total, vendor-claimed >99% accuracy
* No train/test split — categories are the unit of analysis

## Layout

```
src/ocr_bench/
├── engine.py      ← standalone RapidOCR wrapper (mirrors ai4db.ocr.pipeline.OCRPipeline)
├── dataset.py     ← labelme JSON parser + category iterator
├── matcher.py     ← polygon AABB + greedy IoU ≥ 0.5 matcher
├── metrics.py     ← CER/WER via jiwer, detection P/R/F1
├── runner.py      ← orchestrator, writes reports/
├── api.py         ← FastAPI app (this is what `ocr-bench-serve` runs)
├── config.py      ← pydantic-settings (.env loader)
└── paths.py       ← shared paths

scripts/
└── run_benchmark.py   ← CLI entrypoint

ui/
├── index.html     ← dashboard (warm stone theme)
├── style.css      ← DM Sans + JetBrains Mono, calm palette
└── app.js         ← vanilla JS, no build

reports/           ← generated, gitignored
docs/plan.md       ← design doc
```

## Caveats

* **n = 55 images is small**. The numbers are directional, not statistically
  deep. The dashboard footer says so.
* **Polygon → AABB IoU** is used (not rotated IoU). Fine for line text where
  boxes are tight rectangles; upgrade path noted if slanted text appears.
* **One engine only** — multi-engine comparison is intentionally out of scope.
* **No persistence** beyond JSON files.
* **PP-OCRv6 tiny does not support Japanese** text. Use `small` or `medium`
  if your dataset contains Japanese characters.
* **First run with a new model** downloads ~50-150 MB from ModelScope.
  Subsequent runs use the cached model.

## Post-processing (SymSpell + KBBI)

The corrector is a 3-stage pipeline:
1. **Char normalization** — full-width to ASCII, smart quotes, dashes (always runs when corrector enabled).
2. **KBBI DP word segmentation** — re-inserts lost spaces in joined-up text (opt-in via `ENABLE_WORD_SEGMENTATION`).
3. **SymSpell fuzzy match** — fixes per-word typos (runs only when segmentation changed something or OCR confidence < 0.7).

### Honest finding from the benchmark

We measured all combinations on the 55-image dataset. Results:

| Configuration | CER | WER | Verdict |
|---|---|---|---|
| Corrector OFF (baseline) | **0.121** | **0.385** | best |
| Corrector ON (SymSpell only) | 0.214 | 0.594 | worse on both |
| Corrector + segmentation | 0.176 | 0.963 | CER up, WER destroyed |

**Why segmentation hurts**: Indonesian is highly agglutinative. KBBI contains root words ("urus", "ambil") but not affixed forms ("mengurus", "mengambil"). When the OCR produces "MENGURUSRUMAHTANGGA", segmentation correctly finds "URUS RUMAH TANGGA" but mis-segments "MENGURUS" into "MENG + URUS" because KBBI doesn't have "MENGURUS" as a single entry. The character-level CER drops (good), but the word-level WER explodes (bad).

**Recommendation**: keep `ENABLE_SYMSPELL_CORRECTION=false` and
`ENABLE_WORD_SEGMENTATION=false` by default. SymSpell without frequency data
also tends to introduce wrong "corrections" because all KBBI entries have
frequency 1, so ties resolve alphabetically rather than by commonness.

The corrector is a tool, not a magic win. Toggle it on, look at the per-image
results in the dashboard, and judge for your dataset.
