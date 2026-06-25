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
# 1. Install deps (includes rapidocr-onnxruntime==1.4.4 — same as ai4db)
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

## Outputs

After a run, the following are written (gitignored):

```
reports/
├── summary.csv              ← one row per category + OVERALL row
├── summary.json             ← same data as JSON
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
device with zero ai4db footprint**. We pin the same library ai4db uses:

| | ai4db | ocr-benchmark |
|---|---|---|
| OCR library | `rapidocr-onnxruntime>=1.4` (resolves to 1.4.4) | `rapidocr-onnxruntime==1.4.4` (locked) |
| Init | `RapidOCR()` no-args | `RapidOCR()` no-args |
| Model pack | PP-OCRv4 (default in wheel) | PP-OCRv4 (default in wheel) |
| Backend deps | fastapi, piper-tts, sounddevice, scipy, … | minimal: cv2, numpy, pillow, fastapi |

**Result**: same library version → same model pack → same OCR behavior.
Numbers you see in the dashboard represent what ai4db produces on these images.

## Deploy on a new device

```bash
git clone <this repo>
cd ocr-benchmark
uv sync
uv run python scripts/run_benchmark.py
uv run ocr-bench-serve
```

No ai4db checkout, no Python 3.13 required, no external model download
(model pack ships inside the `rapidocr-onnxruntime` wheel).

## Version drift — when ai4db updates its OCR backend

1. `cd …/ai4db && uv pip show rapidocr-onnxruntime` → note new version
2. Edit `pyproject.toml` → bump pin to the new version
3. `uv lock && uv sync` → new model pack downloaded
4. `uv run python scripts/run_benchmark.py` → re-baseline numbers
5. Commit the new pin + refreshed `reports/summary.csv`

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
└── paths.py       ← shared paths

scripts/
└── run_benchmark.py   ← CLI entrypoint

ui/
├── index.html     ← dashboard
├── style.css      ← design tokens, dark glass
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