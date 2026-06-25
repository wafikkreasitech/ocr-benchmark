# OCR Benchmark — Plan

> **Beauty-design, lazy-where-it-doesn't-hurt, full-where-it-must.**
> Standalone OCR benchmark · Dataset: `D:\kerjaan\kreasi\kimfu\ocr-benchmark\IMG_OCR_IND_CN`
>
> **Portability**: this repo does **not** depend on the `ai4db` package. It
> pins the same OCR backend (`rapidocr-onnxruntime`) directly, so it runs on
> any device with just `uv sync`. See §4 for the rationale.

---

## 1 · Why this exists

ai4db ships an OCR pipeline (`OCRPipeline` + `rapidocr_onnxruntime`) used in a
deafblind assistive platform. The team needs to know **how accurate that OCR
backend is on real Indonesian documents** — KTP, koran, kontrak, invoice,
whiteboard, novel. **This repo measures it** and ships a dashboard that makes
the answer legible to a non-ML reviewer. We use the same backend library as
ai4db, but without depending on the ai4db package — see §4.

### Dataset that fits the target

`IMG_OCR_IND_CN/` (Appen-style multi-language OCR corpus, Indonesian subset):

| Property | Value |
|---|---|
| Language | **Latin Indonesian** (target ✔) |
| Images | 55 total (51 JPG + 4 PNG) across 11 categories |
| Categories | BADGES, BILLS, BOOK CONTENTS, CONTRACTS, FORMS, IDENTITY CARDS, NEWSPAPERS, NOTES, PAPERS, TRADE DOCUMENTS, WHITEBOARD |
| Ground truth | **labelme JSON sidecar** per image |
| GT shape | `shapes[].points` (polygon or rectangle) + `shapes[].label` (transcript) |
| Total transcripts | **1,674 lines**, 1,464 unique — vendor-claimed **>99%** accuracy on both position and content |
| Splits | None — flat per category. We use **category as split** in the UI. |

Sample transcripts (real Latin Indonesian — KTP, koran, kontrak, akademik):
> `PROVINSI JAWA BARAT` · `MIMI/UPEKS` · `PT. Angin Sepoi` ·
> `Adriannoor, Fahrian. 2015. Seruyan Optimis Bebas Filariasis 2020` ·
> `Perselisihan` · `Biaya Dokumen dan lain-lain`

This is the **right** dataset for the target. Earlier draft used a Hanacaraka
YOLO set — wrong target, wrong GT format, dropped.

---

## 2 · Scope

| In | Out |
|---|---|
| Reuse `ai4db.ocr.pipeline.OCRPipeline` engine | Re-training any model |
| Parse labelme JSON → polygons + transcripts per image | Manual annotation |
| Per-line: detect, recognize, IoU-match against GT polygons | Multi-engine comparison (one engine under test) |
| Per-image: CER, WER, line F1, confidence mean, latency | Online/persistent storage |
| Aggregate: per-category + overall CER/WER/F1 + throughput | Auth, multi-user |
| Beautiful single-page dashboard | Multi-page React app, build pipeline |

### What the benchmark actually measures

1. **Detection** — for each GT polygon, did the engine emit a text line whose
   polygon overlaps at IoU ≥ 0.5? → per-image TP/FP/FN.
2. **Recognition** — for matched pairs, normalize both strings and compute
   **CER** (character error rate) and **WER** (word error rate).
3. **End-to-end** — for each image, joined predicted text vs joined GT text
   gives a coarse full-page CER (useful but noisy).
4. **Speed** — wall-clock per image, throughput.

This is the honest measurement the dataset supports.

---

## 3 · Architecture

```
ocr-benchmark/
├── pyproject.toml            # uv; deps: rapidocr-onnxruntime, opencv-python, pillow, fastapi, uvicorn, jiwer
├── README.md
├── src/
│   └── ocr_bench/
│       ├── __init__.py
│       ├── engine.py         # Wraps rapidocr (same backend ai4db uses); yields (polygon, text, score)
│       ├── dataset.py        # labelme JSON parser → (image_path, [(poly, text)])
│       ├── matcher.py        # IoU≥0.5 polygon matching (Greedy by score)
│       ├── metrics.py        # CER, WER, detection P/R/F1
│       ├── runner.py         # Iterate categories → write reports/*.json + summary.csv
│       └── api.py            # FastAPI: /api/run, /api/results/<category>, /api/summary, /api/image/...
├── scripts/
│   └── run_benchmark.py      # CLI: uv run python -m scripts.run_benchmark
├── reports/                  # Generated, gitignored
│   ├── per_category/<name>.json
│   └── summary.csv
├── docs/
│   └── plan.md               # ← this file
└── ui/
    ├── index.html            # Single page, vanilla, no build
    ├── style.css             # Design tokens, dark glass
    └── app.js                # Fetches + renders, hand-rolled SVG charts
```

Ponytail rule: **no** plugin registry, **no** config file, **no** factory.
One engine, one parser, one matcher, one runner, one API, one HTML page.

---

## 14 · Post-processing findings (recorded honestly)

We measured the SymSpell + KBBI corrector on the same 55-image dataset:

| Configuration | CER | WER |
|---|---|---|
| Baseline (no corrector) | 0.121 | 0.385 |
| Corrector ON — SymSpell only | 0.214 | 0.594 |
| Corrector ON — char norm + KBBI segmentation + SymSpell | 0.176 | 0.963 |

**Conclusion**: the corrector as built is a **net loss** on this dataset, and
it stays off by default (`ENABLE_SYMSPELL_CORRECTION=false`,
`ENABLE_WORD_SEGMENTATION=false`).

Why KBBI segmentation hurts WER: Indonesian is agglutinative. KBBI contains
root words ("urus", "ambil") but not affixed forms ("mengurus", "mengambil").
When OCR emits `"MENGURUSRUMAHTANGGA"`, segmentation correctly finds
`"URUS RUMAH TANGGA"` but mis-segments `"MENGURUS"` into `"MENG + URUS"` —
CER drops (good), WER explodes (bad).

Why SymSpell alone hurts CER: KBBI CSV has no frequency column, so every
entry gets frequency 1. SymSpell tie-breaks alphabetically; suggestions for
already-correct words often come back as different wrong KBBI entries
(`"Hello" → "tello"`).

**Upgrade path** (not built — would need a proper Indonesian stemmer or
frequency data): integrate Sastrawi stemming for affixed-form lookup before
SymSpell.

---

## 4 · Engine wrapper (standalone, mirrors ai4db backend)

### Decision: standalone, not a dependency on `ai4db`

ai4db's `OCRPipeline` is a thin wrapper around `rapidocr_onnxruntime.RapidOCR`.
We deliberately **do not import it**. Reasons:

* The repo's purpose is to move OCR benchmarking to a **different device
  with zero ai4db footprint** — depending on `ai4db` would force Python 3.13,
  fastapi, piper-tts, sounddevice, scipy, and the whole ai4db dep tree.
* ai4db's wrapper discards bounding boxes (it joins all text into one
  string). The benchmark needs boxes + scores. So we can't just import it
  anyway — we'd have to fork the logic. Better to own a 30-line wrapper.
* The backend library (`rapidocr-onnxruntime`) ships its own ONNX model pack
  in the wheel. No separate model download, no Git LFS, no ai4db checkout.

### Model identity guarantee

The whole point is to **benchmark what ai4db actually runs**. ai4db uses
`RapidOCR()` with no-args, which loads the **default config and model pack**
shipped in `rapidocr-onnxruntime`'s wheel — currently **PP-OCRv4** (det/rec/cls).
We pin the **exact same version** so the model pack is byte-identical.

* **ai4db pin**: `rapidocr-onnxruntime>=1.4` (currently resolved to **1.4.4**)
* **ocr-benchmark pin**: `rapidocr-onnxruntime==1.4.4` (locked)

If the team upgrades ai4db's `rapidocr-onnxruntime` to a newer version, update
the pin here in lockstep and rerun the benchmark — that re-baselines the
"what ai4db does today" number. See §12.

### Sync contract

If ai4db ever changes its OCR backend (different model pack, different lib,
post-processing), **manually mirror the change here**. A 5-line diff in
`engine.py` + a version bump in `pyproject.toml` is the entire upgrade cost.

### Implementation

```python
# src/ocr_bench/engine.py
import time
from dataclasses import dataclass
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR

@dataclass
class LinePrediction:
    polygon: list[list[float]]   # 4-pt box from rapidocr
    text: str
    score: float

@dataclass
class PagePrediction:
    image: str
    lines: list[LinePrediction]
    elapsed_ms: float

class BenchEngine:
    """Standalone OCR wrapper. Same backend ai4db uses (rapidocr-onnxruntime
    pinned to the same version), but preserves polygons + scores for IoU
    matching and CER/WER scoring."""
    def __init__(self):
        self._ocr = RapidOCR()  # ponytail: default config mirrors ai4db

    def predict(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        result, _ = self._ocr(str(image_path))
        lines = [LinePrediction(poly, txt, sc) for (poly, txt, sc) in (result or [])]
        return PagePrediction(
            image=image_path.name,
            lines=lines,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )
```

---

## 5 · Dataset parser

labelme JSON is small and predictable. One parse, one shape.

```python
# src/ocr_bench/dataset.py
import json
from pathlib import Path

@dataclass
class GroundTruthLine:
    polygon: list[list[float]]
    text: str

@dataclass
class GroundTruthPage:
    image_path: Path
    category: str
    lines: list[GroundTruthLine]

def load_category(category_dir: Path) -> list[GroundTruthPage]:
    pages = []
    for jpg in sorted(category_dir.glob("*.jpg")):
        json_path = jpg.with_suffix(".json")
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        lines = [
            GroundTruthLine(polygon=s["points"], text=s["label"])
            for s in data.get("shapes", [])
            if s.get("label", "").strip()
        ]
        pages.append(GroundTruthPage(jpg, category_dir.name, lines))
    return pages
```

---

## 6 · Matcher (detection)

Greedy IoU ≥ 0.5 matching sorted by prediction score (descending) — standard
COCO-style. Polygon → axis-aligned bbox for IoU (lines are tight rectangles;
rotated-IoU is overkill, see §10).

```python
# src/ocr_bench/matcher.py  (sketch)
def match(gt_lines, pr_lines, iou_threshold=0.5):
    """Returns (matches, unmatched_gt, unmatched_pr)."""
    pairs = sorted(
        ((iou(g.poly, p.polygon), g, p) for g in gt_lines for p in pr_lines),
        key=lambda x: -x[0],
    )
    used_g, used_p, matches = set(), set(), []
    for iou_val, g, p in pairs:
        if iou_val < iou_threshold: break
        if id(g) in used_g or id(p) in used_p: continue
        matches.append((g, p))
        used_g.add(id(g)); used_p.add(id(p))
    unmatched_gt = [g for g in gt_lines if id(g) not in used_g]
    unmatched_pr = [p for p in pr_lines if id(p) not in used_p]
    return matches, unmatched_gt, unmatched_pr
```

---

## 7 · Metrics

| Metric | What it tells you |
|---|---|
| **Detection Precision** | Of predicted lines, how many overlap a real line? |
| **Detection Recall** | Of real lines, how many did OCR find? |
| **Detection F1** | Single-number summary of layout detection |
| **CER (per line)** | Character error rate on matched pairs — `jiwer.cer` |
| **WER (per line)** | Word error rate on matched pairs — `jiwer.wer` |
| **End-to-end CER** | Joined-page CER (coarse, surfaced as "noisy" in UI) |
| **Mean confidence** | OCR score on matched lines |
| **Throughput** | images/sec, mean ms/image |
| **Empty-output rate** | % images where OCR returned 0 lines |

All metrics computed per category and aggregated overall. CSV summary has one
row per category with all the above.

**Normalization before CER/WER**: lowercase, strip whitespace, collapse runs
of whitespace, NFKC normalize. This is the standard fair comparison.

---

## 8 · UI mockup — beauty with restraint

Single page, two views: **Overview · Per-category · Per-image**.

### Overview (default)

```
┌──────────────────────────────────────────────────────────────────┐
│  OCR BENCHMARK · ai4db × Indonesian documents                    │
│  ──────────────────────────────────────────────────────────────  │
│                                                                  │
│   ▌ RUN ALL                       Last run: 14s ago · 55 imgs    │
│                                                                  │
│   OVERALL                                                          │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│   │ DETECT F1│  │  CER     │  │  WER     │  │  SPEED   │         │
│   │  0.71    │  │  0.084   │  │  0.31    │  │ 1.2 im/s │         │
│   └──────────┘  └──────────┘  └──────────┘  └──────────┘         │
│                                                                  │
│   CER BY CATEGORY            DETECTION F1 BY CATEGORY            │
│   ▇▇▇▇▇▇▇▇▇▇▇                ▇▇▇▇▇▇▇▇▇▇▇                       │
│                                                                  │
│   PER-CATEGORY TABLE                                              │
│   category         imgs  lines  det_F1   CER    WER   ms/img      │
│   IDENTITY CARDS     5    142    0.83   0.062  0.22    820        │
│   NEWSPAPERS         5    198    0.68   0.114  0.41   1420        │
│   CONTRACTS          5    174    0.79   0.041  0.18    980        │
│   ...                                                            │
│                                                                  │
│   ─────  CLICK A CATEGORY OR ROW FOR DRILL-DOWN  ─────           │
└──────────────────────────────────────────────────────────────────┘
```

### Per-image drill-down

```
┌──────────────────────────────────────────────────────────────────┐
│  ‹ back    IDENTITY CARDS / INDONESIAN_CARD_ZZ_…01.jpg           │
│                                                                  │
│   ┌─────────────────────────────┐   GT 32 lines · PRD 28 lines    │
│   │                             │   Matched 27 · FP 1 · FN 5     │
│   │   [image with overlaid      │   Matched CER 0.054            │
│   │    GT=green, PR=blue,       │   Mean conf 0.91               │
│   │    unmatched=dim]           │   Latency 740 ms                │
│   │                             │                                 │
│   └─────────────────────────────┘   SAMPLE COMPARISON             │
│                                     GT:   PROVINSI JAWA BARAT     │
│                                     PRD:  PROVINSI JAWA BARAT  ✓  │
│                                     GT:   NIK                     │
│                                     PRD:  NIK                  ✓  │
│                                     GT:   axenyan0223@gmail.com   │
│                                     PRD:  axenyan0223@gmail.com ✓ │
└──────────────────────────────────────────────────────────────────┘
```

### Design tokens

* **Palette**: deep ink `#0b0d12`, glass cards `rgba(255,255,255,0.04)`, 1px border `rgba(255,255,255,0.08)`, accent `#7c5cff`.
* **Type**: system stack; numbers in `font-variant-numeric: tabular-nums`.
* **Motion**: 120ms ease-out on hover; no entrance animation.
* **Density**: 8px grid; cards `padding: 24px`; type `12 / 14 / 16 / 24 / 40`.
* **Charts**: hand-rolled SVG. No chart library.
* **Empty state**: one muted line. No spinner, no skeleton.

---

## 9 · API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/run?categories=*` | Run benchmark over given categories (or all); writes reports |
| `GET`  | `/api/summary` | Per-category + overall metrics JSON |
| `GET`  | `/api/results/<category>` | Per-image detail for one category |
| `GET`  | `/api/image/<category>/<filename>` | Serves original image bytes |
| `GET`  | `/` | The dashboard HTML |

Synchronous POST with a status line — 55 images finishes in ~1-2 minutes.

---

## 10 · Phased tasks

| # | Task | Files | Done when |
|---|---|---|---|
| 1 | Scaffold `pyproject.toml`, `src/ocr_bench/`, scripts | `pyproject.toml`, `__init__.py` | `uv sync` succeeds |
| 2 | Engine wrapper (reuse rapidocr, preserve polygons) | `engine.py` | one-liner predict on a sample returns lines |
| 3 | labelme JSON parser + category iterator | `dataset.py` | yields pages for one category |
| 4 | Polygon → AABB + IoU + greedy matcher | `matcher.py` | unit smoke test on 3 GT vs 4 PR |
| 5 | CER/WER + detection P/R/F1 | `metrics.py` | smoke test returns values in `[0,1]` |
| 6 | Runner — CLI per category, writes JSON+CSV | `runner.py`, `scripts/run_benchmark.py` | `reports/summary.csv` exists |
| 7 | FastAPI + JSON endpoints | `api.py` | curl `/api/summary` returns data |
| 8 | Dashboard HTML/CSS/JS | `ui/*` | page loads, table renders, click-through works |
| 9 | Overlay rendering (SVG boxes on image) | `api.py` overlay helper, `ui/app.js` | drill-down shows colored boxes |
| 10 | README run instructions | `README.md` | copy-paste commands work end-to-end |
| 11 | Smoke test | `tests/test_smoke.py` | one assertion that summary is sane |

---

## 11 · Dependencies (lazy, standalone)

```toml
dependencies = [
  "rapidocr-onnxruntime==1.4.4", # LOCKED — same as ai4db so model pack is byte-identical (see §4)
  "opencv-python>=4.9",           # image read + overlay drawing
  "numpy>=1.26",
  "pillow>=10",
  "fastapi>=0.115",
  "uvicorn[standard]>=0.30",
  "jiwer>=3.0",                   # CER/WER computation
]
```

That's it. **No** `ai4db` (intentional — see §4). **No** pandas (CSV write is
6 lines), **no** matplotlib (SVG hand-rolled), **no** pytest fixtures, **no**
Tailwind, **no** React.

### Deploy on a new device — checklist

```bash
git clone <this repo>
cd ocr-benchmark
uv sync
uv run python scripts/run_benchmark.py     # generates reports/
uv run ocr-bench-serve                     # opens dashboard
```

No ai4db checkout, no Python 3.13 required (whatever `rapidocr-onnxruntime`
supports), no external model download (model pack lives in the wheel).

### Version drift — what to do when ai4db updates `rapidocr-onnxruntime`

1. `cd …/ai4db && uv pip show rapidocr-onnxruntime` → note new version
2. `cd …/ocr-benchmark && edit pyproject.toml` → bump pin to new version
3. `uv lock && uv sync` → new model pack downloaded
4. `uv run python scripts/run_benchmark.py` → re-baseline numbers
5. Commit the new pin + refreshed `reports/summary.csv`

---

## 12 · Risks & what we deliberately skip

* **55 images is small** — results are directional, not statistically deep.
  We surface this in the UI ("directional; n=55"). When the team adds more
  labeled Indonesian docs, the benchmark scales automatically.
* **No train/test split** — we use category as the unit of analysis. If the
  team wants stratified k-fold later, add it then.
* **Polygon → AABB IoU** — fine for line text where boxes are rectangles.
  Rotated-IoU upgrade path noted if slanted text appears.
* **Single engine** — multi-engine comparison is the next project, not this
  one. Adding it now would mean a plugin system we don't need.
* **No persistence** — JSON files in `reports/`, git-trackable. Postgres is
  premature.
* **No auth** — local-only. Add when exposed.
* **Engine drift from ai4db** — we don't import ai4db. The `rapidocr-onnxruntime`
  pin is locked to ai4db's resolved version (currently `==1.4.4`). If ai4db
  bumps its pin, sync here in the same commit and re-baseline the benchmark
  numbers. The version-drift checklist is in §11.

---

## 13 · Definition of done

* `uv run python scripts/run_benchmark.py` produces `reports/summary.csv` and
  `reports/per_category/*.json` in under 2 minutes on this machine.
* `uv run ocr-bench-serve` opens a dashboard at `http://127.0.0.1:8765` with
  overall CER/WER/F1, per-category table, click-through to per-image view
  with overlaid boxes and side-by-side text comparison.
* `uv run pytest tests/` exits 0.
* README has copy-paste run instructions.

Then ship it.