# Plan · Add New Dataset Alongside Existing One

> **Goal**: support the new dataset at `dataset/dataset/{testing,training}_data`
> **alongside** the existing `IMG_OCR_IND_CN/`, **fully configurable from the
> frontend** (pick a dataset, run, view results — no CLI/curl needed), while
> the legacy flow stays intact. Both annotation formats must parse correctly
> and both layouts must be navigable from the dashboard.

---

## 1 · What we have today

| Path | Role | Format | Annotations |
|---|---|---|---|
| `IMG_OCR_IND_CN/<category>/*.jpg\|*.png` | **Old / legacy** dataset (default) | labelme v4 — `shapes[].label` + `shapes[].points` (rect/poly) | 1 per image, sidecar JSON |
| `dataset/dataset/testing_data/{images,annotations}/` | **New** test split | FUNSD-style — top-level `form[]`, each entry has `box [x1,y1,x2,y2]`, `text`, `label` ∈ {`other`,`question`,`answer`,`header`}, nested `words[]` | 1 per image, **matching filename** in `annotations/` |
| `dataset/dataset/training_data/{images,annotations}/` | **New** train split (use for benchmark too — full set, not subset) | same as above | same as above |

Sizes: **old = 11 categories, 55 images, 1,674 GT lines**. **new testing = 50 imgs, new training = 149 imgs, total 199 imgs, ~5k GT entries**.

The new dataset is currently **completely invisible** to the app — only `IMG_OCR_IND_CN` is reachable via `DEFAULT_DATASET_ROOT` and the parser in `dataset.py` only handles labelme v4.

The new dataset also has **no category subdirectories** — its categories are implicit (`testing_data` vs `training_data`) or absent (everything is flat). The whole "category" concept the runner / UI uses needs an adapter.

---

## 2 · Design — keep it boring

### 2.1 · Multi-dataset registry

Treat each dataset as a **named root** with its own parser:

```
DATASETS = {
    "ind_cn": {
        "label": "IMG_OCR_IND_CN (labelme)",
        "root":  PACKAGE_ROOT / "IMG_OCR_IND_CN",
        "loader": load_labelme_category,         # existing parser
    },
    "new": {
        "label": "FUNSD-form (testing+training)",
        "root":  PACKAGE_ROOT / "dataset" / "dataset",
        "loader": load_funsd_category,           # new parser
    },
}
```

Add **one env knob** `OCR_BENCH_DATASET=ind_cn|new` to choose the active one (default `ind_cn` so nothing changes for existing users). The runner / UI / API / CLI all accept a `dataset` parameter that overrides this.

### 2.2 · Category mapping for the new dataset

The new dataset has no per-category subdirectories. We **synthesize** two categories from the split dir names so the existing UI / report layout keeps working:

| Dataset root dir | Synthetic category name |
|---|---|
| `testing_data/images` | `TESTING DATA` |
| `training_data/images` | `TRAINING DATA` |

Each synthesized category is a flat bag of `{*.png}` matched to `annotations/{basename}.json` (same-stem join, identical to the old dataset's convention). That keeps `list_categories()` / `load_category()` / `iter_all_images()` working unchanged from the caller's perspective.

### 2.3 · New annotation parser

Add `load_funsd_category(category_dir)` to `dataset.py`:

- Glob `*.png` in the dir (new dataset is PNG-only).
- For each image, look up `annotations/<stem>.json` (sibling dir, mapped via the synthetic category index).
- Parse `form[]`:
  - Skip entries where `text.strip() == ""` (most are blank in the new format — confirmed by sampling).
  - Convert `box` `[x1,y1,x2,y2]` → polygon `[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]` so the existing matcher doesn't care about coordinate shape.
  - Use `entry.text` as the transcript; `entry.label` is preserved as metadata (informational only — not used for metrics in v1).
- Return the same `GroundTruthPage` / `GroundTruthLine` shape as today.

### 2.4 · Loader dispatch

Generalize `list_categories(root)` and `load_category(category_dir)` to **detect format per root**:

- If `category_dir` contains `images/` + `annotations/` siblings → **FUNSD-style**.
- Else → existing labelme-style.

Detection lives in `dataset.py` so the runner / API don't change call sites. Each root self-describes its parser; a tiny `_pick_loader(dir)` helper routes.

### 2.5 · Dataset-root resolution

Update `paths.py`:

```python
DATASETS = {
    "ind_cn": PACKAGE_ROOT / "IMG_OCR_IND_CN",
    "new":    PACKAGE_ROOT / "dataset" / "dataset",
}
DEFAULT_DATASET_KEY = os.environ.get("OCR_BENCH_DATASET", "ind_cn")
DEFAULT_DATASET_ROOT = DATASETS[DEFAULT_DATASET_KEY]
```

`api_image()` and `api_run()` look up the root from the active dataset key. We also accept `dataset=new` as a query/body param on `/api/run` (one-off, not persisted to `.env`).

### 2.6 · CLI & API surface

| Surface | Old | New |
|---|---|---|
| `scripts/run_benchmark.py` | `--dataset <path>` (single root) | `--dataset ind_cn\|new` (registry key) — **back-compat**: existing path arg still works (treated as a one-off ad-hoc root). |
| `POST /api/run` | `category` param only | adds optional `dataset` param (`ind_cn` \| `new`). Default = current `OCR_BENCH_DATASET`. |
| `GET /api/config` | exposes `ocr_version`, etc. | adds `dataset` (active key) and `datasets` (available keys + labels) so the UI can populate a selector. |
| `GET /api/summary` | reports from the active dataset only | unchanged. Summary files now stamp `dataset` key in the JSON so history can be filtered later. |

### 2.7 · UI — fully configurable from the frontend

The frontend must let a user (a) **pick a dataset**, (b) **run the benchmark on it**, and (c) **view the results** — without ever touching a terminal. Three coordinated surfaces:

#### 2.7.1 · Run-panel dataset picker (primary control)

Top of the existing dashboard, right next to the model card selector:

```
┌── Run panel ──────────────────────────────────────────────┐
│  Dataset:  [ IMG_OCR_IND_CN (labelme)   ▼ ]   ← dropdown  │
│  Category: [ All ▼ ] / [ TESTING DATA ▼ ]  (dynamic)     │
│  Model:    [ PP-OCRv6 · tiny ▼ ]                          │
│  [ ▶ Run benchmark ]                                      │
└───────────────────────────────────────────────────────────┘
```

- The **Dataset** dropdown is populated from `GET /api/config.datasets` (returns the registry: `[ {key, label, n_images, n_lines, format} ]`).
- When the dataset changes, the **Category** dropdown repopulates from the chosen dataset's `list_categories()` (new endpoint `GET /api/datasets/<key>/categories` — see §2.7.4).
- "All" stays the default — same behaviour as today, just scoped to the active dataset.
- Existing model card, knob sliders, and Run button keep working; only the new dropdowns are added above them.

#### 2.7.2 · New "Datasets" page (`ui/datasets.html`)

A full-page overview of every registered dataset, reached from a top-nav link. Cards, one per dataset:

```
┌── IMG_OCR_IND_CN (labelme) ──────────  [ Active ✓ ] ──┐
│  11 categories · 55 images · 1,674 GT lines            │
│  [ ▶ Run ]  [ 📊 View last results ]                    │
└────────────────────────────────────────────────────────┘

┌── FUNSD-form (testing + training) ─────────────────────┐
│  2 categories · 199 images · ~5k GT entries            │
│  [ ▶ Run ]  [ 📊 View last results ]                    │
└────────────────────────────────────────────────────────┘
```

- **Active** badge shows which dataset is currently selected (from `GET /api/config.dataset`).
- **Run** triggers `POST /api/run?dataset=<key>` — same code path as the Run panel, so SSE progress + cancellation + locks all work.
- **View last results** jumps to `/api/results/<first-category-of-dataset>` so the existing per-image overlay page works without any per-dataset UI duplication.
- While a run is in progress, the card shows a live spinner using the existing `/api/progress/stream` payload (filtered by `dataset`).

#### 2.7.3 · Top-nav addition

One new top-nav link on every page (`index.html` + `datasets.html`):

```
Datasets   |   History   |   …
```

`<a href="datasets.html">Datasets</a>` + matching active-state class on the datasets page. No client router — plain links, so the page works even before `app.js` finishes loading.

#### 2.7.4 · New API endpoints the frontend needs

| Endpoint | Purpose | Returns |
|---|---|---|
| `GET /api/config` | extended (already exists) | adds `dataset` (active key) + `datasets` (registry list). |
| `GET /api/datasets` | convenience — list all datasets with stats | `[ {key, label, format, n_categories, n_images, n_lines, active} ]` |
| `GET /api/datasets/<key>/categories` | populate the Category dropdown when a dataset is selected | `[ {name, n_images, n_lines}, … ]` (404 if key unknown) |
| `POST /api/run?dataset=<key>` | run on chosen dataset | same response shape as today; lock file stamped with `dataset` so /api/progress reflects it |

All other endpoints (`/api/summary`, `/api/results/<cat>`, `/api/image/<cat>/<file>`, `/api/history*`) stay unchanged — they continue to operate on the most recent run, and the summary JSON is now stamped with the active `dataset` so the dashboard banner can show "Results for: FUNSD-form · TESTING DATA".

#### 2.7.5 · Frontend state model

Extend the existing `selectedModel` pattern in `app.js`:

```js
const selectedDataset = { key: "ind_cn", category: null };  // category = null → all
```

- `selectedDataset.key` updates from either the Run-panel dropdown **or** the Datasets-page card click — both stay in sync via a tiny `datasetChanged(key)` event.
- `selectedDataset.category` resets to `null` (All) whenever `key` changes.
- The Run button posts `selectedDataset.key` + `selectedDataset.category` (when not null) + the existing model + knobs to `/api/run`.
- The active dataset badge in the top-right shows the current selection at all times.

Existing UI code is untouched except for: (a) wiring the new dropdowns, (b) the dataset-change event, (c) the `/api/run` URL builder includes `dataset=`. Everything else — progress stream, summary table, per-category drill-down, history, model cards, knob sliders — keeps working as-is.

### 2.8 · History tagging

`_save_to_history()` stamps `dataset: <key>` on every snapshot. `/api/history` returns a `datasets` facet so users can filter the history table by dataset in a future iteration (out of scope here — the data is captured, the filter UI is not). **ponytail**: don't ship the filter UI until someone asks.

### 2.9 · Reports — keep outputs separate

`summary.json` / `summary.csv` / `per_category/*.json` keep overwriting on each run (same as today). They now reflect the dataset the run was made on. Each per-category JSON gets a `dataset` field stamped at write time so when both datasets exist on disk we can tell which run wrote what. No new file layout needed.

---

## 3 · Files to touch

| File | Change |
|---|---|
| `src/ocr_bench/paths.py` | Add `DATASETS` registry + key env var. |
| `src/ocr_bench/dataset.py` | Add `load_funsd_category` + `_pick_loader` dispatch. |
| `src/ocr_bench/runner.py` | Accept `dataset_key` in `run()`; pass through to loader. Stamp `dataset` in history & per-cat report. |
| `src/ocr_bench/api.py` | `POST /api/run?dataset=…`; `GET /api/config` exposes `dataset` & `datasets`; **NEW** `GET /api/datasets` + `GET /api/datasets/<key>/categories`; `api_image` routes to the correct root; `summary.json` not modified. |
| `scripts/run_benchmark.py` | Accept `--dataset ind_cn\|new` (back-compat path arg preserved). |
| `ui/index.html` | One new dataset picker `<select>` in the Run panel + Category dropdown + top-nav link to `datasets.html`. |
| `ui/datasets.html` | **NEW** — full-page list of datasets + per-row "Run" / "View" actions + active badge. |
| `ui/app.js` | `selectedDataset` state, dataset-change event, populate both dropdowns (≤ 60 LOC); `datasets.html` page init (≤ 80 LOC). |
| `ui/style.css` | Reuse existing card styles; one new rule for the dataset grid + active badge. |
| `.env.example` | Document `OCR_BENCH_DATASET`. |
| `README.md` | One paragraph in Quick Start + a "Datasets" section. |
| `docs/plan-new-dataset.md` | This file. |

**Total new code**: ~150 LOC across dataset parser, API wiring, UI page. **No deletions**, only additions.

---

## 4 · Risks & how we defuse them

| Risk | Mitigation |
|---|---|
| Old flow breaks (some user still has `.env` pointing at the old path) | Default key is `ind_cn`. Existing `OCR_BENCH_DATASET=<path>` continues to work — paths override keys. |
| New dataset has empty-text entries that inflate line counts | Parser skips `text.strip() == ""` (verified: most form entries are blank metadata). |
| New dataset is PNG-only | Loader globs `.png`. Existing labelme path already supports both jpg/png, no change needed. |
| Run report from dataset A is overwritten by run on dataset B | Acceptable in v1 — `summary.json` / per-cat JSON stamped with `dataset`. A v2 multi-dataset switch in the UI can opt to keep separate outputs. |
| UI 404 on the new `datasets.html` if user opens it before the JS bundle ships the nav link | Page loads via plain `<a href>`, no client router. Single static file works the moment it's deployed. |
| Filename collisions between the two datasets' per-category JSONs (`BADGES AND PASSES.json` vs `TESTING DATA.json`) | Slug-safe: distinct names today. `dataset.py` adds the `dataset` key to the report to disambiguate later. |
| `__MACOSX/` cruft inside `dataset/` | Already excluded by the existing `.gitignore` style rule (anything starting with `_`). |
| `--dataset` CLI ambiguity (path vs key) | Try registry-key lookup first; fall back to treating arg as a literal Path. Same back-compat behaviour as today. |

---

## 5 · What we explicitly skip (YAGNI)

- Multi-dataset simultaneous run / fused report (single-active-dataset is enough for v1).
- A `dataset` filter in the History table UI — data is captured, UI deferred.
- A live "preview GT polygons on image" view of the new dataset (the existing overlay machinery works once `overlays` carry the FUNSD polygon, which they do automatically since we convert `box` → polygon).
- Streaming image bytes for `.zip`-bundled datasets (not relevant — both datasets are extracted on disk).
- Renaming `IMG_OCR_IND_CN/` or moving anything in `dataset/dataset/` — keep the directory layout the user already has.

---

## 6 · Acceptance

A reviewer running the following commands sees both datasets work, with no regression on the old one:

```bash
# Legacy — must work exactly as before
uv run python scripts/run_benchmark.py --dataset ind_cn
uv run ocr-bench-serve   # → http://127.0.0.1:8765  (defaults to IMG_OCR_IND_CN)

# New dataset
uv run python scripts/run_benchmark.py --dataset new
# or via UI: pick "FUNSD-form" in the Run panel → click Run
```

And from the UI (the **frontend-configurable** path — must work end-to-end):

1. Open `/` → see old categories (BADGES, BILLS, …) by default. Top-right "Active dataset" badge shows `IMG_OCR_IND_CN`.
2. Click **Datasets** in the top nav → `/datasets.html` lists both datasets with image/line counts and "Run" / "View last results" buttons.
3. On `/datasets.html`, click **Run** on the `FUNSD-form` card → page shows progress bar; SSE stream shows `TESTING DATA` then `TRAINING DATA` categories ticking through.
4. Run finishes → auto-link to `/api/results/TESTING DATA` shows the first FUNSD image with GT/pred overlays.
5. Back on `/`, pick **FUNSD-form** from the new Run-panel **Dataset** dropdown → Category dropdown repopulates with `All / TESTING DATA / TRAINING DATA` → pick `TESTING DATA` → click Run → progress + results land in the existing dashboard without any page-specific code.
6. Pick `IMG_OCR_IND_CN` from the dropdown → all 11 categories reappear → click Run → legacy behaviour unchanged.
7. "Active dataset" badge in the top-right tracks whichever dataset the user most recently selected — never lies.

```bash
# Static check
uv run python -m ocr_bench.dataset   # ponytail self-check: both parsers return ≥1 page
```

---

## 7 · Open questions for the user (before implementation)

1. Should the new dataset's **training_data split** be benchmarked too, or treated as out-of-scope for the OCR benchmark (training_data is for model training, not evaluation)? My read: **include both splits** as separate synthetic categories so the user can compare, but emit a one-line warning in the runner when training_data is selected (e.g. "note: this is training data — metrics reflect fit, not generalization").
2. Dataset label in the UI — "FUNSD-form" or just "New dataset"? My default: **`FUNSD-form`** (descriptive of the format).
3. Top-nav position for the "Datasets" link: rightmost, or grouped with "History"? My default: **rightmost**, after History.

If no objections, plan is locked and I'll implement.