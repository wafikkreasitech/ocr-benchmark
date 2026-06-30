"""Shared paths for the benchmark package.

Multi-dataset registry: every supported dataset lives behind a stable key
(``ind_cn``, ``new``). The active key is resolved from ``OCR_BENCH_DATASET``:
- if the env var matches a known key  -> use that registry entry
- if it's a non-empty path            -> treat as a one-off ad-hoc root (legacy back-compat)
- otherwise                            -> fall back to the default key (``ind_cn``)

Reports live under ``reports/`` (unchanged).
"""
from __future__ import annotations

import os
from pathlib import Path

# src/ocr_bench/paths.py → src/ocr_bench/ → src/ → package root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

# Each entry: registry key -> dataset root on disk.
# Add new datasets here — every other layer (api, runner, UI) reads from this dict.
DATASETS: dict[str, Path] = {
    "ind_cn": PACKAGE_ROOT / "IMG_OCR_IND_CN",
    "new":    PACKAGE_ROOT / "dataset" / "dataset",
}

DEFAULT_DATASET_KEY = "ind_cn"

REPORTS_ROOT = PACKAGE_ROOT / "reports"
HISTORY_ROOT = REPORTS_ROOT / "history"
UI_ROOT = PACKAGE_ROOT / "ui"

# Lazy: read env var so tests / CI can override at import time.
_env = os.environ.get("OCR_BENCH_DATASET", "").strip()

def resolve_dataset_root(key_or_path: str | None = None) -> tuple[str, Path]:
    """Return ``(key, root_path)`` for the requested dataset.

    Resolution order:
      1. ``key_or_path`` argument if provided.
      2. ``OCR_BENCH_DATASET`` env var.
      3. Default key (``ind_cn``).

    If the resolved value is a known registry key  -> its root.
    If it's a non-empty string that isn't a key     -> legacy path override.
    Empty / unknown / missing                       -> default key.
    """
    raw = (key_or_path or _env or DEFAULT_DATASET_KEY).strip()
    if raw in DATASETS:
        return raw, DATASETS[raw]
    if raw and raw != DEFAULT_DATASET_KEY:
        # Legacy / ad-hoc path override.
        return raw, Path(raw)
    return DEFAULT_DATASET_KEY, DATASETS[DEFAULT_DATASET_KEY]

DEFAULT_DATASET_ROOT = resolve_dataset_root()[1]