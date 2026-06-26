"""Shared paths for the benchmark package.

Dataset lives next to the package root. Reports live under ``reports/``.
"""
from __future__ import annotations

from pathlib import Path

# src/ocr_bench/paths.py → src/ocr_bench/ → src/ → package root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PACKAGE_ROOT / "IMG_OCR_IND_CN"
REPORTS_ROOT = PACKAGE_ROOT / "reports"
HISTORY_ROOT = REPORTS_ROOT / "history"
UI_ROOT = PACKAGE_ROOT / "ui"

# Lazy: read env var so tests / CI can override
import os

DEFAULT_DATASET_ROOT = Path(os.environ.get("OCR_BENCH_DATASET", str(DATASET_ROOT)))