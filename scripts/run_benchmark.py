"""CLI: ``uv run python scripts/run_benchmark.py`` — runs the full benchmark.

``--dataset`` accepts either a registry key (``ind_cn`` | ``new``) or a literal
filesystem path (legacy back-compat: if the value isn't a known key, it's
treated as a ``Path``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ocr_bench.runner import run
from ocr_bench.paths import DATASETS, DEFAULT_DATASET_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description="Run OCR benchmark.")
    p.add_argument(
        "--dataset", default=None,
        help=f"Dataset key ({'|'.join(DATASETS)}) or a literal root path. "
             f"Default: env OCR_BENCH_DATASET or '{DEFAULT_DATASET_ROOT.name}'.",
    )
    p.add_argument(
        "--category", action="append", default=None,
        help="Restrict to category name(s); repeatable. Default: all.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    # Registry-key wins over legacy path: if the value is a known key we let
    # the runner resolve via resolve_dataset_root(); otherwise we treat it as
    # a literal Path (preserves the old CLI behaviour).
    root: Path | None = None
    dataset_key: str | None = None
    if args.dataset:
        if args.dataset in DATASETS:
            dataset_key = args.dataset
        else:
            root = Path(args.dataset)

    run(
        root=root,
        only_categories=args.category,
        verbose=not args.quiet,
        dataset_key=dataset_key,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())