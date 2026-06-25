"""CLI: ``uv run python scripts/run_benchmark.py`` — runs the full benchmark."""
from __future__ import annotations

import argparse
import sys

from ocr_bench.runner import run
from ocr_bench.paths import DEFAULT_DATASET_ROOT


def main() -> int:
    p = argparse.ArgumentParser(description="Run OCR benchmark on IMG_OCR_IND_CN dataset.")
    p.add_argument("--dataset", default=None, help="Dataset root (default: ./IMG_OCR_IND_CN).")
    p.add_argument(
        "--category", action="append", default=None,
        help="Restrict to category name(s); repeatable. Default: all.",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    run(
        root=__import__("pathlib").Path(args.dataset) if args.dataset else None,
        only_categories=args.category,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())