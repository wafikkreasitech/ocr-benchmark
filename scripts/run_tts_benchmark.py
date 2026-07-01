"""CLI: run the TTS benchmark over the OCR results.

    uv run python -m scripts.run_tts_benchmark            # speak OCR output (pred)
    uv run python -m scripts.run_tts_benchmark --source gt  # speak ground truth

Writes reports/tts_summary.json + .csv. Requires the OCR benchmark to have run
first (reports/per_category/*.json) and the Piper voice downloaded
(scripts/download_voice.py).
"""
from __future__ import annotations

import argparse
import logging
import sys

from ocr_bench.tts_runner import run


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS speed benchmark over OCR results")
    ap.add_argument("--source", choices=["pred", "gt"], default=None,
                    help="which text to speak: pred (OCR output, default) or gt (clean)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    summary = run(args.source)
    o = summary["overall"]
    live = "faster than real-time" if o["rtf_mean"] < 1.0 else "SLOWER than real-time"
    print(f"\nOK  {o['pages']} pages spoken  |  RTF={o['rtf_mean']:.3f} ({live})")
    print(f"    synth={o['synth_ms_mean']:.0f}ms/page  "
          f"first-chunk={o['first_chunk_ms_mean']:.0f}ms  "
          f"{o['chars_per_sec_mean']:.0f} chars/s  {o['failures']} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
