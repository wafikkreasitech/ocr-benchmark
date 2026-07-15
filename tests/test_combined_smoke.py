"""Smoke test — combined OCR+TTS end-to-end runner sanity.

Skipped when the dataset or the Piper voice isn't present, mirroring
test_smoke.py / test_tts_smoke.py. Runs on a tiny 1-category, 2-image subset
so it stays fast; asserts the per-page timing math is internally consistent
(total_ms == ocr_ms + overhead_ms + tts_ms) and metrics are sane.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from ocr_bench.config import get_settings

_VOICE = Path(get_settings().piper_voice_path)
_DATASET = Path("IMG_OCR_IND_CN")
_needs_all = pytest.mark.skipif(
    not _VOICE.exists() or not _DATASET.exists(),
    reason="needs both the Piper voice and IMG_OCR_IND_CN dataset present",
)


@_needs_all
def test_combined_mini_run():
    from ocr_bench.combined_runner import COMBINED_STATUS_PATH, COMBINED_SUMMARY_PATH, run as run_combined

    with tempfile.TemporaryDirectory() as tmp:
        backup = Path(tmp) / "reports_backup"
        live_reports = Path("reports")
        if live_reports.exists():
            shutil.copytree(live_reports, backup)

        try:
            summary = run_combined(only_categories=["IDENTITY CARDS"], source="pred")
        finally:
            if backup.exists():
                shutil.rmtree(live_reports, ignore_errors=True)
                shutil.copytree(backup, live_reports)

    assert summary, "combined run returned empty summary"
    overall = summary["overall"]
    assert overall["pages"] > 0, "expected at least one page synthesized"
    assert overall["ocr_ms_mean"] > 0
    assert overall["tts_ms_mean"] > 0
    assert overall["total_ms_mean"] > 0
    # Internal consistency: total should equal ocr + overhead + tts (within
    # floating-point noise) at the per-category level.
    for cat in summary["per_category"]:
        if cat["pages"] == 0:
            continue
        expected_total = cat["ocr_ms_mean"] + cat["overhead_ms_mean"] + cat["tts_ms_mean"]
        assert abs(cat["total_ms_mean"] - expected_total) < 1.0, (
            f"{cat['category']}: total_ms_mean {cat['total_ms_mean']} != "
            f"ocr+overhead+tts {expected_total}"
        )
    assert overall["ocr_backend"] in ("cpu", "cuda", "tensorrt")
    assert overall["tts_backend"] in ("cpu", "cuda")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
