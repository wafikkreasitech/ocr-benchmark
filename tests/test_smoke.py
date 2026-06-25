"""Smoke test — sanity-check the benchmark pipeline.

Runs the full benchmark once on a tiny subset (2 categories × 2 images each)
and asserts that the metrics are sane. Skipped if `reports/` already has a
fresh run — we don't want to re-run OCR on every test invocation unless asked.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from ocr_bench.matcher import iou_polygon, match
from ocr_bench.metrics import DetectionStats, aggregate_category, cer, wer
from ocr_bench.runner import run as run_benchmark


# ----- pure-function unit checks -----

def test_iou_polygon_perfect_overlap():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert iou_polygon(a, b) == 1.0


def test_iou_polygon_no_overlap():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    b = [[100, 100], [110, 100], [110, 110], [100, 110]]
    assert iou_polygon(a, b) == 0.0


def test_iou_polygon_partial():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]     # area 100
    b = [[5, 5], [15, 5], [15, 15], [5, 15]]     # overlap 5x5=25, union=175
    expected = 25 / 175
    assert abs(iou_polygon(a, b) - expected) < 1e-9


def test_match_simple_pair():
    g = type("G", (), {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]})()
    p = type("P", (), {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]})()
    matches, ug, up = match([g], [p])
    assert len(matches) == 1 and not ug and not up


def test_cer_normalization():
    assert cer("HELLO World", "hello  world") == 0.0
    assert cer("", "") == 0.0
    assert 0 < cer("abc", "axc") < 1


def test_wer_basic():
    assert wer("a b c", "a b d") > 0
    assert wer("a b c", "a b c") == 0.0


def test_detection_stats_f1():
    d = DetectionStats(tp=8, fp=2, fn=2)
    assert abs(d.precision - 0.8) < 1e-9
    assert abs(d.recall - 0.8) < 1e-9
    assert abs(d.f1 - 0.8) < 1e-9


# ----- end-to-end mini-run on real data -----

@pytest.mark.skipif(
    not Path("IMG_OCR_IND_CN").exists(),
    reason="IMG_OCR_IND_CN dataset not present",
)
def test_mini_benchmark_runs():
    """Run on 2 categories × 2 images each, assert metrics are sane."""
    with tempfile.TemporaryDirectory() as tmp:
        backup = Path(tmp) / "reports_backup"
        live_reports = Path("reports")
        if live_reports.exists():
            shutil.copytree(live_reports, backup)

        try:
            overall = run_benchmark(
                root=None,
                only_categories=["IDENTITY CARDS", "NEWSPAPERS"],
                verbose=False,
            )
        finally:
            # restore reports (the test ran the real runner which writes there)
            if backup.exists():
                shutil.rmtree(live_reports, ignore_errors=True)
                shutil.copytree(backup, live_reports)

        # Sanity checks on overall
        assert overall["n_images"] == 10, f"expected 10 images, got {overall['n_images']}"
        assert overall["n_lines"] > 0, "expected GT lines"
        assert 0.0 <= overall["detection_f1"] <= 1.0, "F1 out of range"
        assert 0.0 <= overall["cer_mean"] <= 5.0, "CER wildly out of range"
        assert 0.0 <= overall["wer_mean"] <= 5.0, "WER wildly out of range"


# ----- aggregate smoke -----

def test_aggregate_category_empty():
    s = aggregate_category("EMPTY", [])
    assert s.n_images == 0
    assert s.detection.f1 == 0.0
    assert s.matched_cer_mean == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))