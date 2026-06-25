"""Standalone OCR wrapper.

Mirrors ``ai4db.ocr.pipeline.OCRPipeline`` (which uses
``rapidocr_onnxruntime.RapidOCR``) but preserves bounding boxes + scores so
the benchmark can compute IoU-matched CER/WER and detection F1.

If ai4db ever changes its OCR backend, mirror it here — typically a version
bump + a config tweak. See docs/plan.md §4.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from rapidocr_onnxruntime import RapidOCR


@dataclass
class LinePrediction:
    polygon: list[list[float]]  # 4-pt box from rapidocr: [[x,y], [x,y], [x,y], [x,y]]
    text: str
    score: float


@dataclass
class PagePrediction:
    image: str
    lines: list[LinePrediction]
    elapsed_ms: float


class BenchEngine:
    """Same backend ai4db uses. Preserves polygons for IoU + CER/WER scoring."""

    def __init__(self) -> None:
        # ponytail: no-args = default config shipped in the wheel (PP-OCRv4).
        # Matches ai4db's ``RapidOCR()`` initialization exactly.
        self._ocr = RapidOCR()

    def predict(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        result, _ = self._ocr(str(image_path))
        lines = [
            LinePrediction(polygon=poly, text=txt, score=sc)
            for (poly, txt, sc) in (result or [])
        ]
        return PagePrediction(
            image=image_path.name,
            lines=lines,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


if __name__ == "__main__":  # ponytail: self-check
    import sys
    from .dataset import iter_all_images

    target = sys.argv[1] if len(sys.argv) > 1 else None
    engine = BenchEngine()
    pages = iter_all_images(target)
    if not pages:
        print("no images found")
        raise SystemExit(1)
    p = engine.predict(pages[0].image_path)
    print(f"{p.image}: {len(p.lines)} lines, {p.elapsed_ms:.0f} ms")
    for ln in p.lines[:3]:
        print(f"  score={ln.score:.2f} text={ln.text!r}")