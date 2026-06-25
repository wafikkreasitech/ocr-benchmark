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

import cv2
import numpy as np
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


def _preprocess(img: np.ndarray, min_side: int) -> np.ndarray:
    """Grayscale + CLAHE + 2x upscale for short side. Helps faint/low-contrast text.

    ponytail: keep it simple — one fixed pipeline behind one toggle. Add per-op
    knobs only when measurements show the fixed pipeline hurts some category.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # CLAHE on grayscale; tile size 8x8 is the OpenCV-recommended default.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    h, w = enhanced.shape[:2]
    if min(h, w) < min_side:
        scale = (min_side / min(h, w)) * 2  # 2x beyond the floor
        enhanced = cv2.resize(
            enhanced, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )
    return enhanced


class BenchEngine:
    """Same backend ai4db uses. Preserves polygons for IoU + CER/WER scoring."""

    def __init__(self, *, enable_preprocessing: bool = False, preproc_upscale_min_side: int = 800) -> None:
        # ponytail: no-args = default config shipped in the wheel (PP-OCRv4).
        # Matches ai4db's ``RapidOCR()`` initialization exactly.
        self._ocr = RapidOCR()
        self._enable_preprocessing = enable_preprocessing
        self._preproc_upscale_min_side = preproc_upscale_min_side

    def predict(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        if self._enable_preprocessing:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                img = _preprocess(img, self._preproc_upscale_min_side)
                result, _ = self._ocr(img)
            else:
                result, _ = self._ocr(str(image_path))
        else:
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