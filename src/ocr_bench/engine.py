"""Standalone OCR wrapper.

Mirrors ``ai4db.ocr.pipeline.OCRPipeline`` (which uses
``rapidocr.RapidOCR``) but preserves bounding boxes + scores so
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
from rapidocr import RapidOCR
from rapidocr.utils.typings import ModelType, OCRVersion


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


def _preprocess(img: np.ndarray, min_side: int) -> tuple[np.ndarray, float]:
    """Grayscale + CLAHE + 2x upscale for short side. Returns (image, scale).

    ``scale`` is the geometric multiplier applied to the input. OCR polygons
    returned in this image's coordinate space must be divided by ``scale`` to
    map back to the original image space (used by the IoU matcher).

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
    else:
        scale = 1.0
    return enhanced, scale


class BenchEngine:
    """Same backend ai4db uses. Preserves polygons for IoU + CER/WER scoring."""

    def __init__(self, *, enable_preprocessing: bool = False, preproc_upscale_min_side: int = 800,
                 ocr_version: str = "PP-OCRv6", model_type: str = "small") -> None:
        ver = OCRVersion(ocr_version)
        mtype = ModelType(model_type)
        params = {
            "Det.ocr_version": ver,
            "Det.model_type": mtype,
            "Rec.ocr_version": ver,
            "Rec.model_type": mtype,
        }
        self._ocr = RapidOCR(params=params)
        self._enable_preprocessing = enable_preprocessing
        self._preproc_upscale_min_side = preproc_upscale_min_side

    def predict(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        scale = 1.0
        if self._enable_preprocessing:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                img, scale = _preprocess(img, self._preproc_upscale_min_side)
                result = self._ocr(img)
            else:
                result = self._ocr(str(image_path))
        else:
            result = self._ocr(str(image_path))

        # Remap polygons from preprocessed image space back to original image
        # space so the IoU matcher compares them on equal footing with GT.
        def _unscale(poly: list[list[float]]) -> list[list[float]]:
            if scale == 1.0:
                return poly
            return [[p[0] / scale, p[1] / scale] for p in poly]

        # rapidocr v3 returns RapidOCROutput (boxes/txts/scores attrs) or None.
        if result is None:
            boxes, txts, scores = [], [], []
        else:
            boxes, txts, scores = result.boxes, result.txts, result.scores

        lines = [
            LinePrediction(
                polygon=_unscale([[float(c) for c in pt] for pt in box.tolist()]),
                text=txt,
                score=float(sc),
            )
            for box, txt, sc in zip(boxes, txts, scores)
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