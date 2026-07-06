"""Standalone OCR wrapper.

Mirrors ``ai4db.ocr.pipeline.OCRPipeline`` (which uses
``rapidocr.RapidOCR``) but preserves bounding boxes + scores so
the benchmark can compute IoU-matched CER/WER and detection F1.

If ai4db ever changes its OCR backend, mirror it here — typically a version
bump + a config tweak. See docs/plan.md §4.
"""
from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rapidocr import RapidOCR
from rapidocr.utils.typings import EngineType, ModelType, OCRVersion

log = logging.getLogger(__name__)


def _detect_cuda() -> bool:
    """Auto-detect CUDA availability for onnxruntime-gpu."""
    try:
        import onnxruntime as ort
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _detect_tensorrt() -> bool:
    """Auto-detect TensorRT + cuda-python bindings availability.

    rapidocr's native TensorRT backend needs both the ``tensorrt`` package
    and ``cuda-python`` (for the cudart bindings) — both are pre-installed
    on Jetson via JetPack, but not everywhere, so check before enabling.
    """
    try:
        import tensorrt  # noqa: F401
        from cuda.bindings import runtime  # noqa: F401
        return True
    except Exception:
        return False


def _find_rapidocr_models_dir() -> Path | None:
    """Locate the installed rapidocr package's bundled ``models/`` dir.

    That's where rapidocr downloads/caches its ONNX files — same directory
    the ONNXRuntime backend reads from, so the TensorRT dict extraction (see
    ``_rec_dict_path``) looks in the same place.
    """
    try:
        import rapidocr
        return Path(rapidocr.__file__).parent / "models"
    except Exception:
        return None


def _rec_dict_path(models_dir: Path, ocr_version: str, model_type: str) -> Path | None:
    """Extract the rec character dict embedded in the ONNX model's metadata.

    rapidocr's ONNXRuntime backend reads ``character`` straight from the ONNX
    model's custom_metadata_map. The TensorRT backend has no such metadata
    (engines don't carry it) and falls back to downloading a dict — which for
    PP-OCRv6 resolves to the wrong (PP-OCRv4) dict and breaks decoding
    (IndexError: token id doesn't exist in the smaller dict). Extract once
    and cache alongside the ONNX model.
    """
    cache_name = f"{ocr_version}_rec_{model_type}_dict.txt"
    cache_path = models_dir / cache_name
    if cache_path.exists():
        return cache_path
    onnx_name = f"{ocr_version}_rec_{model_type}.onnx"
    onnx_path = models_dir / onnx_name
    if not onnx_path.exists():
        return None
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        chars = meta.get("character")
        if not chars:
            return None
        cache_path.write_text(chars, encoding="utf-8")
        log.info("Extracted %d-char rec dict -> %s", len(chars.splitlines()), cache_path)
        return cache_path
    except Exception as e:  # noqa: BLE001 — dict extraction must never crash init
        log.warning("Failed to extract rec dict from %s: %s", onnx_path, e)
        return None


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
                 ocr_version: str = "PP-OCRv6", model_type: str = "tiny",
                 det_box_thresh: float = 0.5, det_unclip_ratio: float = 1.6,
                 det_thresh: float = 0.3,
                 det_limit_side_len: int = 1536,
                 use_angle_cls: bool = False,
                 rec_batch_num: int = 6, rec_img_width: int = 320,
                 use_tensorrt: bool = False, trt_cache_dir: str = "models/trt_engines") -> None:
        ver = OCRVersion(ocr_version)
        mtype = ModelType(model_type)
        params = {
            "Det.ocr_version": ver,
            "Det.model_type": mtype,
            "Det.box_thresh": det_box_thresh,
            "Det.thresh": det_thresh,
            "Det.unclip_ratio": det_unclip_ratio,
            "Det.limit_side_len": det_limit_side_len,
            "Rec.ocr_version": ver,
            "Rec.model_type": mtype,
            "Rec.rec_batch_num": rec_batch_num,
            "Rec.rec_img_shape": [3, 48, rec_img_width],
        }
        # ponytail: rapidocr v3 hard-runs the Cls (angle) stage — no on/off toggle
        # exists in its config schema. ``use_angle_cls`` is preserved as a setting
        # for parity with the env knob; if you ever need to disable it, swap
        # RapidOCR for a pipeline that accepts the flag.
        self._use_angle_cls = use_angle_cls

        self.backend = "cpu"
        if use_tensorrt and _detect_tensorrt():
            # rapidocr's native TensorRT backend: builds a fused FP16 .engine
            # on first run (cached to disk keyed by GPU arch, ~40x faster det
            # inference on Jetson vs CUDAExecutionProvider — see docs/plan.md).
            # First run per model/shape combo takes minutes to compile; every
            # run after that loads the cached .engine in <1s.
            from .paths import PACKAGE_ROOT
            cache_dir = Path(trt_cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = PACKAGE_ROOT / cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)

            params["Det.engine_type"] = EngineType.TENSORRT
            params["Rec.engine_type"] = EngineType.TENSORRT
            params["EngineConfig.tensorrt.use_fp16"] = True
            params["EngineConfig.tensorrt.cache_dir"] = str(cache_dir)
            # rapidocr's bundled TRT det profile caps at 2048x2048 (square).
            # Det preprocessing scales the *short* side up to det_limit_side_len
            # while preserving aspect ratio, so a portrait/landscape image can
            # end up with a *long* side well past 2048 (e.g. a 928x1470 ID
            # card at limit_side_len=1536 resizes to 1536x2432) — silently
            # failing TensorRT's shape check (throws inside rapidocr, caught,
            # and returned as an empty/near-empty result with no error
            # surfaced to the caller). Widen the profile's max_shape so it
            # covers realistic aspect ratios at every det_limit_side_len this
            # UI exposes (up to 2048 — see .env.example).
            params["EngineConfig.tensorrt.det_profile"] = {
                "min_shape": [1, 3, 32, 32],
                "opt_shape": [1, 3, det_limit_side_len, det_limit_side_len],
                "max_shape": [1, 3, 4096, 4096],
            }
            # The TensorRT backend has no ONNX metadata to read the rec
            # character dict from (unlike ONNXRuntime) and its own
            # dict-resolution falls back to the wrong (PP-OCRv4) dict for
            # PP-OCRv6 models — extract the correct one from the ONNX model
            # metadata once and point rec_keys_path at it.
            onnx_models_dir = _find_rapidocr_models_dir()
            if onnx_models_dir is not None:
                dict_path = _rec_dict_path(onnx_models_dir, ocr_version, model_type)
                if dict_path is not None:
                    params["Rec.rec_keys_path"] = str(dict_path)
            self.backend = "tensorrt"
            self._ocr = RapidOCR(params=params)
            log.info("TensorRT acceleration enabled (FP16, cache=%s)", cache_dir)
        else:
            if use_tensorrt:
                log.warning("use_tensorrt=True but tensorrt/cuda-python not importable; falling back to CUDA/CPU")
            use_cuda = _detect_cuda()
            params["EngineConfig.onnxruntime.use_cuda"] = use_cuda
            self._ocr = RapidOCR(params=params)
            self.backend = "cuda" if use_cuda else "cpu"
            if use_cuda:
                log.info("CUDA acceleration enabled for onnxruntime")
        self._enable_preprocessing = enable_preprocessing
        self._preproc_upscale_min_side = preproc_upscale_min_side
        self._timeout_s = 300  # 5 min per image
        log.info(
            "Engine initialized: %s %s backend=%s (preprocessing=%s, box_thresh=%.2f, unclip=%.2f, side=%d, angle_cls=%s)",
            ocr_version, model_type, self.backend, enable_preprocessing, det_box_thresh, det_unclip_ratio,
            det_limit_side_len, use_angle_cls,
        )

    def predict(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        log.debug("OCR start: %s", image_path.name)
        result_holder = [None, None]  # [result, error]

        def _run():
            try:
                result_holder[0] = self._do_ocr(image_path)
            except Exception as e:
                result_holder[1] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout_s)

        if thread.is_alive():
            log.error("OCR TIMEOUT: %s after %ds", image_path.name, self._timeout_s)
            print(f"  !! OCR TIMEOUT: {image_path.name} after {self._timeout_s}s — skipping", flush=True)
            return PagePrediction(image=image_path.name, lines=[], elapsed_ms=self._timeout_s * 1000)

        if result_holder[1]:
            log.error("OCR ERROR: %s — %s", image_path.name, result_holder[1])
            print(f"  !! OCR ERROR: {image_path.name} — {result_holder[1]}", flush=True)
            return PagePrediction(image=image_path.name, lines=[], elapsed_ms=(time.perf_counter() - t0) * 1000)

        pred = result_holder[0]
        elapsed = (time.perf_counter() - t0) * 1000
        log.info("OCR done: %s → %d lines in %.0fms", image_path.name, len(pred.lines), elapsed)
        return pred

    def _do_ocr(self, image_path: Path) -> PagePrediction:
        t0 = time.perf_counter()
        scale = 1.0
        if self._enable_preprocessing:
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                log.debug("  image loaded: %dx%d", w, h)
                img, scale = _preprocess(img, self._preproc_upscale_min_side)
                log.debug("  preprocessed: scale=%.2f", scale)
                result = self._ocr(img)
            else:
                log.warning("  cv2.imread failed, passing path directly: %s", image_path)
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
        elapsed = (time.perf_counter() - t0) * 1000
        return PagePrediction(
            image=image_path.name,
            lines=lines,
            elapsed_ms=elapsed,
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