"""CER / WER + detection precision/recall/F1.

CER/WER via ``jiwer``. Strings are normalized: NFKC, lowercase, whitespace
collapse, strip. This is the fair comparison for Latin Indonesian where
capitalization/punctuation are annotation noise.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import jiwer


def normalize(text: str) -> str:
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = " ".join(t.split())  # collapse all whitespace
    return t.strip()


def cer(ref: str, hyp: str) -> float:
    """Character error rate, in [0, inf). 0.0 means perfect."""
    if not ref:
        return 0.0 if not hyp else 1.0
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.cer(r, h)


def wer(ref: str, hyp: str) -> float:
    """Word error rate, in [0, inf)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return jiwer.wer(r, h)


@dataclass
class DetectionStats:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class PageMetrics:
    image: str
    category: str
    n_gt: int
    n_pred: int
    detection: DetectionStats
    matched_cer: list[float]   # CER per matched pair (raw OCR)
    matched_wer: list[float]
    matched_conf: list[float]
    matched_cer_corrected: list[float]   # CER after corrector (empty if disabled)
    matched_wer_corrected: list[float]
    correction_status: dict[str, int]    # {unchanged, corrected, not_found}
    joined_cer: float          # coarse full-page CER (raw)
    joined_cer_corrected: float
    elapsed_ms: float
    empty_output: bool


@dataclass
class CategorySummary:
    category: str
    n_images: int
    n_lines_total: int
    detection: DetectionStats
    matched_cer_mean: float
    matched_cer_median: float
    matched_wer_mean: float
    joined_cer_mean: float
    matched_cer_corrected_mean: float
    matched_wer_corrected_mean: float
    joined_cer_corrected_mean: float
    mean_confidence: float
    mean_ms_per_image: float
    empty_output_rate: float


def aggregate_category(category: str, pages: list[PageMetrics]) -> CategorySummary:
    if not pages:
        return CategorySummary(
            category=category, n_images=0, n_lines_total=0,
            detection=DetectionStats(0, 0, 0),
            matched_cer_mean=0.0, matched_cer_median=0.0, matched_wer_mean=0.0,
            joined_cer_mean=0.0,
            matched_cer_corrected_mean=0.0, matched_wer_corrected_mean=0.0,
            joined_cer_corrected_mean=0.0,
            mean_confidence=0.0, mean_ms_per_image=0.0,
            empty_output_rate=0.0,
        )

    tp = sum(p.detection.tp for p in pages)
    fp = sum(p.detection.fp for p in pages)
    fn = sum(p.detection.fn for p in pages)
    all_cer = [c for p in pages for c in p.matched_cer]
    all_wer = [w for p in pages for w in p.matched_wer]
    all_conf = [c for p in pages for c in p.matched_conf]
    all_cer_c = [c for p in pages for c in p.matched_cer_corrected]
    all_wer_c = [w for p in pages for w in p.matched_wer_corrected]
    joined = [p.joined_cer for p in pages if p.n_gt > 0]
    joined_c = [p.joined_cer_corrected for p in pages if p.n_gt > 0]
    empty = sum(1 for p in pages if p.empty_output)

    all_cer_sorted = sorted(all_cer)
    median_cer = (
        all_cer_sorted[len(all_cer_sorted) // 2] if all_cer_sorted else 0.0
    )

    return CategorySummary(
        category=category,
        n_images=len(pages),
        n_lines_total=sum(p.n_gt for p in pages),
        detection=DetectionStats(tp, fp, fn),
        matched_cer_mean=(sum(all_cer) / len(all_cer)) if all_cer else 0.0,
        matched_cer_median=median_cer,
        matched_wer_mean=(sum(all_wer) / len(all_wer)) if all_wer else 0.0,
        joined_cer_mean=(sum(joined) / len(joined)) if joined else 0.0,
        matched_cer_corrected_mean=(sum(all_cer_c) / len(all_cer_c)) if all_cer_c else 0.0,
        matched_wer_corrected_mean=(sum(all_wer_c) / len(all_wer_c)) if all_wer_c else 0.0,
        joined_cer_corrected_mean=(sum(joined_c) / len(joined_c)) if joined_c else 0.0,
        mean_confidence=(sum(all_conf) / len(all_conf)) if all_conf else 0.0,
        mean_ms_per_image=sum(p.elapsed_ms for p in pages) / len(pages),
        empty_output_rate=empty / len(pages),
    )


def aggregate_overall(per_category: list[CategorySummary]) -> dict:
    """Overall rollup across categories."""
    if not per_category:
        return {"detection_f1": 0.0, "cer_mean": 0.0, "wer_mean": 0.0, "n_images": 0, "n_lines": 0}
    tp = sum(c.detection.tp for c in per_category)
    fp = sum(c.detection.fp for c in per_category)
    fn = sum(c.detection.fn for c in per_category)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    n_imgs = sum(c.n_images for c in per_category)
    cer_mean = sum(c.matched_cer_mean * c.n_images for c in per_category) / n_imgs if n_imgs else 0.0
    wer_mean = sum(c.matched_wer_mean * c.n_images for c in per_category) / n_imgs if n_imgs else 0.0
    cer_c_mean = sum(c.matched_cer_corrected_mean * c.n_images for c in per_category) / n_imgs if n_imgs else 0.0
    wer_c_mean = sum(c.matched_wer_corrected_mean * c.n_images for c in per_category) / n_imgs if n_imgs else 0.0
    return {
        "detection_precision": p,
        "detection_recall": r,
        "detection_f1": f1,
        "cer_mean": cer_mean,
        "wer_mean": wer_mean,
        "cer_corrected_mean": cer_c_mean,
        "wer_corrected_mean": wer_c_mean,
        "n_images": n_imgs,
        "n_lines": sum(c.n_lines_total for c in per_category),
        "n_categories": len(per_category),
    }


if __name__ == "__main__":  # ponytail: self-check
    assert cer("HELLO world", "hello  WORLD") == 0.0, "normalize failed"
    assert cer("abc", "axc") == 1 / 3, f"got {cer('abc','axc')}"
    assert wer("the cat sat", "the cat sit") == 1 / 3, f"got {wer('the cat sat','the cat sit')}"
    d = DetectionStats(7, 2, 1)
    assert abs(d.precision - 7/9) < 1e-9
    assert abs(d.recall - 7/8) < 1e-9
    print("metrics self-check: OK")