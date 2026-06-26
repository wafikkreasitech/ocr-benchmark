"""Benchmark runner — iterates categories, computes metrics, writes reports."""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import get_settings
from .corrector import get_corrector
from .dataset import GroundTruthPage, list_categories, load_category
from .engine import BenchEngine, LinePrediction, PagePrediction
from .matcher import match
from .metrics import (
    CategorySummary,
    DetectionStats,
    PageMetrics,
    aggregate_category,
    aggregate_overall,
    cer,
    wer,
)
from .paths import HISTORY_ROOT, REPORTS_ROOT


def _evaluate_page(page: GroundTruthPage, pred: PagePrediction) -> PageMetrics:
    gt_lines = page.lines
    pr_lines = pred.lines
    matches, unmatched_gt, unmatched_pr = match(gt_lines, pr_lines)

    corrector = get_corrector()
    enabled = corrector.enabled

    matched_cer: list[float] = []
    matched_wer: list[float] = []
    matched_conf: list[float] = []
    matched_cer_c: list[float] = []
    matched_wer_c: list[float] = []
    status_counts: dict[str, int] = {"unchanged": 0, "corrected": 0, "not_found": 0}

    for g, p, _ in matches:
        matched_cer.append(cer(g.text, p.text))
        matched_wer.append(wer(g.text, p.text))
        matched_conf.append(p.score)
        if enabled:
            r = corrector.correct(p.text)
            status_counts[r.status] = status_counts.get(r.status, 0) + 1
            matched_cer_c.append(cer(g.text, r.corrected))
            matched_wer_c.append(wer(g.text, r.corrected))

    gt_joined = "\n".join(g.text for g in gt_lines)
    pr_joined = "\n".join(p.text for p in pr_lines)
    joined_cer_val = cer(gt_joined, pr_joined)

    if enabled:
        pr_joined_c = "\n".join(corrector.correct(p.text).corrected for p in pr_lines)
        joined_cer_c = cer(gt_joined, pr_joined_c)
    else:
        joined_cer_c = 0.0

    return PageMetrics(
        image=page.image_path.name,
        category=page.category,
        n_gt=len(gt_lines),
        n_pred=len(pr_lines),
        detection=DetectionStats(
            tp=len(matches),
            fp=len(unmatched_pr),
            fn=len(unmatched_gt),
        ),
        matched_cer=matched_cer,
        matched_wer=matched_wer,
        matched_conf=matched_conf,
        matched_cer_corrected=matched_cer_c,
        matched_wer_corrected=matched_wer_c,
        correction_status=status_counts,
        joined_cer=joined_cer_val,
        joined_cer_corrected=joined_cer_c,
        elapsed_ms=pred.elapsed_ms,
        empty_output=(len(pr_lines) == 0 and len(gt_lines) > 0),
    )


def _serialize_page_metrics(pm: PageMetrics, page: GroundTruthPage, pred: PagePrediction) -> dict:
    """Per-image JSON: includes overlay data (polygons) for the UI."""
    matches, unmatched_gt, unmatched_pr = match(page.lines, pred.lines)

    corrector = get_corrector()
    enabled = corrector.enabled

    overlays = []
    for g, p, iou_val in matches:
        entry = {
            "gt_polygon": g.polygon, "gt_text": g.text,
            "pr_polygon": p.polygon, "pr_text": p.text,
            "pr_score": p.score, "iou": iou_val, "status": "matched",
            "line_cer": cer(g.text, p.text),
        }
        if enabled:
            r = corrector.correct(p.text)
            entry["pr_text_corrected"] = r.corrected
            entry["correction_status"] = r.status
        overlays.append(entry)
    for g in unmatched_gt:
        overlays.append({
            "gt_polygon": g.polygon, "gt_text": g.text,
            "pr_polygon": None, "pr_text": None, "pr_score": None,
            "iou": None, "status": "missed",
        })
    for p in unmatched_pr:
        entry = {
            "gt_polygon": None, "gt_text": None,
            "pr_polygon": p.polygon, "pr_text": p.text, "pr_score": p.score,
            "iou": None, "status": "spurious",
        }
        if enabled:
            r = corrector.correct(p.text)
            entry["pr_text_corrected"] = r.corrected
            entry["correction_status"] = r.status
        overlays.append(entry)

    return {
        "image": pm.image,
        "category": pm.category,
        "n_gt": pm.n_gt,
        "n_pred": pm.n_pred,
        "detection": asdict(pm.detection),
        "matched_cer_mean": (sum(pm.matched_cer) / len(pm.matched_cer)) if pm.matched_cer else None,
        "matched_wer_mean": (sum(pm.matched_wer) / len(pm.matched_wer)) if pm.matched_wer else None,
        "matched_cer_corrected_mean": (sum(pm.matched_cer_corrected) / len(pm.matched_cer_corrected)) if pm.matched_cer_corrected else None,
        "matched_wer_corrected_mean": (sum(pm.matched_wer_corrected) / len(pm.matched_wer_corrected)) if pm.matched_wer_corrected else None,
        "mean_confidence": (sum(pm.matched_conf) / len(pm.matched_conf)) if pm.matched_conf else None,
        "joined_cer": pm.joined_cer,
        "joined_cer_corrected": pm.joined_cer_corrected,
        "correction_enabled": enabled,
        "correction_status": pm.correction_status,
        "elapsed_ms": pm.elapsed_ms,
        "empty_output": pm.empty_output,
        "overlays": overlays,
    }


def run(root: Path | None = None, only_categories: list[str] | None = None, verbose: bool = True,
        ocr_version: str | None = None, model_type: str | None = None) -> dict:
    cats = list_categories(root)
    if only_categories:
        only_set = set(only_categories)
        cats = [c for c in cats if c.name in only_set]
    if not cats:
        raise SystemExit("no categories found")

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    per_cat_dir = REPORTS_ROOT / "per_category"
    per_cat_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    engine = BenchEngine(
        enable_preprocessing=settings.enable_preprocessing,
        preproc_upscale_min_side=settings.preproc_upscale_min_side,
        ocr_version=ocr_version or settings.ocr_version,
        model_type=model_type or settings.model_type,
    )
    corrector = get_corrector()  # lazy-loads dict if enabled

    overall = []
    overall_start = time.perf_counter()

    # Progress sidecar — read by /api/progress for the dashboard.
    started_at = _now_iso()
    completed: list[dict] = []
    _write_status({
        "running": True,
        "started_at": started_at,
        "total": len(cats),
        "completed": completed,
        "current": None,
    })

    for cat_dir in cats:
        pages = load_category(cat_dir)
        if not pages:
            continue
        if verbose:
            tag = " [corrector ON]" if corrector.enabled else ""
            print(f"[{cat_dir.name}] {len(pages)} images{tag}", flush=True)

        _write_status({
            "running": True,
            "started_at": started_at,
            "total": len(cats),
            "completed": completed,
            "current": {"name": cat_dir.name, "total_images": len(pages), "done_images": 0},
        })

        cat_pages: list[PageMetrics] = []
        per_image_payload = []
        cat_start = time.perf_counter()

        for page in pages:
            pred = engine.predict(page.image_path)
            pm = _evaluate_page(page, pred)
            cat_pages.append(pm)
            per_image_payload.append(_serialize_page_metrics(pm, page, pred))
            _write_status({
                "running": True,
                "started_at": started_at,
                "total": len(cats),
                "completed": completed,
                "current": {
                    "name": cat_dir.name,
                    "total_images": len(pages),
                    "done_images": len(cat_pages),
                },
            })
            if verbose:
                cer_c = (
                    f" CER_c={pm.joined_cer_corrected:.3f}" if corrector.enabled else ""
                )
                print(
                    f"  {page.image_path.name}: GT={pm.n_gt} PR={pm.n_pred} "
                    f"TP={pm.detection.tp} FP={pm.detection.fp} FN={pm.detection.fn} "
                    f"joined_CER={pm.joined_cer:.3f}{cer_c} {pm.elapsed_ms:.0f}ms",
                    flush=True,
                )

        summary = aggregate_category(cat_dir.name, cat_pages)
        overall.append(summary)
        completed.append({
            "name": cat_dir.name,
            "elapsed_s": round(time.perf_counter() - cat_start, 2),
        })

        out_file = per_cat_dir / f"{_slug(cat_dir.name)}.json"
        out_file.write_text(
            json.dumps(
                {
                    "category": cat_dir.name,
                    "summary": asdict(summary),
                    "images": per_image_payload,
                    "corrector_enabled": corrector.enabled,
                    "corrector_settings": {
                        "max_edit_distance": settings.symspell_max_edit_distance,
                        "kbbi_top_n": settings.kbbi_top_n,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    overall_dict = aggregate_overall(overall)
    overall_dict["total_elapsed_s"] = round(time.perf_counter() - overall_start, 2)
    overall_dict["last_run"] = _now_iso()
    overall_dict["corrector_enabled"] = corrector.enabled
    overall_dict["ocr_version"] = ocr_version or settings.ocr_version
    overall_dict["model_type"] = model_type or settings.model_type

    _write_summary_csv(overall, overall_dict)
    _write_overall_json(overall, overall_dict)
    _save_to_history(overall_dict, overall)
    _write_status({
        "running": False,
        "started_at": started_at,
        "finished_at": _now_iso(),
        "total": len(cats),
        "completed": completed,
        "current": None,
    })

    if verbose:
        _print_overall(overall_dict, overall)

    return overall_dict


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


RUN_STATUS_PATH = REPORTS_ROOT / ".run_status.json"


def _write_status(status: dict) -> None:
    """Atomically write the progress sidecar. Read by /api/progress."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = RUN_STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RUN_STATUS_PATH)


def _now_iso() -> str:
    """ISO-8601 UTC timestamp, e.g. 2026-06-25T14:32:07Z."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_summary_csv(per_cat: list[CategorySummary], overall: dict) -> None:
    csv_path = REPORTS_ROOT / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "category", "n_images", "n_lines", "precision", "recall", "f1",
            "cer_mean", "cer_median", "wer_mean", "joined_cer_mean",
            "cer_corrected_mean", "wer_corrected_mean", "joined_cer_corrected_mean",
            "mean_confidence", "mean_ms_per_image", "empty_output_rate",
        ])
        for c in per_cat:
            w.writerow([
                c.category, c.n_images, c.n_lines_total,
                round(c.detection.precision, 4), round(c.detection.recall, 4),
                round(c.detection.f1, 4),
                round(c.matched_cer_mean, 4), round(c.matched_cer_median, 4),
                round(c.matched_wer_mean, 4), round(c.joined_cer_mean, 4),
                round(c.matched_cer_corrected_mean, 4),
                round(c.matched_wer_corrected_mean, 4),
                round(c.joined_cer_corrected_mean, 4),
                round(c.mean_confidence, 4), round(c.mean_ms_per_image, 1),
                round(c.empty_output_rate, 4),
            ])
        w.writerow([
            "OVERALL", overall["n_images"], overall["n_lines"],
            round(overall["detection_precision"], 4),
            round(overall["detection_recall"], 4),
            round(overall["detection_f1"], 4),
            round(overall["cer_mean"], 4), "-",
            round(overall["wer_mean"], 4), "-",
            round(overall.get("cer_corrected_mean", 0.0), 4), "-", "-",
            "-", "-", "-",
        ])


def _write_overall_json(per_cat: list[CategorySummary], overall: dict) -> None:
    out = {
        "overall": overall,
        "per_category": [
            {
                "category": c.category,
                "n_images": c.n_images,
                "n_lines": c.n_lines_total,
                "detection": asdict(c.detection),
                "cer_mean": c.matched_cer_mean,
                "cer_median": c.matched_cer_median,
                "wer_mean": c.matched_wer_mean,
                "joined_cer_mean": c.joined_cer_mean,
                "cer_corrected_mean": c.matched_cer_corrected_mean,
                "wer_corrected_mean": c.matched_wer_corrected_mean,
                "joined_cer_corrected_mean": c.joined_cer_corrected_mean,
                "mean_confidence": c.mean_confidence,
                "mean_ms_per_image": c.mean_ms_per_image,
                "empty_output_rate": c.empty_output_rate,
            }
            for c in per_cat
        ],
    }
    (REPORTS_ROOT / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_to_history(overall: dict, per_cat: list[CategorySummary]) -> None:
    """Save a snapshot of this run to reports/history/ for comparison."""
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

    run_id = overall["last_run"].replace(":", "-").replace("T", "_").replace("Z", "")
    snapshot = {
        "id": run_id,
        "timestamp": overall["last_run"],
        "ocr_version": overall.get("ocr_version", ""),
        "model_type": overall.get("model_type", ""),
        "corrector_enabled": overall.get("corrector_enabled", False),
        "total_elapsed_s": overall.get("total_elapsed_s", 0),
        "overall": overall,
        "per_category": [
            {
                "category": c.category,
                "n_images": c.n_images,
                "n_lines": c.n_lines_total,
                "f1": c.detection.f1,
                "cer": c.matched_cer_mean,
                "wer": c.matched_wer_mean,
                "mean_conf": c.mean_confidence,
                "ms_per_img": c.mean_ms_per_image,
            }
            for c in per_cat
        ],
    }

    out_file = HISTORY_ROOT / f"{run_id}.json"
    out_file.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update index
    index_path = HISTORY_ROOT / "index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            index = []
    else:
        index = []

    # Remove duplicate if same timestamp
    index = [e for e in index if e.get("id") != run_id]
    index.append({
        "id": run_id,
        "timestamp": overall["last_run"],
        "ocr_version": overall.get("ocr_version", ""),
        "model_type": overall.get("model_type", ""),
        "n_images": overall.get("n_images", 0),
        "f1": overall.get("detection_f1", 0),
        "cer": overall.get("cer_mean", 0),
        "wer": overall.get("wer_mean", 0),
    })
    # Keep last 50 runs
    index = index[-50:]
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_overall(overall: dict, per_cat: list[CategorySummary]) -> None:
    print()
    print("=" * 72)
    corrector_on = overall.get("corrector_enabled", False)
    print(
        f"OVERALL · {overall['n_images']} images · {overall['n_lines']} GT lines · "
        f"{overall.get('total_elapsed_s', '?')}s · corrector={'ON' if corrector_on else 'OFF'}"
    )
    print(f"  Detection F1:    {overall['detection_f1']:.4f}  (P={overall['detection_precision']:.3f} R={overall['detection_recall']:.3f})")
    print(f"  CER (matched):   {overall['cer_mean']:.4f}")
    print(f"  WER (matched):   {overall['wer_mean']:.4f}")
    if corrector_on:
        print(f"  CER corrected:   {overall.get('cer_corrected_mean', 0):.4f}")
        print(f"  WER corrected:   {overall.get('wer_corrected_mean', 0):.4f}")
    print()
    header = f"{'category':<25} {'imgs':>5} {'lines':>6} {'F1':>6} {'CER':>6} {'WER':>6}"
    if corrector_on:
        header += f" {'CER_c':>6} {'WER_c':>6}"
    header += f" {'ms/img':>7}"
    print(header)
    for c in per_cat:
        line = (
            f"{c.category[:24]:<25} {c.n_images:>5} {c.n_lines_total:>6} "
            f"{c.detection.f1:>6.3f} {c.matched_cer_mean:>6.3f} "
            f"{c.matched_wer_mean:>6.3f}"
        )
        if corrector_on:
            line += f" {c.matched_cer_corrected_mean:>6.3f} {c.matched_wer_corrected_mean:>6.3f}"
        line += f" {c.mean_ms_per_image:>7.0f}"
        print(line)