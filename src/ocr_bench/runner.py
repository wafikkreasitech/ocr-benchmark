"""Benchmark runner — iterates categories, computes metrics, writes reports."""
from __future__ import annotations

import csv
import json
import logging
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

log = logging.getLogger(__name__)

# Generation counter — each run() bumps it. A running loop bails the moment a
# newer run starts (e.g. user clicks Restart), so two runs never race on the
# same report files. ponytail: in-process counter; fine because BackgroundTasks
# run in this same process. Becomes a shared store only if runs ever go remote.
_RUN_GEN = 0


def _next_gen() -> int:
    global _RUN_GEN
    _RUN_GEN += 1
    return _RUN_GEN


class _Superseded(Exception):
    """Raised inside the loop when a newer run has started."""


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


def _serialize_page_metrics(pm: PageMetrics, page: GroundTruthPage, pred: PagePrediction,
                            iou_threshold: float = 0.5) -> dict:
    """Per-image JSON: includes overlay data (polygons) for the UI."""
    matches, unmatched_gt, unmatched_pr = match(page.lines, pred.lines, iou_threshold=iou_threshold)

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
        ocr_version: str | None = None, model_type: str | None = None,
        det_overrides: dict | None = None) -> dict:
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
    ver = ocr_version or settings.ocr_version
    mtype = model_type or settings.model_type
    overrides = det_overrides or {}
    box_thresh = overrides.get("det_box_thresh", settings.det_box_thresh)
    det_thresh = overrides.get("det_thresh", settings.det_thresh)
    unclip = overrides.get("det_unclip_ratio", settings.det_unclip_ratio)
    limit_side = overrides.get("det_limit_side_len", settings.det_limit_side_len)
    use_angle_cls = overrides.get("use_angle_cls", settings.use_angle_cls)
    rec_batch = overrides.get("rec_batch_num", settings.rec_batch_num)
    rec_width = overrides.get("rec_img_width", settings.rec_img_width)
    # Pre / post overrides — previously hardcoded from settings, now overridable
    # per-run so the UI toggles / sliders actually take effect.
    enable_preprocessing = overrides.get("enable_preprocessing", settings.enable_preprocessing)
    iou_threshold = overrides.get("iou_threshold", settings.iou_threshold)
    enable_symspell_correction = overrides.get("enable_symspell_correction", settings.enable_symspell_correction)
    enable_word_segmentation = overrides.get("enable_word_segmentation", settings.enable_word_segmentation)
    symspell_max_edit_distance = overrides.get("symspell_max_edit_distance", settings.symspell_max_edit_distance)
    kbbi_top_n = overrides.get("kbbi_top_n", settings.kbbi_top_n)
    log.info("=== Benchmark start: %s %s, %d categories ===", ver, mtype, len(cats))
    log.info("Config: preprocessing=%s, iou=%.2f, corrector=%s, box_thresh=%.2f, unclip=%.2f",
             enable_preprocessing, iou_threshold,
             enable_symspell_correction, box_thresh, unclip)

    engine = BenchEngine(
        enable_preprocessing=enable_preprocessing,
        preproc_upscale_min_side=settings.preproc_upscale_min_side,
        ocr_version=ver,
        model_type=mtype,
        det_box_thresh=box_thresh,
        det_thresh=det_thresh,
        det_unclip_ratio=unclip,
        det_limit_side_len=limit_side,
        use_angle_cls=use_angle_cls,
        rec_batch_num=rec_batch,
        rec_img_width=rec_width,
    )
    # Build a per-run corrector when any override differs from .env so the UI
    # toggles/sliders actually take effect. The singleton is reused when
    # nothing was overridden.
    from .corrector import Corrector
    corrector_overridden = (
        enable_symspell_correction != settings.enable_symspell_correction
        or enable_word_segmentation != settings.enable_word_segmentation
        or symspell_max_edit_distance != settings.symspell_max_edit_distance
        or kbbi_top_n != settings.kbbi_top_n
    )
    if corrector_overridden:
        run_settings_obj = settings.model_copy(update={
            "enable_symspell_correction": enable_symspell_correction,
            "enable_word_segmentation": enable_word_segmentation,
            "symspell_max_edit_distance": symspell_max_edit_distance,
            "kbbi_top_n": kbbi_top_n,
        })
        corrector = Corrector(run_settings_obj)
    else:
        corrector = get_corrector()  # reuse singleton — no override differs

    overall = []
    overall_start = time.perf_counter()

    gen = _next_gen()  # claim this run; older loops will see a newer gen and bail

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

    run_settings = {
        "ocr_version": ver,
        "model_type": mtype,
        "det_box_thresh": box_thresh,
        "det_thresh": det_thresh,
        "det_unclip_ratio": unclip,
        "det_limit_side_len": limit_side,
        "use_angle_cls": use_angle_cls,
        "rec_batch_num": rec_batch,
        "rec_img_width": rec_width,
        "enable_preprocessing": enable_preprocessing,
        "preproc_upscale_min_side": settings.preproc_upscale_min_side,
        "iou_threshold": iou_threshold,
        "enable_symspell_correction": enable_symspell_correction,
        "enable_word_segmentation": enable_word_segmentation,
        "symspell_max_edit_distance": symspell_max_edit_distance,
        "kbbi_top_n": kbbi_top_n,
    }

    try:
        return _run_categories(cats, engine, corrector, settings, run_settings,
                                started_at, completed, overall,
                                overall_start, per_cat_dir, verbose, gen,
                                iou_threshold=iou_threshold)
    except _Superseded:
        log.info("Run %d superseded by a newer run; bailing without touching status", gen)
        return {}
    except BaseException as e:  # incl. SystemExit/KeyboardInterrupt — never strand the lock
        log.exception("Benchmark failed; clearing run status")
        if gen == _RUN_GEN:  # only clear if we still own the lock
            _write_status({
                "running": False,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "total": len(cats),
                "completed": completed,
                "current": None,
                "error": f"{type(e).__name__}: {e}",
            })
        raise


def _run_categories(cats, engine, corrector, settings, run_settings,
                    started_at, completed, overall, overall_start, per_cat_dir,
                    verbose, gen, iou_threshold: float = 0.5) -> dict:
    def _check_superseded():
        if gen != _RUN_GEN:
            raise _Superseded()

    for cat_idx, cat_dir in enumerate(cats, 1):
        _check_superseded()
        pages = load_category(cat_dir)
        if not pages:
            log.warning("[%d/%d] %s: no images, skipping", cat_idx, len(cats), cat_dir.name)
            continue
        tag = " [corrector ON]" if corrector.enabled else ""
        print(f"[{cat_idx}/{len(cats)}] {cat_dir.name}: {len(pages)} images{tag}", flush=True)
        log.info("[%d/%d] %s: %d images%s", cat_idx, len(cats), cat_dir.name, len(pages), tag)

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

        for img_idx, page in enumerate(pages, 1):
            _check_superseded()
            print(f"  [{cat_idx}/{len(cats)}] img {img_idx}/{len(pages)}: {page.image_path.name} — OCR start",
                  flush=True)
            t_img = time.perf_counter()
            pred = engine.predict(page.image_path)
            img_ms = (time.perf_counter() - t_img) * 1000
            print(f"  [{cat_idx}/{len(cats)}] img {img_idx}/{len(pages)}: {page.image_path.name} — "
                  f"OCR done, {len(pred.lines)} lines, {img_ms:.0f}ms", flush=True)
            log.info("  [%d/%d] img %d/%d: %s — OCR done, %d lines, %.0fms",
                     cat_idx, len(cats), img_idx, len(pages),
                     page.image_path.name, len(pred.lines), pred.elapsed_ms)
            pm = _evaluate_page(page, pred)
            log.info("  [%d/%d] img %d/%d: %s — metrics: GT=%d PR=%d TP=%d FP=%d FN=%d CER=%.3f",
                     cat_idx, len(cats), img_idx, len(pages),
                     page.image_path.name, pm.n_gt, pm.n_pred,
                     pm.detection.tp, pm.detection.fp, pm.detection.fn,
                     pm.joined_cer)
            cat_pages.append(pm)
            per_image_payload.append(_serialize_page_metrics(pm, page, pred, iou_threshold=iou_threshold))
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

        cat_elapsed = round(time.perf_counter() - cat_start, 2)
        summary = aggregate_category(cat_dir.name, cat_pages)
        overall.append(summary)
        completed.append({
            "name": cat_dir.name,
            "elapsed_s": cat_elapsed,
        })
        log.info("[%d/%d] %s: done in %.1fs, F1=%.3f CER=%.3f",
                 cat_idx, len(cats), cat_dir.name, cat_elapsed,
                 summary.detection.f1, summary.matched_cer_mean)

        out_file = per_cat_dir / f"{_slug(cat_dir.name)}.json"
        out_file.write_text(
            json.dumps(
                {
                    "category": cat_dir.name,
                    "summary": asdict(summary),
                    "images": per_image_payload,
                    "corrector_enabled": corrector.enabled,
                    "corrector_settings": {
                        "max_edit_distance": run_settings["symspell_max_edit_distance"],
                        "kbbi_top_n": run_settings["kbbi_top_n"],
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    total_elapsed = round(time.perf_counter() - overall_start, 2)
    overall_dict = aggregate_overall(overall)
    overall_dict["total_elapsed_s"] = total_elapsed
    overall_dict["last_run"] = _now_iso()
    overall_dict["corrector_enabled"] = corrector.enabled
    overall_dict["ocr_version"] = run_settings["ocr_version"]
    overall_dict["model_type"] = run_settings["model_type"]
    overall_dict["det_box_thresh"] = run_settings["det_box_thresh"]
    overall_dict["det_thresh"] = run_settings["det_thresh"]
    overall_dict["det_unclip_ratio"] = run_settings["det_unclip_ratio"]
    overall_dict["det_limit_side_len"] = run_settings["det_limit_side_len"]
    overall_dict["use_angle_cls"] = run_settings["use_angle_cls"]
    overall_dict["rec_batch_num"] = run_settings["rec_batch_num"]
    overall_dict["rec_img_width"] = run_settings["rec_img_width"]
    overall_dict["enable_preprocessing"] = run_settings["enable_preprocessing"]
    overall_dict["iou_threshold"] = run_settings["iou_threshold"]
    overall_dict["enable_symspell_correction"] = run_settings["enable_symspell_correction"]
    overall_dict["enable_word_segmentation"] = run_settings["enable_word_segmentation"]
    overall_dict["symspell_max_edit_distance"] = run_settings["symspell_max_edit_distance"]
    overall_dict["kbbi_top_n"] = run_settings["kbbi_top_n"]

    log.info("=== Benchmark done: %d images in %.1fs, F1=%.3f CER=%.3f ===",
             overall_dict["n_images"], total_elapsed,
             overall_dict["detection_f1"], overall_dict["cer_mean"])

    _check_superseded()  # don't clobber a newer run's reports/status
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
    """Atomically write the progress sidecar. Read by /api/progress.

    Stamps ``updated_at`` on every write so the UI can tell a live run from a
    dead one (a crashed process leaves a stale ``running:true`` file behind).

    ``os.replace`` on Windows can fail with WinError 5 when the target is briefly
    held by an AV scanner or a reader; retry a few times before falling back.
    """
    import os
    import time

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    status = {**status, "updated_at": _now_iso()}
    payload = json.dumps(status, ensure_ascii=False)
    tmp = RUN_STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")

    last_err: Exception | None = None
    for attempt in range(8):
        try:
            os.replace(tmp, RUN_STATUS_PATH)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.05 * (attempt + 1))
    # ponytail: best-effort direct write if rename keeps losing the race on Windows.
    RUN_STATUS_PATH.write_text(payload, encoding="utf-8")
    try:
        tmp.unlink()
    except OSError:
        pass
    if last_err is not None:
        import warnings
        warnings.warn(f"_write_status fell back to non-atomic write: {last_err}")


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
    """Save a snapshot of this run to reports/history/ for comparison.

    Stores every knob that influenced the run so future runs can be diffed
    against the best-known config per category.
    """
    HISTORY_ROOT.mkdir(parents=True, exist_ok=True)

    run_id = overall["last_run"].replace(":", "-").replace("T", "_").replace("Z", "")
    snapshot = {
        "id": run_id,
        "timestamp": overall["last_run"],
        # Config snapshot — all knobs, so we can diff against the best run later.
        "config": {
            "ocr_version": overall.get("ocr_version", ""),
            "model_type": overall.get("model_type", ""),
            "det_box_thresh": overall.get("det_box_thresh", 0.5),
            "det_thresh": overall.get("det_thresh", 0.3),
            "det_unclip_ratio": overall.get("det_unclip_ratio", 1.6),
            "det_limit_side_len": overall.get("det_limit_side_len", 1536),
            "use_angle_cls": overall.get("use_angle_cls", False),
            "rec_batch_num": overall.get("rec_batch_num", 6),
            "rec_img_width": overall.get("rec_img_width", 320),
            "enable_preprocessing": overall.get("enable_preprocessing", False),
            "iou_threshold": overall.get("iou_threshold", 0.5),
            "enable_symspell_correction": overall.get("enable_symspell_correction", False),
            "enable_word_segmentation": overall.get("enable_word_segmentation", False),
            "symspell_max_edit_distance": overall.get("symspell_max_edit_distance", 1),
            "kbbi_top_n": overall.get("kbbi_top_n", 0),
        },
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
    cfg = snapshot["config"]
    index.append({
        "id": run_id,
        "timestamp": overall["last_run"],
        # Flat keys preserved for UI backward-compat (history table reads these).
        "ocr_version": cfg["ocr_version"],
        "model_type": cfg["model_type"],
        "config": cfg,  # nested block — full snapshot for /api/history/best etc.
        "n_images": overall.get("n_images", 0),
        "f1": overall.get("detection_f1", 0),
        "cer": overall.get("cer_mean", 0),
        "wer": overall.get("wer_mean", 0),
        "cer_corrected": overall.get("cer_corrected_mean", 0),
        "wer_corrected": overall.get("wer_corrected_mean", 0),
        "total_elapsed_s": overall.get("total_elapsed_s", 0),
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