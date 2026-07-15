"""Combined OCR+TTS end-to-end benchmark runner.

Runs OCR then immediately TTS on each page in one pass, mirroring ai4db's
actual pipeline (camera → OCR → TTS → audio) end-to-end so a single number
— total_ms — can be compared honestly against running each stage isolated
(see runner.py / tts_runner.py).

Deliberately loads BOTH the OCR engine and the TTS voice into GPU/RAM at the
same time — that's the point of this benchmark: measure the real combined
cost, not each stage in isolation.

Writes reports/combined_summary.json (+ .csv), independent of the existing
summary.json (OCR-only) and tts_summary.json (TTS-only) so neither is
clobbered.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from pathlib import Path

from .config import get_settings
from .dataset import GroundTruthPage, list_categories, load_category
from .engine import BenchEngine
from .paths import REPORTS_ROOT
from .sysmon import ResourceMonitor
from .tts_engine import TTSEngine, _detect_cuda

log = logging.getLogger(__name__)

COMBINED_STATUS_PATH = REPORTS_ROOT / ".combined_status.json"
COMBINED_SUMMARY_PATH = REPORTS_ROOT / "combined_summary.json"
COMBINED_SUMMARY_CSV = REPORTS_ROOT / "combined_summary.csv"

_CSV_COLUMNS = ["category", "pages", "chars", "ocr_ms_mean", "tts_ms_mean",
                "overhead_ms_mean", "total_ms_mean", "rtf_mean",
                "audio_seconds_total", "ocr_failures", "tts_failures"]

# Separate generation counter from runner.py's OCR-only runs — a combined run
# and a plain OCR run can't clobber each other's "am I still the active run"
# check since they use different sidecars anyway, but keep this isolated too.
_RUN_GEN = 0


def _next_gen() -> int:
    global _RUN_GEN
    _RUN_GEN += 1
    return _RUN_GEN


class _Superseded(Exception):
    """Raised inside the loop when a newer combined run has started."""


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(status: dict) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({**status, "updated_at": _now_iso()}, ensure_ascii=False)
    COMBINED_STATUS_PATH.write_text(payload, encoding="utf-8")


def _page_text(page: GroundTruthPage, pred_lines: list, source: str) -> str:
    """Join a page's line texts. source='pred' uses OCR output, 'gt' uses truth."""
    if source == "gt":
        return " ".join(l.text.strip() for l in page.lines if l.text and l.text.strip())
    return " ".join(l.text.strip() for l in pred_lines if l.text and l.text.strip())


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": statistics.fmean(values), "median": statistics.median(values)}


def run(only_categories: list[str] | None = None,
        ocr_version: str | None = None, model_type: str | None = None,
        det_overrides: dict | None = None, dataset_key: str | None = None,
        source: str | None = None) -> dict:
    """Run OCR immediately followed by TTS on every page; aggregate end-to-end timing.

    Returns the summary dict; also writes combined_summary.{json,csv}.
    """
    from .paths import resolve_dataset_root
    dataset_key, root = resolve_dataset_root(dataset_key)
    cats = list_categories(root)
    if only_categories:
        only_set = set(only_categories)
        cats = [c for c in cats if c.name in only_set]
    if not cats:
        raise SystemExit("no categories found")

    settings = get_settings()
    source = (source or settings.tts_source or "pred").lower()
    if source not in ("pred", "gt"):
        source = "pred"

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
    use_tensorrt = overrides.get("use_tensorrt", settings.use_tensorrt)
    enable_preprocessing = overrides.get("enable_preprocessing", settings.enable_preprocessing)

    log.info("=== Combined OCR+TTS benchmark start: %s %s, %d categories, source=%s ===",
             ver, mtype, len(cats), source)

    gen = _next_gen()

    # Sample CPU/RAM/GPU/temp every 2s — both engines loaded simultaneously so
    # this is the number that actually answers "what does running both cost".
    monitor = ResourceMonitor(interval_s=2.0)
    monitor.start()

    started_at = _now_iso()
    completed: list[dict] = []
    total_pages = sum(len(load_category(c)) for c in cats)
    _write_status({
        "running": True, "started_at": started_at, "total": total_pages,
        "done": 0, "completed": completed, "current": None,
        "dataset": dataset_key, "source": source,
        "system": monitor.latest,
    })

    try:
        ocr_engine = BenchEngine(
            enable_preprocessing=enable_preprocessing,
            preproc_upscale_min_side=settings.preproc_upscale_min_side,
            ocr_version=ver, model_type=mtype,
            det_box_thresh=box_thresh, det_thresh=det_thresh,
            det_unclip_ratio=unclip, det_limit_side_len=limit_side,
            use_angle_cls=use_angle_cls, rec_batch_num=rec_batch,
            rec_img_width=rec_width, use_tensorrt=use_tensorrt,
            trt_cache_dir=settings.trt_cache_dir,
        )
        use_cuda = settings.use_cuda_tts and _detect_cuda()
        tts_engine = TTSEngine(settings.piper_voice_path, use_cuda=use_cuda)  # raises if voice missing
    except Exception as e:  # noqa: BLE001 — init failure must clear the lock, not strand it
        log.exception("Combined benchmark failed to init engines")
        _write_status({"running": False, "started_at": started_at,
                       "finished_at": _now_iso(), "error": f"{type(e).__name__}: {e}"})
        monitor.stop()
        raise

    per_category: list[dict] = []
    all_ocr_ms: list[float] = []
    all_tts_ms: list[float] = []
    all_overhead_ms: list[float] = []
    all_total_ms: list[float] = []
    all_rtf: list[float] = []
    total_chars = 0
    total_audio = 0.0
    total_ocr_failures = 0
    total_tts_failures = 0
    done = 0
    overall_start = time.perf_counter()

    def _check_superseded():
        if gen != _RUN_GEN:
            raise _Superseded()

    try:
        for cat_idx, cat_dir in enumerate(cats, 1):
            _check_superseded()
            pages = load_category(cat_dir)
            if not pages:
                continue
            print(f"[{cat_idx}/{len(cats)}] {cat_dir.name}: {len(pages)} images (combined OCR+TTS)",
                  flush=True)

            cat_ocr_ms: list[float] = []
            cat_tts_ms: list[float] = []
            cat_overhead_ms: list[float] = []
            cat_total_ms: list[float] = []
            cat_rtf: list[float] = []
            cat_chars = 0
            cat_audio = 0.0
            cat_ocr_fail = 0
            cat_tts_fail = 0
            pages_done = 0

            for img_idx, page in enumerate(pages, 1):
                _check_superseded()
                done += 1
                _write_status({
                    "running": True, "started_at": started_at, "total": total_pages,
                    "done": done, "completed": completed,
                    "current": {"name": cat_dir.name, "file": page.image_path.name,
                                "total_images": len(pages), "done_images": img_idx - 1},
                    "dataset": dataset_key, "source": source,
                    "system": monitor.latest,
                })

                # ── OCR stage ──
                try:
                    pred = ocr_engine.predict(page.image_path)
                    ocr_ms = pred.elapsed_ms
                except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the run
                    log.warning("Combined: OCR failed on %s: %s", page.image_path.name, e)
                    cat_ocr_fail += 1
                    total_ocr_failures += 1
                    continue

                # ── inter-stage overhead: text extraction/join between OCR output and TTS input ──
                t_over0 = time.perf_counter()
                text = _page_text(page, pred.lines, source)
                overhead_ms = (time.perf_counter() - t_over0) * 1000
                if not text:
                    continue

                # ── TTS stage, fed directly with the OCR output (no round-trip through disk) ──
                try:
                    _pcm, r = tts_engine.synthesize(text)
                except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the run
                    log.warning("Combined: TTS failed on %s: %s", page.image_path.name, e)
                    cat_tts_fail += 1
                    total_tts_failures += 1
                    continue

                total_ms = ocr_ms + overhead_ms + r.synth_ms
                print(f"  [{cat_idx}/{len(cats)}] img {img_idx}/{len(pages)}: {page.image_path.name} — "
                      f"OCR={ocr_ms:.0f}ms TTS={r.synth_ms:.0f}ms total={total_ms:.0f}ms", flush=True)
                cat_ocr_ms.append(ocr_ms)
                cat_tts_ms.append(r.synth_ms)
                cat_overhead_ms.append(overhead_ms)
                cat_total_ms.append(total_ms)
                cat_rtf.append(r.rtf)
                cat_chars += r.n_chars
                cat_audio += r.audio_seconds
                pages_done += 1

            cat_elapsed = round(sum(cat_total_ms) / 1000, 2) if cat_total_ms else 0.0
            per_category.append({
                "category": cat_dir.name,
                "pages": pages_done,
                "chars": cat_chars,
                "ocr_ms_mean": _agg(cat_ocr_ms)["mean"],
                "tts_ms_mean": _agg(cat_tts_ms)["mean"],
                "overhead_ms_mean": _agg(cat_overhead_ms)["mean"],
                "total_ms_mean": _agg(cat_total_ms)["mean"],
                "rtf_mean": _agg(cat_rtf)["mean"],
                "audio_seconds_total": cat_audio,
                "ocr_failures": cat_ocr_fail,
                "tts_failures": cat_tts_fail,
            })
            completed.append({"name": cat_dir.name, "elapsed_s": cat_elapsed})
            all_ocr_ms += cat_ocr_ms
            all_tts_ms += cat_tts_ms
            all_overhead_ms += cat_overhead_ms
            all_total_ms += cat_total_ms
            all_rtf += cat_rtf
            total_chars += cat_chars
            total_audio += cat_audio

        total_elapsed = round(time.perf_counter() - overall_start, 2)
        overall = {
            "pages": sum(c["pages"] for c in per_category),
            "chars": total_chars,
            "ocr_ms_mean": _agg(all_ocr_ms)["mean"],
            "tts_ms_mean": _agg(all_tts_ms)["mean"],
            "overhead_ms_mean": _agg(all_overhead_ms)["mean"],
            "total_ms_mean": _agg(all_total_ms)["mean"],
            "total_ms_median": _agg(all_total_ms)["median"],
            "rtf_mean": _agg(all_rtf)["mean"],
            "audio_seconds_total": total_audio,
            "ocr_failures": total_ocr_failures,
            "tts_failures": total_tts_failures,
            "total_elapsed_s": total_elapsed,
            "source": source,
            "dataset": dataset_key,
            "ocr_version": ver,
            "model_type": mtype,
            "ocr_backend": ocr_engine.backend,
            "tts_backend": tts_engine.backend,
            "voice": Path(settings.piper_voice_path).stem,
            "last_run": _now_iso(),
        }
        # Resource usage rolled up over the whole run — both engines resident
        # in GPU/RAM together, so this is the real combined-load number.
        overall["resources"] = monitor.summary()
        summary = {"overall": overall, "per_category": per_category}

        _check_superseded()  # don't clobber a newer run's reports/status
        REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
        COMBINED_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
        with COMBINED_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
            w.writeheader()
            for c in per_category:
                w.writerow({k: c[k] for k in _CSV_COLUMNS})

        _write_status({
            "running": False, "started_at": started_at, "finished_at": _now_iso(),
            "total": total_pages, "done": done, "completed": completed,
            "current": None, "dataset": dataset_key, "source": source,
        })
        log.info("=== Combined benchmark done: %d pages in %.1fs, total_ms_mean=%.0f "
                 "(ocr=%.0f + overhead=%.0f + tts=%.0f) ===",
                 overall["pages"], total_elapsed, overall["total_ms_mean"],
                 overall["ocr_ms_mean"], overall["overhead_ms_mean"], overall["tts_ms_mean"])
        return summary
    except _Superseded:
        log.info("Combined run %d superseded by a newer run; bailing without touching status", gen)
        return {}
    except BaseException as e:  # incl. SystemExit/KeyboardInterrupt — never strand the lock
        log.exception("Combined benchmark failed; clearing run status")
        if gen == _RUN_GEN:  # only clear if we still own the lock
            _write_status({
                "running": False, "started_at": started_at, "finished_at": _now_iso(),
                "total": total_pages, "completed": completed, "current": None,
                "error": f"{type(e).__name__}: {e}",
            })
        raise
    finally:
        monitor.stop()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    s = run()
    o = s["overall"]
    print(f"Combined benchmark: {o['pages']} pages, total_ms_mean={o['total_ms_mean']:.0f}ms "
          f"(ocr={o['ocr_ms_mean']:.0f}ms + overhead={o['overhead_ms_mean']:.0f}ms + "
          f"tts={o['tts_ms_mean']:.0f}ms), {o['ocr_failures']} ocr failures, "
          f"{o['tts_failures']} tts failures")
