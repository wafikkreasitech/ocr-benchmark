"""TTS benchmark runner — speaks the OCR results, measures synthesis speed.

Reads the OCR benchmark's reports/per_category/*.json, joins each page's text
(predicted by default — exactly what ai4db's TTS receives in production), runs
Piper over it, and aggregates latency / RTF / throughput per category + overall
into reports/tts_summary.json (+ .csv).

No ground-truth audio exists, so we measure speed honestly, not voice quality.
See docs/plan-tts.md §6.
"""
from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path

from .config import get_settings
from .paths import REPORTS_ROOT
from .tts_engine import TTSEngine

log = logging.getLogger(__name__)

TTS_STATUS_PATH = REPORTS_ROOT / ".tts_status.json"
TTS_SUMMARY_PATH = REPORTS_ROOT / "tts_summary.json"
TTS_SUMMARY_CSV = REPORTS_ROOT / "tts_summary.csv"

_CSV_COLUMNS = ["category", "pages", "chars", "rtf_mean", "rtf_median",
                "synth_ms_mean", "first_chunk_ms_mean", "chars_per_sec_mean",
                "audio_seconds_total", "failures"]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_status(status: dict) -> None:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({**status, "updated_at": _now_iso()}, ensure_ascii=False)
    TTS_STATUS_PATH.write_text(payload, encoding="utf-8")


def _page_text(image: dict, source: str) -> str:
    """Join a page's line texts. source='pred' uses OCR output, 'gt' uses truth."""
    key = "gt_text" if source == "gt" else "pr_text"
    lines = [ov.get(key, "") for ov in image.get("overlays", [])]
    return " ".join(t.strip() for t in lines if t and t.strip())


def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0}
    return {"mean": statistics.fmean(values), "median": statistics.median(values)}


def run(source: str | None = None) -> dict:
    """Synthesize every OCR-result page and aggregate speed metrics.

    Returns the summary dict; also writes tts_summary.{json,csv}.
    """
    settings = get_settings()
    source = (source or settings.tts_source or "pred").lower()
    if source not in ("pred", "gt"):
        source = "pred"

    per_cat_dir = REPORTS_ROOT / "per_category"
    cat_files = sorted(per_cat_dir.glob("*.json")) if per_cat_dir.exists() else []
    if not cat_files:
        _write_status({"running": False, "error": "no OCR reports — run the OCR benchmark first"})
        raise RuntimeError("no reports/per_category/*.json — run the OCR benchmark first")

    # Count total pages for progress.
    total_pages = 0
    parsed: list[dict] = []
    for f in cat_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        parsed.append(data)
        total_pages += len(data.get("images", []))

    _write_status({"running": True, "started_at": _now_iso(),
                   "total": total_pages, "done": 0, "current": None, "source": source})

    engine = TTSEngine(settings.piper_voice_path)  # raises if voice missing

    per_category: list[dict] = []
    all_rtf: list[float] = []
    all_synth: list[float] = []
    all_first: list[float] = []
    all_cps: list[float] = []
    total_chars = 0
    total_audio = 0.0
    total_failures = 0
    done = 0

    for data in parsed:
        category = data.get("category", "?")
        rtf_l, synth_l, first_l, cps_l = [], [], [], []
        cat_chars = 0
        cat_audio = 0.0
        cat_fail = 0
        pages_spoken = 0

        for image in data.get("images", []):
            done += 1
            text = _page_text(image, source)
            if not text:
                continue
            _write_status({"running": True, "total": total_pages, "done": done,
                           "current": f"{category} / {image.get('image', '?')}",
                           "source": source})
            try:
                _pcm, r = engine.synthesize(text)
            except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the run
                log.warning("TTS failed on %s: %s", image.get("image"), e)
                cat_fail += 1
                total_failures += 1
                continue
            pages_spoken += 1
            rtf_l.append(r.rtf); synth_l.append(r.synth_ms)
            first_l.append(r.first_chunk_ms); cps_l.append(r.chars_per_sec)
            cat_chars += r.n_chars; cat_audio += r.audio_seconds

        rtf_agg = _agg(rtf_l)
        per_category.append({
            "category": category,
            "pages": pages_spoken,
            "chars": cat_chars,
            "rtf_mean": rtf_agg["mean"],
            "rtf_median": rtf_agg["median"],
            "synth_ms_mean": _agg(synth_l)["mean"],
            "first_chunk_ms_mean": _agg(first_l)["mean"],
            "chars_per_sec_mean": _agg(cps_l)["mean"],
            "audio_seconds_total": cat_audio,
            "failures": cat_fail,
        })
        all_rtf += rtf_l; all_synth += synth_l; all_first += first_l; all_cps += cps_l
        total_chars += cat_chars; total_audio += cat_audio

    overall = {
        "pages": sum(c["pages"] for c in per_category),
        "chars": total_chars,
        "rtf_mean": _agg(all_rtf)["mean"],
        "rtf_median": _agg(all_rtf)["median"],
        "synth_ms_mean": _agg(all_synth)["mean"],
        "first_chunk_ms_mean": _agg(all_first)["mean"],
        "chars_per_sec_mean": _agg(all_cps)["mean"],
        "audio_seconds_total": total_audio,
        "failures": total_failures,
        "source": source,
        "voice": Path(settings.piper_voice_path).stem,
        "last_run": _now_iso(),
    }
    summary = {"overall": overall, "per_category": per_category}

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    TTS_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    with TTS_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for c in per_category:
            w.writerow({k: c[k] for k in _CSV_COLUMNS})

    _write_status({"running": False, "total": total_pages, "done": done,
                   "current": None, "source": source, "completed_at": _now_iso()})
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=__import__("sys").stderr)
    s = run()
    o = s["overall"]
    print(f"TTS benchmark: {o['pages']} pages, RTF={o['rtf_mean']:.3f}, "
          f"synth={o['synth_ms_mean']:.0f}ms/page, {o['failures']} failures")
