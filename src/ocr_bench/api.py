"""FastAPI: serves dashboard + JSON endpoints.

Endpoints:
  POST /api/run                 run benchmark (background; poll /api/progress)
  GET  /api/progress            live run status (running, completed, current)
  GET  /api/summary             aggregated metrics (overall + per-category)
  GET  /api/results/<category>  per-image detail incl. overlay data
  GET  /api/image/<cat>/<file>  raw image bytes
  GET  /api/models              available OCR model combinations
  GET  /api/config              current runtime config
  GET  /api/tts?text=…          synthesize text -> WAV (503 if voice missing)
  POST /api/tts/run             run TTS benchmark over OCR results (background)
  GET  /api/tts/summary         aggregated TTS metrics (overall + per-category)
  GET  /                        dashboard (ui/index.html)
  GET  /tts                     TTS benchmark dashboard (ui/tts.html)
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .paths import DATASETS, DEFAULT_DATASET_ROOT, HISTORY_ROOT, REPORTS_ROOT, UI_ROOT, resolve_dataset_root
from .runner import RUN_STATUS_PATH, run as run_benchmark

# Lazy TTS engine — loaded on first /api/tts call so a clone with no voice
# model still starts and serves the OCR dashboard. None until loaded.
_tts_engine = None


def _get_tts_engine():
    """Return a cached TTSEngine, or raise HTTPException(503) if unavailable.

    503 (not 500) so the UI's Listen button can show "voice model not
    downloaded — run scripts/download_voice" instead of a stack trace.
    """
    global _tts_engine
    if _tts_engine is not None:
        return _tts_engine
    from .config import get_settings
    from .tts_engine import TTSEngine
    voice_path = Path(get_settings().piper_voice_path)
    if not voice_path.exists():
        raise HTTPException(
            503,
            f"TTS voice model not found at {voice_path}. "
            f"Run: uv run python -m scripts.download_voice",
        )
    try:
        _tts_engine = TTSEngine(str(voice_path))
    except Exception as e:  # noqa: BLE001 — piper load failure -> 503, not 500
        raise HTTPException(503, f"TTS voice failed to load: {e}") from e
    return _tts_engine

AVAILABLE_MODELS = {
    "PP-OCRv6": {"model_types": ["tiny", "small", "medium"], "default": "small",
                 "desc": "Latest, best accuracy"},
    "PP-OCRv5": {"model_types": ["mobile", "server"], "default": "mobile",
                 "desc": "Per-language rec models"},
    "PP-OCRv4": {"model_types": ["mobile", "server"], "default": "mobile",
                 "desc": "Legacy, stable"},
}

# Display labels for the dataset registry. Keep keys in sync with paths.DATASETS.
_DATASET_LABELS: dict[str, str] = {
    "ind_cn": "IMG_OCR_IND_CN (labelme)",
    "new":    "FUNSD-form (testing + training)",
}


def _quick_load(cat_dir):
    """Cheap load: only parses each sidecar JSON once. Cached by mtime."""
    from .dataset import load_category
    return load_category(cat_dir)


def _image_root_for_category(category: str, filename: str):
    """Find which dataset root contains ``<category>/<filename>``.

    Order:
      1. If the latest run's summary.json stamps a ``dataset`` and that root
         contains the file, use it (avoids guessing when both datasets happen
         to have a ``TRAINING DATA`` category).
      2. Otherwise scan all DATASETS roots and return the first match.
    """
    # 1. honour the most recent run's dataset stamp
    summary = REPORTS_ROOT / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
            ds = (data.get("overall") or {}).get("dataset") or ""
            if ds and ds in DATASETS:
                root = DATASETS[ds]
                if (root / category / filename).exists() or (root / category / "images" / filename).exists():
                    return root
        except (json.JSONDecodeError, OSError):
            pass
    # 2. fall back to scanning every registered dataset
    for root in DATASETS.values():
        if (root / category / filename).exists() or (root / category / "images" / filename).exists():
            return root
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="OCR Benchmark", version="0.1.0")

    # A run with no status update for this long is assumed dead (the per-image
    # OCR timeout is 5 min; give it margin before declaring the lock stale).
    STALE_AFTER_S = 420

    def _parse_iso(ts: str | None) -> datetime | None:
        if not ts:
            return None
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None

    def _is_stale(status: dict) -> bool:
        last = _parse_iso(status.get("updated_at") or status.get("started_at"))
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last).total_seconds() > STALE_AFTER_S

    @app.post("/api/run")
    def api_run(background: BackgroundTasks, category: str | None = None,
                ocr_version: str | None = None, model_type: str | None = None,
                det_box_thresh: float | None = None,
                det_thresh: float | None = None,
                det_unclip_ratio: float | None = None,
                det_limit_side_len: int | None = None,
                use_angle_cls: bool | None = None,
                rec_batch_num: int | None = None,
                rec_img_width: int | None = None,
                enable_preprocessing: bool | None = None,
                iou_threshold: float | None = None,
                enable_symspell_correction: bool | None = None,
                enable_word_segmentation: bool | None = None,
                symspell_max_edit_distance: int | None = None,
                kbbi_top_n: int | None = None,
                dataset: str | None = None,
                force: bool = False):
        if RUN_STATUS_PATH.exists():
            try:
                current = json.loads(RUN_STATUS_PATH.read_text(encoding="utf-8"))
                if current.get("running") and not force and not _is_stale(current):
                    return {"ok": False, "already_running": True}
            except (json.JSONDecodeError, OSError):
                pass
        only = [category] if category else None
        overrides = {
            k: v for k, v in {
                "det_box_thresh": det_box_thresh,
                "det_thresh": det_thresh,
                "det_unclip_ratio": det_unclip_ratio,
                "det_limit_side_len": det_limit_side_len,
                "use_angle_cls": use_angle_cls,
                "rec_batch_num": rec_batch_num,
                "rec_img_width": rec_img_width,
                "enable_preprocessing": enable_preprocessing,
                "iou_threshold": iou_threshold,
                "enable_symspell_correction": enable_symspell_correction,
                "enable_word_segmentation": enable_word_segmentation,
                "symspell_max_edit_distance": symspell_max_edit_distance,
                "kbbi_top_n": kbbi_top_n,
            }.items() if v is not None
        }
        background.add_task(run_benchmark, None, only, False,
                            ocr_version, model_type, overrides, dataset)
        return {"ok": True, "started": True}

    def _read_status() -> dict:
        active_key, _ = resolve_dataset_root()
        idle = {"running": False, "total": 0, "completed": [], "current": None,
                "dataset": active_key}
        if not RUN_STATUS_PATH.exists():
            return idle
        try:
            status = json.loads(RUN_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return idle
        if status.get("running") and _is_stale(status):
            status["stale"] = True  # UI shows a calm "looks stuck" notice + re-run
        return status

    @app.get("/api/progress")
    def api_progress():
        return JSONResponse(_read_status())

    @app.get("/api/progress/stream")
    async def api_progress_stream():
        """Server-Sent Events: push run status as it changes.

        EventSource auto-reconnects on page refresh, so a run that started in
        the background stays visible without the user re-triggering anything.
        ponytail: poll the sidecar file every 1s and emit on change — no pub/sub
        needed for a single-process app; revisit if runs ever go multi-worker.
        """
        async def gen():
            last = None
            try:
                while True:
                    status = _read_status()
                    payload = json.dumps(status)
                    if payload != last:
                        yield f"data: {payload}\n\n"
                        last = payload
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Client disconnected (page refresh / Ctrl+C). Quiet exit; the
                # CancelledError trace otherwise drowns the user's terminal.
                return

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/summary")
    def api_summary():
        path = REPORTS_ROOT / "summary.json"
        if not path.exists():
            raise HTTPException(404, "no reports yet — run POST /api/run or scripts/run_benchmark.py")
        data = json.loads(path.read_text(encoding="utf-8"))
        if "last_run" not in data.get("overall", {}):
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            data.setdefault("overall", {})["last_run"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        return JSONResponse(data)

    @app.get("/api/models")
    def api_models():
        from .config import get_settings
        s = get_settings()
        return {
            "available": AVAILABLE_MODELS,
            "current": {"ocr_version": s.ocr_version, "model_type": s.model_type},
        }

    @app.get("/api/config")
    def api_config():
        from .config import get_settings
        from .corrector import get_corrector
        s = get_settings()
        c = get_corrector()
        active_key, _ = resolve_dataset_root()
        # Lightweight: just expose the active key + available keys. The UI
        # calls /api/datasets for full counts so this stays small on every
        # page load.
        return {
            "dataset": active_key,
            "dataset_keys": [
                {"key": k, "label": _DATASET_LABELS.get(k, k)} for k in DATASETS
            ],
            "enable_symspell_correction": s.enable_symspell_correction,
            "enable_word_segmentation": s.enable_word_segmentation,
            "symspell_max_edit_distance": s.symspell_max_edit_distance,
            "kbbi_top_n": s.kbbi_top_n,
            "kbbi_loaded": c._loaded,
            "kbbi_size": len(c._kbbi_set) if c._kbbi_set else 0,
            "iou_threshold": s.iou_threshold,
            "enable_preprocessing": s.enable_preprocessing,
            "ocr_version": s.ocr_version,
            "model_type": s.model_type,
            "det_box_thresh": s.det_box_thresh,
            "det_thresh": s.det_thresh,
            "det_unclip_ratio": s.det_unclip_ratio,
            "det_limit_side_len": s.det_limit_side_len,
            "use_angle_cls": s.use_angle_cls,
            "rec_batch_num": s.rec_batch_num,
            "rec_img_width": s.rec_img_width,
        }

    def _synth_response(text: str) -> Response:
        """Synthesize -> WAV, with timing metrics on response headers so the
        playground can show the real RTF for exactly what was spoken."""
        text = (text or "").strip()
        if not text:
            raise HTTPException(400, "empty text")
        from .tts_engine import pcm_to_wav
        engine = _get_tts_engine()  # raises 503 if voice missing
        pcm, r = engine.synthesize(text)
        return Response(
            content=pcm_to_wav(pcm, r.sample_rate),
            media_type="audio/wav",
            headers={
                # Exposed to fetch() so the playground reads accurate numbers.
                "Access-Control-Expose-Headers": "X-Synth-Ms,X-Audio-Seconds,X-Rtf,X-Chars,X-First-Chunk-Ms,X-Chars-Per-Sec",
                "X-Synth-Ms": f"{r.synth_ms:.1f}",
                "X-Audio-Seconds": f"{r.audio_seconds:.3f}",
                "X-Rtf": f"{r.rtf:.4f}",
                "X-Chars": str(r.n_chars),
                "X-First-Chunk-Ms": f"{r.first_chunk_ms:.1f}",
                "X-Chars-Per-Sec": f"{r.chars_per_sec:.1f}",
            },
        )

    @app.get("/api/tts")
    def api_tts_get(text: str):
        """Synthesize short ``text`` to WAV (query param). Powers Listen buttons."""
        return _synth_response(text)

    @app.post("/api/tts")
    async def api_tts_post(request: Request):
        """Synthesize long text (JSON body {text}). Used by the playground —
        whole OCR pages exceed a safe GET URL length."""
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001
            raise HTTPException(400, "invalid JSON body")
        return _synth_response((body or {}).get("text", ""))

    @app.post("/api/tts/run")
    def api_tts_run(background: BackgroundTasks, source: str | None = None,
                    force: bool = False):
        """Run the TTS benchmark over the current OCR reports (background)."""
        from .tts_runner import TTS_STATUS_PATH, run as run_tts
        if TTS_STATUS_PATH.exists():
            try:
                current = json.loads(TTS_STATUS_PATH.read_text(encoding="utf-8"))
                if current.get("running") and not force and not _is_stale(current):
                    return {"ok": False, "already_running": True}
            except (json.JSONDecodeError, OSError):
                pass
        background.add_task(run_tts, source)
        return {"ok": True, "started": True}

    @app.get("/api/tts/progress")
    def api_tts_progress():
        from .tts_runner import TTS_STATUS_PATH
        idle = {"running": False, "total": 0, "done": 0, "current": None}
        if not TTS_STATUS_PATH.exists():
            return JSONResponse(idle)
        try:
            status = json.loads(TTS_STATUS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return JSONResponse(idle)
        if status.get("running") and _is_stale(status):
            status["stale"] = True
        return JSONResponse(status)

    @app.get("/api/tts/summary")
    def api_tts_summary():
        path = REPORTS_ROOT / "tts_summary.json"
        if not path.exists():
            raise HTTPException(404, "no TTS reports yet — run POST /api/tts/run")
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/results/{category}")
    def api_results(category: str):
        for f in (REPORTS_ROOT / "per_category").glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("category") == category:
                return JSONResponse(data)
        raise HTTPException(404, f"category not found: {category}")

    @app.get("/api/datasets")
    def api_datasets():
        """List every registered dataset with image/line counts.

        Powers the Datasets page and the Run-panel dataset dropdown.
        """
        from .dataset import list_categories
        active_key, _ = resolve_dataset_root()
        out = []
        for key, root in DATASETS.items():
            cats = list_categories(root)
            pages = []
            for cat in cats:
                pages.extend(_quick_load(cat))
            out.append({
                "key": key,
                "label": _DATASET_LABELS.get(key, key),
                "format": "funsd" if key == "new" else "labelme",
                "root": str(root),
                "n_categories": len(cats),
                "n_images": len(pages),
                "n_lines": sum(len(p.lines) for p in pages),
                "active": key == active_key,
            })
        return JSONResponse({"datasets": out, "active": active_key})

    @app.get("/api/datasets/{key}/categories")
    def api_dataset_categories(key: str):
        """Categories for a specific dataset — populates the Category dropdown."""
        if key not in DATASETS:
            raise HTTPException(404, f"unknown dataset: {key}")
        from .dataset import list_categories
        root = DATASETS[key]
        cats = list_categories(root)
        out = [{"name": "All", "n_images": 0, "n_lines": 0}]
        for cat in cats:
            pages = _quick_load(cat)
            out.append({
                "name": cat.name,
                "n_images": len(pages),
                "n_lines": sum(len(p.lines) for p in pages),
            })
        # "All" gets the totals across this dataset
        out[0]["n_images"] = sum(c["n_images"] for c in out[1:])
        out[0]["n_lines"] = sum(c["n_lines"] for c in out[1:])
        return JSONResponse({"dataset": key, "categories": out})

    @app.get("/api/history")
    def api_history():
        index_path = HISTORY_ROOT / "index.json"
        if not index_path.exists():
            return {"runs": []}
        try:
            runs = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"runs": []}
        # Backfill flat keys from nested config for rows written by code paths
        # that didn't keep the top-level fields (e.g. before the recent saver
        # change). Idempotent — only fills when missing.
        for r in runs:
            cfg = r.get("config") or {}
            if not r.get("ocr_version") and cfg.get("ocr_version"):
                r["ocr_version"] = cfg["ocr_version"]
            if not r.get("model_type") and cfg.get("model_type"):
                r["model_type"] = cfg["model_type"]
        return {"runs": list(reversed(runs))}  # newest first

    @app.get("/api/history/best")
    def api_history_best(metric: str = "f1"):
        """Per-category best run across history. ``metric`` ∈ {f1, cer, wer}.

        For F1 higher is better; for CER/WER lower is better.
        """
        index_path = HISTORY_ROOT / "index.json"
        if not index_path.exists():
            return {"best": {}, "runs": []}
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"best": {}, "runs": []}

        # Load every per-run file once; index alone doesn't carry per-cat metrics.
        runs: list[dict] = []
        for entry in index:
            path = HISTORY_ROOT / f"{entry['id']}.json"
            if not path.exists():
                continue
            try:
                runs.append(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                continue

        # Best per-category by selected metric.
        # ponytail: O(runs × cats) is fine; runs is capped at 50 by the saver.
        minimize = metric in ("cer", "wer")
        best: dict[str, dict] = {}
        for run in runs:
            for cat in run.get("per_category", []):
                name = cat["category"]
                score = cat.get(metric)
                if score is None:
                    continue
                cur = best.get(name)
                if cur is None or (minimize and score < cur["score"]) or (not minimize and score > cur["score"]):
                    best[name] = {
                        "category": name,
                        "score": score,
                        "metric": metric,
                        "run_id": run["id"],
                        "timestamp": run["timestamp"],
                        "config": run.get("config", {}),
                    }
        return {"best": best, "metric": metric, "n_runs": len(runs)}

    @app.get("/api/history/{run_id}")
    def api_history_detail(run_id: str):
        safe = "".join(c if c.isalnum() or c in "_-" else "" for c in run_id)
        path = HISTORY_ROOT / f"{safe}.json"
        if not path.exists():
            raise HTTPException(404, f"run not found: {run_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Backfill: per-run files written without top-level ocr_version/model_type.
        cfg = data.get("config") or {}
        ov = data.get("overall") or {}
        if not data.get("ocr_version"):
            data["ocr_version"] = cfg.get("ocr_version") or ov.get("ocr_version") or ""
        if not data.get("model_type"):
            data["model_type"] = cfg.get("model_type") or ov.get("model_type") or ""
        # Dataset key is in cfg.overall for newer runs but might be missing in
        # older snapshots; leave it blank rather than guess.
        if not data.get("dataset"):
            ds = cfg.get("dataset") or ov.get("dataset") or ""
            if ds:
                data["dataset"] = ds
        return JSONResponse(data)

    @app.post("/api/history/backfill")
    def api_history_backfill():
        """One-shot migration: rewrite every history file so top-level
        ``ocr_version``, ``model_type``, and ``dataset`` are present.

        Older snapshots from before the saver change only stored these inside
        ``config``/``overall``; the UI reads top-level so the table would show
        ``? · ?``. Idempotent — only patches missing fields, never overwrites.
        """
        if not HISTORY_ROOT.exists():
            return {"patched": 0, "scanned": 0}
        patched = 0
        scanned = 0
        for path in HISTORY_ROOT.glob("*.json"):
            if path.name == "index.json":
                continue
            scanned += 1
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            cfg = data.get("config") or {}
            ov = data.get("overall") or {}
            changed = False
            for k in ("ocr_version", "model_type", "dataset"):
                if not data.get(k):
                    src = cfg.get(k) or ov.get(k) or ""
                    if src:
                        data[k] = src
                        changed = True
            if changed:
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
                patched += 1
        return {"patched": patched, "scanned": scanned}

    @app.get("/api/image/{category}/{filename}")
    def api_image(category: str, filename: str):
        # category is the dir name; resolve safely.
        # Pick the root by reading the latest summary.json's dataset key — that
        # way the URL works whether the user ran ind_cn or new.
        safe_cat = "".join(c if c.isalnum() or c in " _-" else "_" for c in category)
        safe_name = Path(filename).name  # strip any path traversal
        root = _image_root_for_category(safe_cat, safe_name)
        if root is None:
            raise HTTPException(404, f"image not found: {category}/{filename}")
        path = root / safe_cat / safe_name
        if not path.exists():
            # FUNSD images live in <root>/<cat>/images/<name>, not <root>/<cat>/<name>
            alt = root / safe_cat / "images" / safe_name
            if alt.exists():
                path = alt
            else:
                raise HTTPException(404, f"image not found: {category}/{filename}")
        suffix = path.suffix.lower()
        media = "image/png" if suffix == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media)

    @app.get("/")
    def index():
        idx = UI_ROOT / "index.html"
        if not idx.exists():
            raise HTTPException(404, "ui/index.html not found")
        return FileResponse(idx, media_type="text/html")

    @app.get("/datasets.html")
    def datasets_page():
        page = UI_ROOT / "datasets.html"
        if not page.exists():
            raise HTTPException(404, "ui/datasets.html not found")
        return FileResponse(page, media_type="text/html")

    @app.get("/tts")
    def tts_page():
        page = UI_ROOT / "tts.html"
        if not page.exists():
            raise HTTPException(404, "ui/tts.html not found")
        return FileResponse(page, media_type="text/html")

    if UI_ROOT.exists():
        # ponytail: disable caching so JS/CSS edits land without a hard refresh.
        # Default Starlette StaticFiles sends Cache-Control: max-age=3600 which
        # makes a 1-hour window where old UI code keeps running.
        class _NoCacheStaticFiles(StaticFiles):
            def file_response(self, *args, **kwargs):
                resp = super().file_response(*args, **kwargs)
                resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                resp.headers["Pragma"] = "no-cache"
                resp.headers["Expires"] = "0"
                return resp
        app.mount("/static", _NoCacheStaticFiles(directory=str(UI_ROOT)), name="static")

    return app


app = create_app()


def serve() -> None:
    """Entry point for ``ocr-bench-serve`` console script."""
    import logging
    import uvicorn
    from .config import get_settings

    # Configure logging so benchmark progress appears in journalctl
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    logging.getLogger("ocr_bench").setLevel(logging.INFO)

    s = get_settings()
    uvicorn.run(app, host=s.serve_host, port=s.serve_port, log_level="warning")


if __name__ == "__main__":  # ponytail: self-check
    serve()