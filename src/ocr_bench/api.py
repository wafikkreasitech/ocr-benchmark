"""FastAPI: serves dashboard + JSON endpoints.

Endpoints:
  POST /api/run                 run benchmark (background; poll /api/progress)
  GET  /api/progress            live run status (running, completed, current)
  GET  /api/summary             aggregated metrics (overall + per-category)
  GET  /api/results/<category>  per-image detail incl. overlay data
  GET  /api/image/<cat>/<file>  raw image bytes
  GET  /api/models              available OCR model combinations
  GET  /api/config              current runtime config
  GET  /                        dashboard (ui/index.html)
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .paths import DEFAULT_DATASET_ROOT, HISTORY_ROOT, REPORTS_ROOT, UI_ROOT
from .runner import RUN_STATUS_PATH, run as run_benchmark

AVAILABLE_MODELS = {
    "PP-OCRv6": {"model_types": ["tiny", "small", "medium"], "default": "small",
                 "desc": "Latest, best accuracy"},
    "PP-OCRv5": {"model_types": ["mobile", "server"], "default": "mobile",
                 "desc": "Per-language rec models"},
    "PP-OCRv4": {"model_types": ["mobile", "server"], "default": "mobile",
                 "desc": "Legacy, stable"},
}


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
            }.items() if v is not None
        }
        background.add_task(run_benchmark, None, only, False,
                            ocr_version, model_type, overrides)
        return {"ok": True, "started": True}

    def _read_status() -> dict:
        idle = {"running": False, "total": 0, "completed": [], "current": None}
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
        return {
            "enable_symspell_correction": s.enable_symspell_correction,
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

    @app.get("/api/results/{category}")
    def api_results(category: str):
        for f in (REPORTS_ROOT / "per_category").glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("category") == category:
                return JSONResponse(data)
        raise HTTPException(404, f"category not found: {category}")

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
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/image/{category}/{filename}")
    def api_image(category: str, filename: str):
        # category is the dir name; resolve safely
        safe_cat = "".join(c if c.isalnum() or c in " _-" else "_" for c in category)
        safe_name = Path(filename).name  # strip any path traversal
        path = DEFAULT_DATASET_ROOT / safe_cat / safe_name
        if not path.exists():
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