"""FastAPI: serves dashboard + JSON endpoints.

Endpoints:
  POST /api/run                 run benchmark (writes reports)
  GET  /api/summary             aggregated metrics (overall + per-category)
  GET  /api/results/<category>  per-image detail incl. overlay data
  GET  /api/image/<cat>/<file>  raw image bytes
  GET  /                        dashboard (ui/index.html)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .paths import DEFAULT_DATASET_ROOT, REPORTS_ROOT, UI_ROOT
from .runner import run as run_benchmark


def create_app() -> FastAPI:
    app = FastAPI(title="OCR Benchmark", version="0.1.0")

    @app.post("/api/run")
    def api_run(category: str | None = None):
        only = [category] if category else None
        run_benchmark(root=None, only_categories=only, verbose=False)
        return {"ok": True}

    @app.get("/api/summary")
    def api_summary():
        path = REPORTS_ROOT / "summary.json"
        if not path.exists():
            raise HTTPException(404, "no reports yet — run POST /api/run or scripts/run_benchmark.py")
        data = json.loads(path.read_text(encoding="utf-8"))
        # Backfill last_run from file mtime if missing (older reports)
        if "last_run" not in data.get("overall", {}):
            ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            data.setdefault("overall", {})["last_run"] = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        return JSONResponse(data)

    @app.get("/api/config")
    def api_config():
        """Expose current runtime config (feature flag state) to the dashboard."""
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
        }

    @app.get("/api/results/{category}")
    def api_results(category: str):
        for f in (REPORTS_ROOT / "per_category").glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("category") == category:
                return JSONResponse(data)
        raise HTTPException(404, f"category not found: {category}")

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
        app.mount("/static", StaticFiles(directory=str(UI_ROOT)), name="static")

    return app


app = create_app()


def serve() -> None:
    """Entry point for ``ocr-bench-serve`` console script."""
    import uvicorn
    from .config import get_settings
    s = get_settings()
    uvicorn.run(app, host=s.serve_host, port=s.serve_port, log_level="warning")


if __name__ == "__main__":  # ponytail: self-check
    serve()