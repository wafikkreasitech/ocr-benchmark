"""Download the Piper voice ai4db uses for Indonesian TTS.

Fetches id_ID-news_tts-medium (.onnx + .onnx.json) into
models/piper-voices/id/ — the same relative path ai4db's config expects
(config.py LANGUAGE_PROFILES["id"]). Piper needs BOTH files: the .onnx.json
carries sample rate + phoneme map.

Usage:
    uv run python -m scripts.download_voice

ponytail: stdlib urllib, no huggingface_hub dep for two file downloads.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

# rhasspy/piper-voices HF layout: <lang>/<locale>/<name>/<quality>/<file>
_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium"
_FILES = ["id_ID-news_tts-medium.onnx", "id_ID-news_tts-medium.onnx.json"]

# DEST = directory the .onnx + .onnx.json land in. Resolve in this order:
#   1. PIPER_VOICE_DIR env var (explicit override, useful in Docker / CI)
#   2. <repo>/models/piper-voices/id  (default, matches docker-compose volume)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = Path(os.environ.get("PIPER_VOICE_DIR") or _REPO_ROOT / "models" / "piper-voices" / "id")

_CHUNK = 64 * 1024  # refresh progress every 64 KiB
_BAR_W = 30  # bar width in chars


def _human(n: int) -> str:
    """1024-based: 12.3 MB, 1.4 KB, etc."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def _print_progress(name: str, done: int, total: int) -> None:
    if total > 0:
        pct = min(100, done * 100 // total)
        filled = pct * _BAR_W // 100
        bar = "█" * filled + "░" * (_BAR_W - filled)
        line = f"\r    [{bar}] {pct:3d}%  {_human(done)} / {_human(total)}  {name}"
    else:
        # server didn't send Content-Length — show bytes only
        line = f"\r    {_human(done)}  {name}"
    print(line, end="", flush=True)


def _fetch(url: str, target: Path, name: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "ocr-benchmark/download_voice"})
    with urllib.request.urlopen(req) as resp, target.open("wb") as out:
        total = int(resp.headers.get("Content-Length", "0") or 0)
        # surface the resolved URL + size so users on slow links see progress
        # isn't frozen on the redirect
        if total > 0:
            print(f"    → {_human(total)}  ({resp.url})", flush=True)
        else:
            print(f"    → size unknown, streaming  ({resp.url})", flush=True)
        done = 0
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            _print_progress(name, done, total)
    size = target.stat().st_size
    print(f"\r  [ok] {name} ({size:,} bytes)" + " " * 8)


def main() -> int:
    try:
        DEST.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        print(f"x cannot create {DEST}: {e}", file=sys.stderr)
        print(f"  hint: sudo chown -R $USER {DEST.parent}  "
              f"or set PIPER_VOICE_DIR=/writable/path", file=sys.stderr)
        return 2
    for name in _FILES:
        target = DEST / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  [ok] {name} (already present, {target.stat().st_size:,} bytes)")
            continue
        url = f"{_BASE}/{name}"
        print(f"  downloading {name}")
        try:
            _fetch(url, target, name)
        except Exception as e:  # noqa: BLE001 — surface any network/HTTP failure plainly
            if target.exists():
                target.unlink()
            print(f"\r  x failed: {name} - {e}" + " " * 8, file=sys.stderr)
            return 1
    print(f"\nVoice ready: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
