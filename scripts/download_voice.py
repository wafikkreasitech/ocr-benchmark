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


def _report(done: int, block: int, total: int) -> None:
    if total > 0:
        pct = min(100, done * block * 100 // total)
        print(f"\r    {pct:3d}%", end="", flush=True)


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
            print(f"  [ok] {name} (already present)")
            continue
        url = f"{_BASE}/{name}"
        print(f"  downloading {name}")
        try:
            urllib.request.urlretrieve(url, target, _report)
            print(f"\r  [ok] {name} ({target.stat().st_size:,} bytes)")
        except Exception as e:  # noqa: BLE001 — surface any network/HTTP failure plainly
            if target.exists():
                target.unlink()
            print(f"\r  x failed: {name} - {e}", file=sys.stderr)
            return 1
    print(f"\nVoice ready: {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
