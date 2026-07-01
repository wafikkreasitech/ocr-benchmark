# OCR Benchmark — common workflows
# Usage: `make <target>` (Windows: use `make` from Git Bash, or run via uv)

.PHONY: help install sync run serve test smoke clean reset all \
        voice tts-build tts-up tts-down tts-logs tts-restart tts-ps tts-voice \
        tts-bench tts-summary tts-shell

help:
	@echo "OCR Benchmark · make targets:"
	@echo ""
	@echo "  Local (host, uv):"
	@echo "    make install      - uv sync (install deps)"
	@echo "    make run          - run OCR benchmark (writes reports/)"
	@echo "    make serve        - start dashboard on http://127.0.0.1:8765"
	@echo "    make test         - run smoke tests"
	@echo "    make clean        - delete reports/"
	@echo "    make reset        - clean + re-install"
	@echo ""
	@echo "  TTS (host):"
	@echo "    make voice        - download Piper voice into models/piper-voices/id"
	@echo "    make tts-bench    - run TTS benchmark (writes reports/tts_summary.json)"
	@echo ""
	@echo "  Docker (daemon, 'make tts-up' = up and running in background):"
	@echo "    make tts-build    - build the docker image"
	@echo "    make tts-up       - build (if needed) + start daemon in background"
	@echo "    make tts-down     - stop daemon"
	@echo "    make tts-restart  - restart daemon"
	@echo "    make tts-logs     - tail daemon logs (Ctrl-C to exit)"
	@echo "    make tts-ps       - show container status"
	@echo "    make tts-voice    - download Piper voice into ./models (host path)"
	@echo "    make tts-shell    - open a shell in the running container"

install sync:
	uv sync

run:
	uv run python scripts/run_benchmark.py

serve:
	uv run ocr-bench-serve

test smoke:
	uv sync --extra dev
	.venv/Scripts/python.exe -m pytest tests/ -v  # Windows; use .venv/bin/python on Unix

clean:
	rm -rf reports/per_category reports/summary.csv reports/summary.json

reset: clean
	uv sync --reinstall

all: install run serve

# ── TTS (host) ───────────────────────────────────────────────
# Download Piper Indonesian voice into ./models/piper-voices/id/.
# PIPER_VOICE_DIR override lets you land it elsewhere (Docker build context, etc).
voice:
	uv run python -m scripts.download_voice

# Run TTS benchmark on the host. Reads reports/per_category/*.json first,
# so make sure `make run` (OCR) has completed at least once.
tts-bench:
	uv run python -m ocr_bench.tts_runner

tts-summary:
	@cat reports/tts_summary.json 2>/dev/null | head -40 || echo "no tts_summary.json yet — run 'make tts-bench'"

# ── Docker daemon ─────────────────────────────────────────────
# The image's ENTRYPOINT is the API server, so `docker compose run` with a
# different command needs --entrypoint override. We never use `run` here —
# everything goes through `up -d` so the daemon stays up.

COMPOSE := docker compose -f docker-compose.yml

tts-build:
	$(COMPOSE) build

# Start the daemon in the background. Idempotent: safe to re-run.
tts-up: tts-voice tts-build
	$(COMPOSE) up -d
	@echo "OCR-Benchmark running at http://localhost:8766"
	@echo "TTS dashboard:           http://localhost:8766/tts"
	@docker compose -f docker-compose.yml ps

# Same as tts-up but skips the voice download (faster when voice already cached).
tts-up-fast: tts-build
	$(COMPOSE) up -d
	@echo "OCR-Benchmark running at http://localhost:8766"
	@docker compose -f docker-compose.yml ps

tts-down:
	$(COMPOSE) down

tts-restart: tts-down tts-up

tts-logs:
	$(COMPOSE) logs -f --tail=100

tts-ps:
	$(COMPOSE) ps

# Run download_voice.py ON THE HOST into ./models. The volume in
# docker-compose.yml mounts ./models into the container as /app/models:ro,
# so downloading here is enough — the daemon picks it up after restart.
tts-voice:
	@mkdir -p models/piper-voices/id
	uv run python -m scripts.download_voice

# Drop into a shell inside the running container (user `bench`).
tts-shell:
	$(COMPOSE) exec ocr-bench /bin/bash || $(COMPOSE) exec ocr-bench /bin/sh