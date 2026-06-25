# OCR Benchmark — common workflows
# Usage: `make <target>` (Windows: use `make` from Git Bash, or run via uv)

.PHONY: help install sync run serve test smoke clean reset all

help:
	@echo "OCR Benchmark · make targets:"
	@echo "  make install   - uv sync (install deps)"
	@echo "  make run       - run full benchmark (writes reports/)"
	@echo "  make serve     - start dashboard on http://127.0.0.1:8765"
	@echo "  make test      - run smoke tests"
	@echo "  make clean     - delete reports/"
	@echo "  make reset     - clean + re-install"
	@echo "  make all       - install + run + serve"

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