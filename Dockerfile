# ─── Stage 1: builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# Remove full opencv + reinstall headless only (headless needs no libGL)
RUN uv pip uninstall opencv-python opencv-python-headless; \
    uv pip install --no-cache --reinstall opencv-python-headless==4.13.0.92

# Clean caches
RUN find .venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
    find .venv -name "*.pyc" -delete 2>/dev/null; \
    rm -rf .venv/lib/python*/site-packages/*/tests 2>/dev/null; \
    true

# ─── Stage 2: runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd -r bench && useradd -r -g bench -m bench

WORKDIR /app

COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/src /app/src
COPY --from=builder /build/ui /app/ui
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

RUN mkdir -p /app/reports /app/IMG_OCR_IND_CN && chown -R bench:bench /app

USER bench

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/api/config')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "ocr_bench.api:app", "--host", "0.0.0.0", "--port", "8765"]
