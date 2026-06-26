# ─── Stage 1: builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install project
COPY . .
RUN uv sync --frozen --no-dev

# ─── Stage 2: runtime ───────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN groupadd -r bench && useradd -r -g bench -m bench

WORKDIR /app

# Copy only what's needed from builder
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/src /app/src
COPY --from=builder /build/ui /app/ui
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

# Dataset and reports will be mounted as volumes
RUN mkdir -p /app/reports /app/IMG_OCR_IND_CN && chown -R bench:bench /app

USER bench

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/api/config')" || exit 1

ENTRYPOINT ["python", "-m", "uvicorn", "ocr_bench.api:app", "--host", "0.0.0.0", "--port", "8765"]
