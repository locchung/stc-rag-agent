# syntax=docker/dockerfile:1
# Hai tầng: tầng đầu dựng .venv bằng uv, tầng sau chỉ chứa runtime.
# Index KHÔNG nướng vào image được vì embed cần Ollama - nó nằm trong volume var/
# và do service "ingest" dựng (xem docker-compose.yml).

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# --no-editable: cài bản wheel thật, nên package nằm trong site-packages.
# Vì vậy tầng dưới PHẢI đặt SEATECCO_ROOT, không suy ra gốc repo từ __file__ được.
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    SEATECCO_ROOT=/app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY data ./data

RUN useradd --create-home --uid 10001 seatecco && mkdir -p /app/var && chown -R seatecco /app/var
USER seatecco

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "seatecco_rag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
