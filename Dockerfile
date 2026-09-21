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

FROM python:3.12-slim-bookworm AS service
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
  CMD python -c "import os,urllib.request,sys; p=os.environ.get('PORT','8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=3).status == 200 else 1)"

# Cloud Run bơm PORT vào; Fly và Render cũng vậy. exec để uvicorn thành PID 1 và
# nhận được SIGTERM khi nền tảng thu hồi container. WEB_CONCURRENCY=1 trên Cloud Run:
# nền tảng tự scale theo instance, 1 worker là 134MiB thay vì 295MiB.
CMD ["sh", "-c", "exec uvicorn seatecco_rag.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-2}"]


# Bản TỰ CHỨA cho nền tảng không có đĩa bền (Cloud Run): nướng luôn index vào image,
# nên container không cần volume nào. Phải chạy ingest TRƯỚC khi build target này:
#
#   python -m seatecco_rag.ingest.index
#   docker build --target selfcontained -t <image> .
#
# Cập nhật tài liệu = dựng lại index, build lại image, deploy lại.
FROM service AS selfcontained
COPY --chown=seatecco var/index ./var/index
