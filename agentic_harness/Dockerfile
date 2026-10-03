FROM python:3.13-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY pyproject.toml README.md ./
COPY agentic_harness ./agentic_harness

RUN pip install --no-cache-dir --prefix=/install ".[ibkr]"

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=builder /install /usr/local
COPY agentic_harness ./agentic_harness

CMD ["python", "-m", "agentic_harness.ibkr_oauth"]
