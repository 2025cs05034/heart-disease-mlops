# syntax=docker/dockerfile:1.7

# ============================================================================
# Stage 1 - builder: install dependencies into a virtual environment
# ============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements.txt

# ============================================================================
# Stage 2 - runtime: minimal image, non-root user, copy venv + app
# ============================================================================
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/best_model.pkl \
    SERVICE_NAME=heart-disease-api \
    SERVICE_VERSION=1.0.0 \
    LOG_LEVEL=INFO

# Create non-root user
RUN groupadd --system app && useradd --system --gid app --uid 1001 --home /app app

WORKDIR /app

# Copy installed virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy source + trained model
COPY --chown=app:app src/ ./src/
COPY --chown=app:app models/ ./models/

USER 1001

EXPOSE 8000

# Healthcheck hits /health (cheap, no model load required)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
