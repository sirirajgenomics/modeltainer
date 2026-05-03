# =============================================================================
# ModelTainer API Gateway — Dockerfile
# =============================================================================
# Multi-stage build: builder installs deps, final image is minimal.
# =============================================================================

FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------------------------------------------------------------------------
FROM python:3.12-slim AS final

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser /app
USER appuser

# Models config is volume-mounted at runtime via compose.yaml
ENV MODELS_CONFIG=/config/models.yaml

EXPOSE 8080
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080"]
