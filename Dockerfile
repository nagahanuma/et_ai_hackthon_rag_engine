# Multi-stage production container for GCP Cloud Run
FROM python:3.11-slim

# System setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/model_cache \
    PORT=8080

WORKDIR /app

# Install minimal OS dependencies for psycopg2 and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to avoid heavy CUDA runtime dependencies (saves ~2GB image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache BAAI/bge-m3 embedding weights during build on CPU
# This prevents cold-start latency timeouts on GCP Cloud Run
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3', device='cpu')"

# Copy application source code and frontend assets
COPY src/ ./src/
COPY frontend/ ./frontend/

# Expose dynamic port (Cloud Run sets PORT=8080 by default)
EXPOSE 8080

# Health check to ensure service readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/v1/health || exit 1

# Start Uvicorn ASGI server with dynamic port support and 2 multi-process workers for GIL-free parallel execution
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2"]
