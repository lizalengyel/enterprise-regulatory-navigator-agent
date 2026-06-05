FROM python:3.13-slim

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv --no-cache-dir

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ ./src/
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/

# Copy pre-built vectorstore (22MB — avoids re-ingestion at startup)
COPY data/vectorstore/ ./data/vectorstore/

# Copy entrypoint
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# HuggingFace model cache volume mount point
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV OLLAMA_HOST=http://ollama:11434

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
