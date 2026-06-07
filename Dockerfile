FROM python:3.13-slim

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package metadata + README (hatchling needs README at build time)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install all Python dependencies
# --extra-index-url ensures torch resolves to the CPU-only wheel (~200MB vs 2GB CUDA)
RUN pip install -e . \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

# Copy remaining application files
COPY ui/ ./ui/
COPY .streamlit/ ./.streamlit/
COPY data/vectorstore/ ./data/vectorstore/
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# HuggingFace cache dir + ownership
RUN mkdir -p /home/appuser/.cache && chown -R appuser:appuser /home/appuser /app

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/.cache/huggingface
ENV OLLAMA_HOST=http://ollama:11434

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
