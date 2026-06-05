#!/bin/bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
MODEL="${LLM_MODEL:-qwen3:1.7b}"

echo "[entrypoint] Waiting for Ollama at ${OLLAMA_HOST} ..."
until curl -sf "${OLLAMA_HOST}/api/version" > /dev/null 2>&1; do
    sleep 3
done
echo "[entrypoint] Ollama is ready."

echo "[entrypoint] Pulling model ${MODEL} (skipped if already cached) ..."
curl -sf "${OLLAMA_HOST}/api/pull" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"${MODEL}\"}" | tail -1
echo ""
echo "[entrypoint] Model ready."

echo "[entrypoint] Starting Streamlit ..."
exec uv run streamlit run ui/app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none
