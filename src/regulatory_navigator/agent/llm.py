from __future__ import annotations

import os

LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:1.7b")


def get_llm(temperature: float = 0.0):
    """
    Return a LangChain ChatOllama instance.

    Override the model via env var:
        LLM_MODEL=qwen3:8b uv run python app.py
    """
    from langchain_ollama import ChatOllama
    return ChatOllama(model=LLM_MODEL, temperature=temperature)
