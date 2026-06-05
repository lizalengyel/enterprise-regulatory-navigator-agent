from __future__ import annotations

import logging
import threading

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64
DEVICE     = "cpu"   # swap to "cuda" or "mps" if available

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                logger.info("Loading embedding model %s ...", MODEL_NAME)
                _model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of passage texts. Returns L2-normalised vectors."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query string. BGE v1.5 uses no instruction prefix."""
    return embed_texts([query])[0]
