from __future__ import annotations

import logging
import threading

from sentence_transformers import CrossEncoder

from .retriever import SearchResult

logger = logging.getLogger(__name__)

RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_TOP_N  = 5

_reranker: CrossEncoder | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                logger.info("Loading reranker model %s ...", RERANKER_MODEL)
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    results: list[SearchResult],
    top_n: int = DEFAULT_TOP_N,
) -> list[SearchResult]:
    """
    Rerank retrieval results using a cross-encoder.

    Args:
        query:   The original query string.
        results: Candidate SearchResults from the bi-encoder retriever (top 20).
        top_n:   Number of results to return after reranking.

    Returns:
        Top-n SearchResults ordered by cross-encoder score descending.
    """
    if not results:
        return []

    reranker = get_reranker()
    pairs = [(query, r.text) for r in results]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True,
    )

    top = [r for _, r in ranked[:top_n]]

    logger.debug(
        "rerank: %d candidates → top %d | scores %.3f–%.3f",
        len(results), len(top), ranked[0][0], ranked[-1][0],
    )
    return top
