from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from .embeddings import embed_query

logger = logging.getLogger(__name__)

CHROMA_DIR    = Path("data/vectorstore")
COLLECTION    = "regulatory_docs"
DEFAULT_TOP_K = 20

# Normalise agent-friendly aliases to the canonical document_type values
# stored in ChromaDB metadata (set during ingestion in chunking.py).
_REGULATION_ALIASES: dict[str, str] = {
    "AI_ACT":  "AI Act",
    "AI ACT":  "AI Act",
    "AIACT":   "AI Act",
    "GDPR":    "GDPR",
    "DORA":    "DORA",
    "NIS2":    "NIS2",
    "NIS_2":   "NIS2",
}


# Data model

@dataclass
class SearchResult:
    text:          str
    doc_name:      str
    document_type: str
    article:       str | None
    page_start:    int
    page_end:      int
    chunk_index:   int
    score:         float   # cosine similarity — higher is more relevant


# Collection (lazy singleton)

_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION)
    return _collection


# Filter builder

def normalise(regulation: str) -> str:
    return _REGULATION_ALIASES.get(regulation.upper().strip(), regulation.strip())


def build_where(regulations: list[str] | None) -> dict | None:
    if not regulations:
        return None
    canonical = [normalise(r) for r in regulations]
    if len(canonical) == 1:
        return {"document_type": canonical[0]}
    return {"document_type": {"$in": canonical}}


# Main entry point

def retrieve(
    query: str,
    regulations: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> list[SearchResult]:
    """
    Semantic search over the regulatory vector store.

    Args:
        query:       Natural-language question or statement.
        regulations: Optional allowlist of document types to search within,
                     e.g. ["GDPR", "AI Act"]. None searches all documents.
        top_k:       Number of results to return.

    Returns:
        List of SearchResult ordered by descending cosine similarity.
    """
    collection = get_collection()
    query_vector = embed_query(query)
    where = build_where(regulations)

    kwargs: dict = dict(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where

    raw = collection.query(**kwargs)

    results: list[SearchResult] = []
    for text, meta, distance in zip(
        raw["documents"][0],
        raw["metadatas"][0],
        raw["distances"][0],
    ):
        results.append(SearchResult(
            text=text,
            doc_name=meta["doc_name"],
            document_type=meta["document_type"],
            article=meta["article"] or None,
            page_start=meta["page_start"],
            page_end=meta["page_end"],
            chunk_index=meta["chunk_index"],
            score=1.0 - distance,   # cosine distance → similarity
        ))

    logger.debug(
        "retrieve: query=%r regulations=%s → %d results",
        query[:60], regulations, len(results),
    )
    return results
