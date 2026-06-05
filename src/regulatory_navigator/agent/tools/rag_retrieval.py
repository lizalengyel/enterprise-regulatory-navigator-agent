from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ...rag.subgraph import run_rag


@tool
def rag_retrieval_tool(
    query: str,
    regulations: list[str] | None = None,
) -> dict[str, Any]:
    """
    Retrieve relevant regulatory evidence from the vector store using RAG.
    Searches GDPR, DORA, NIS2, and AI Act regulatory documents.
    Uses semantic search (bi-encoder) followed by cross-encoder reranking.

    Args:
        query:       Natural-language question about regulatory obligations.
        regulations: Optional list of regulations to filter by,
                     e.g. ["GDPR", "DORA"]. None searches all documents.

    Returns:
        Dictionary with evidence items, score, covered regulations, and gaps.
    """
    pack = run_rag(
        query=query,
        regulations=regulations,
        required_regulations=regulations,
    )

    return {
        "evidence": [
            {
                "text":       e.text,
                "source":     e.source,
                "regulation": e.regulation,
                "title":      e.title,
                "page":       e.page,
                "section":    e.section,
                "score":      e.score,
            }
            for e in pack.evidence
        ],
        "evidence_score":       pack.evidence_score,
        "sufficient":           pack.sufficient,
        "regulations_covered":  pack.regulations_covered,
        "gaps":                 pack.gaps,
    }
