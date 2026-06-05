from __future__ import annotations

from typing_extensions import TypedDict

from .evidence import EvidencePack
from .retriever import SearchResult


class RAGState(TypedDict):
    # inputs
    query:                str
    regulations:          list[str] | None   # retrieval filter, e.g. ["GDPR", "AI Act"]
    required_regulations: list[str] | None   # gap-check list for EvidencePack
    # intermediate
    candidates:           list[SearchResult]
    reranked:             list[SearchResult]
    # output
    evidence_pack:        EvidencePack | None
