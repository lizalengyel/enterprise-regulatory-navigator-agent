from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .retriever import SearchResult


# Data models

class Evidence(BaseModel):
    """A single retrieved evidence item used by the agent."""

    text:       str
    source:     str                         # doc_name,  e.g. "gdpr"
    regulation: str | None = None           # document_type, e.g. "GDPR"
    title:      str | None = None           # citation string, e.g. "GDPR · Article 17 · p. 12"
    page:       int | None = None           # 1-indexed page number
    section:    str | None = None           # article heading, e.g. "Article 17"
    score:      float | None = None         # cross-encoder relevance score
    metadata:   dict[str, Any] = Field(default_factory=dict)


class EvidencePack(BaseModel):
    """Structured evidence returned by the RAG subgraph."""

    query:                str
    evidence:             list[Evidence]
    evidence_score:       float
    regulations_covered:  list[str]
    source_count:         int
    sufficient:           bool
    gaps:                 list[str] = Field(default_factory=list)


# Conversion from our SearchResult

def make_citation(result: SearchResult) -> str:
    parts = [result.document_type]
    if result.article:
        parts.append(result.article)
    parts.append(f"p. {result.page_start + 1}")
    return " · ".join(parts)


def search_result_to_evidence(result: SearchResult) -> Evidence:
    """Convert a SearchResult from the retriever/reranker into an Evidence item."""
    return Evidence(
        text=result.text,
        source=result.doc_name,
        regulation=result.document_type,
        title=make_citation(result),
        page=result.page_start + 1,
        section=result.article,
        score=result.score,
        metadata={
            "doc_name":      result.doc_name,
            "document_type": result.document_type,
            "article":       result.article,
            "page_start":    result.page_start,
            "page_end":      result.page_end,
            "chunk_index":   result.chunk_index,
        },
    )


# Deduplication

def deduplicate_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Remove duplicate chunks based on source, page, and text prefix."""
    seen: set[tuple[str, int | None, str]] = set()
    unique: list[Evidence] = []
    for item in evidence:
        key = (item.source, item.page, item.text[:200])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


# Scoring

def average_relevance(evidence: list[Evidence]) -> float:
    scores = [item.score for item in evidence if item.score is not None]
    if not scores:
        return 0.6
    return sum(scores) / len(scores)


def source_diversity_score(evidence: list[Evidence]) -> float:
    unique_sources = len({item.source for item in evidence})
    if unique_sources >= 3:
        return 1.0
    if unique_sources == 2:
        return 0.75
    return 0.5


def regulation_coverage_score(evidence: list[Evidence]) -> float:
    unique_regulations = {item.regulation for item in evidence if item.regulation}
    n = len(unique_regulations)
    if n >= 3:
        return 1.0
    if n == 2:
        return 0.75
    if n == 1:
        return 0.5
    return 0.0


def calculate_evidence_score(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    score = (
        0.5  * average_relevance(evidence)
        + 0.25 * source_diversity_score(evidence)
        + 0.25 * regulation_coverage_score(evidence)
    )
    return round(min(score, 1.0), 3)


# Gap analysis

def identify_evidence_gaps(
    evidence: list[Evidence],
    required_regulations: list[str],
) -> list[str]:
    """Identify missing regulation coverage."""
    covered = {item.regulation for item in evidence if item.regulation}
    gaps: list[str] = []
    for regulation in required_regulations:
        if regulation not in covered:
            gaps.append(f"No evidence found for required regulation: {regulation}")
    if not evidence:
        gaps.append("No relevant evidence was retrieved.")
    return gaps


# Main builder

def build_evidence_pack(
    query: str,
    results: list[SearchResult],
    required_regulations: list[str] | None = None,
    sufficiency_threshold: float = 0.65,
) -> EvidencePack:
    """Build an EvidencePack from reranked SearchResults."""
    evidence = [search_result_to_evidence(r) for r in results]
    evidence = deduplicate_evidence(evidence)

    evidence_score = calculate_evidence_score(evidence)
    regulations_covered = sorted({e.regulation for e in evidence if e.regulation})
    gaps = identify_evidence_gaps(evidence, required_regulations or [])
    sufficient = evidence_score >= sufficiency_threshold and not gaps

    return EvidencePack(
        query=query,
        evidence=evidence,
        evidence_score=evidence_score,
        regulations_covered=regulations_covered,
        source_count=len({e.source for e in evidence}),
        sufficient=sufficient,
        gaps=gaps,
    )


# Prompt formatting

def format_context(pack: EvidencePack) -> str:
    """Numbered context block ready to inject into an LLM prompt."""
    blocks = [
        f"[{i + 1}] {e.title}\n{e.text.strip()}"
        for i, e in enumerate(pack.evidence)
    ]
    return "\n\n".join(blocks)


def format_citations(pack: EvidencePack) -> str:
    """Compact citation list for appending to an agent response."""
    lines = ["Sources:"] + [
        f"[{i + 1}] {e.title}" for i, e in enumerate(pack.evidence)
    ]
    return "\n".join(lines)
