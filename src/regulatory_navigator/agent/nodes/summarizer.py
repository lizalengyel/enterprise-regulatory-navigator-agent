from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..prompts import SUMMARIZER_SYSTEM
from ..state import AgentState
from ._constants import EVIDENCE_SUFFICIENCY_THRESHOLD

logger = logging.getLogger(__name__)


def recompute_evidence_score(evidence: list[dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    scores             = [e["score"] for e in evidence if e.get("score") is not None]
    avg_relevance      = sum(scores) / len(scores) if scores else 0.6
    unique_sources     = len({e["source"] for e in evidence})
    unique_regulations = {e["regulation"] for e in evidence if e.get("regulation")}
    source_diversity   = 1.0 if unique_sources >= 3 else (0.75 if unique_sources == 2 else 0.5)
    reg_coverage       = (1.0 if len(unique_regulations) >= 3 else
                          0.75 if len(unique_regulations) == 2 else
                          0.5 if len(unique_regulations) == 1 else 0.0)
    return round(min(0.5 * avg_relevance + 0.25 * source_diversity + 0.25 * reg_coverage, 1.0), 3)


def external_evidence_summarizer_node(state: AgentState) -> dict[str, Any]:
    """Summarize raw web search results, merge into evidence, and recompute score."""
    t_start = time.time()
    external = state.get("external_evidence", [])
    raw_text = "\n\n".join(
        f"Search: {r['query']}\nResult: {r['result']}" for r in external
    )

    prompt  = ChatPromptTemplate.from_messages([
        ("system", SUMMARIZER_SYSTEM),
        ("human",  "/no_think\nQuery: {query}\n\nWeb results:\n{raw}"),
    ])
    chain   = prompt | get_llm()
    summary: str = chain.invoke({"query": state["user_query"], "raw": raw_text}).content

    web_items: list[dict[str, Any]] = [
        {
            "text":       r["result"],
            "source":     "web_search",
            "regulation": None,
            "title":      f"Web: {r.get('url', r['query'])}",
            "page":       None,
            "section":    None,
            "score":      None,
            "url":        r.get("url"),
        }
        for r in external
    ] + [{
        "text":       summary,
        "source":     "web_search_summary",
        "regulation": None,
        "title":      "Web Search Summary",
        "page":       None,
        "section":    None,
        "score":      None,
        "url":        None,
    }]

    merged_evidence = state.get("evidence", []) + web_items
    new_score       = recompute_evidence_score(merged_evidence)
    new_gaps        = [] if new_score >= EVIDENCE_SUFFICIENCY_THRESHOLD else state.get("evidence_gaps", [])

    trace_entry = {
        "node":           "external_evidence_summarizer",
        "timestamp":      t_start,
        "summary_length": len(summary),
        "new_score":      new_score,
    }
    logger.info("external_evidence_summarizer: summary_length=%d  new_score=%.3f", len(summary), new_score)

    return {
        "evidence":       merged_evidence,
        "evidence_score": new_score,
        "evidence_gaps":  new_gaps,
        "trace":          [trace_entry],
    }
