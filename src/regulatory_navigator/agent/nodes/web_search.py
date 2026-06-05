from __future__ import annotations

import logging
import time
from typing import Any

from ..state import AgentState
from ..tools import web_search_tool

logger = logging.getLogger(__name__)


def gap_to_search_query(base_query: str, gap: str) -> str:
    if ":" in gap:
        regulation = gap.split(":")[-1].strip()
        return f"{base_query} {regulation} EU regulatory obligations compliance"
    return f"{base_query} EU regulation compliance"


def web_search_node(state: AgentState) -> dict[str, Any]:
    """Search the web via Tavily to fill evidence gaps. Increments the iteration counter."""
    t_start = time.time()
    query      = state["user_query"]
    gaps       = state.get("evidence_gaps", [])
    iterations = state.get("web_search_iterations", 0)

    search_queries = (
        [gap_to_search_query(query, gap) for gap in gaps[:2]]
        if gaps else
        [f"{query} EU regulation compliance"]
    )

    results: list[dict[str, Any]] = []
    for sq in search_queries:
        try:
            hits = web_search_tool.invoke({"query": sq})
            for hit in hits:
                results.append({"query": sq, "result": hit["content"], "url": hit["url"]})
            logger.info("web_search: query=%r  hits=%d", sq[:60], len(hits))
        except Exception as exc:
            logger.warning("web_search failed for %r: %s", sq, exc)

    trace_entry = {
        "node":      "web_search",
        "timestamp": t_start,
        "queries":   search_queries,
        "results":   len(results),
        "iteration": iterations + 1,
    }

    return {
        "external_evidence":     state.get("external_evidence", []) + results,
        "web_search_iterations": iterations + 1,
        "trace":                 [trace_entry],
    }
