from __future__ import annotations

import logging
import time
from typing import Any

from ..state import AgentState
from ..tools import rag_retrieval_tool

logger = logging.getLogger(__name__)


def rag_node(state: AgentState) -> dict[str, Any]:
    """Run the RAG subgraph and flatten results into agent state."""
    t_start = time.time()
    query       = state["user_query"]
    regulations = state.get("selected_regulations")

    result = rag_retrieval_tool.invoke({"query": query, "regulations": regulations})

    trace_entry = {
        "node":                "rag",
        "timestamp":           t_start,
        "regulations":         regulations,
        "evidence_score":      result["evidence_score"],
        "sufficient":          result["sufficient"],
        "regulations_covered": result["regulations_covered"],
        "gaps":                result["gaps"],
    }

    logger.info(
        "rag_node: score=%.3f  sufficient=%s  covered=%s",
        result["evidence_score"], result["sufficient"], result["regulations_covered"],
    )

    return {
        "evidence":             result["evidence"],
        "evidence_score":       result["evidence_score"],
        "evidence_gaps":        result["gaps"],
        "regulations_covered":  result["regulations_covered"],
        "sufficient":           result["sufficient"],
        "trace":                [trace_entry],
    }
