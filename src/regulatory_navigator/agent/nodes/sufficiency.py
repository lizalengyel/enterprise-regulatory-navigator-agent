from __future__ import annotations

import logging
import time
from typing import Any

from ..state import AgentState
from ._constants import EVIDENCE_SUFFICIENCY_THRESHOLD, MAX_WEB_SEARCH_ITERATIONS

logger = logging.getLogger(__name__)


def evidence_sufficiency_node(state: AgentState) -> dict[str, Any]:
    """
    Check whether retrieved evidence meets the sufficiency threshold.
    Sets web_search_required — iteration counter is incremented in web_search_node.
    """
    t_start = time.time()
    score      = state.get("evidence_score", 0.0)
    gaps       = state.get("evidence_gaps", [])
    iterations = state.get("web_search_iterations", 0)
    max_iter   = state.get("max_web_search_iterations", MAX_WEB_SEARCH_ITERATIONS)

    sufficient      = score >= EVIDENCE_SUFFICIENCY_THRESHOLD and not gaps
    search_required = not sufficient and iterations < max_iter

    trace_entry = {
        "node":                "evidence_sufficiency",
        "timestamp":           t_start,
        "evidence_score":      score,
        "gaps":                gaps,
        "sufficient":          sufficient,
        "web_search_required": search_required,
        "iterations":          iterations,
    }

    logger.info(
        "evidence_sufficiency: score=%.3f  sufficient=%s  web_search_required=%s  iter=%d/%d",
        score, sufficient, search_required, iterations, max_iter,
    )

    return {
        "web_search_required": search_required,
        "trace":               [trace_entry],
    }
