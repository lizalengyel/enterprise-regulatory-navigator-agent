from __future__ import annotations

import time
from typing import Any

from ..state import AgentState
from ..tools import risk_scoring_tool


def risk_scorer_node(state: AgentState) -> dict[str, Any]:
    """Compute a deterministic composite risk score using risk_scoring_tool (parallel with checklist)."""
    t_start = time.time()
    risk    = state.get("risk_assessment", {})

    result = risk_scoring_tool.invoke({
        "regulations":       state.get("selected_regulations", []),
        "severity":          risk.get("severity", "medium"),
        "obligations_count": len(state.get("obligations", [])),
        "evidence_score":    state.get("evidence_score", 0.5),
    })

    return {
        "risk_score": result,
        "trace": [{
            "node":            "risk_scorer",
            "timestamp":       t_start,
            "composite_score": result["composite_score"],
            "risk_level":      result["risk_level"],
        }],
    }
