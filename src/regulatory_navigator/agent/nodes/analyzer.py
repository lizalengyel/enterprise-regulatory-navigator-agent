from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..llm import get_llm
from ..prompts import RISK_ANALYZER_SYSTEM
from ..state import AgentState

logger = logging.getLogger(__name__)


class RiskAssessmentOutput(BaseModel):
    risks:       list[str] = Field(description="List of identified regulatory risks")
    obligations: list[str] = Field(description="List of concrete obligations the entity must fulfil")
    severity:    str       = Field(description="Overall severity level: 'high', 'medium', or 'low'")


def risk_obligation_analyzer_node(state: AgentState) -> dict[str, Any]:
    """Analyze risks and obligations from the evidence."""
    t_start = time.time()
    if not state.get("evidence"):
        logger.warning("risk_obligation_analyzer: no evidence — returning fallback")
        return {
            "risk_assessment": {
                "risks":       ["Insufficient evidence to assess regulatory risk."],
                "obligations": [],
                "severity":    "unknown",
            },
            "obligations": [],
            "trace": [{
                "node":      "risk_obligation_analyzer",
                "timestamp": t_start,
                "fallback":  True,
            }],
        }

    evidence_text = "\n\n".join(
        f"[{i+1}] {e.get('title', '')}\n{e['text']}"
        for i, e in enumerate(state.get("evidence", []))
    )

    prompt  = ChatPromptTemplate.from_messages([
        ("system", RISK_ANALYZER_SYSTEM),
        ("human",  "/no_think\nQuery: {query}\n\nProfile: {profile}\n\nEvidence:\n{evidence}"),
    ])
    chain   = prompt | get_llm().with_structured_output(RiskAssessmentOutput)
    result: RiskAssessmentOutput = chain.invoke({
        "query":    state["user_query"],
        "profile":  str(state.get("profile", {})),
        "evidence": evidence_text,
    })

    trace_entry = {
        "node":        "risk_obligation_analyzer",
        "timestamp":   t_start,
        "severity":    result.severity,
        "risks":       len(result.risks),
        "obligations": len(result.obligations),
    }
    logger.info(
        "risk_obligation_analyzer: severity=%s  risks=%d  obligations=%d",
        result.severity, len(result.risks), len(result.obligations),
    )

    return {
        "risk_assessment": result.model_dump(),
        "obligations":     result.obligations,
        "trace":           [trace_entry],
    }
