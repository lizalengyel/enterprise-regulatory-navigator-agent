from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..prompts import ANSWER_COMPOSER_SYSTEM
from ..state import AgentState

logger = logging.getLogger(__name__)


def answer_composer_node(state: AgentState) -> dict[str, Any]:
    """Compose the final answer from all gathered information."""
    evidence_text  = "\n\n".join(
        f"[{i+1}] {e.get('title', 'Source')}\n{e['text']}"
        for i, e in enumerate(state.get("evidence", []))
    )
    checklist_text = "\n".join(state.get("checklist", []))

    prompt   = ChatPromptTemplate.from_messages([
        ("system", ANSWER_COMPOSER_SYSTEM),
        ("human",  (
            "/no_think\n"
            "Query: {query}\n\n"
            "Risk Assessment: {risk_assessment}\n\n"
            "Checklist:\n{checklist}\n\n"
            "Evidence:\n{evidence}"
        )),
    ])
    chain    = prompt | get_llm(temperature=0.1)
    response = chain.invoke({
        "query":           state["user_query"],
        "risk_assessment": str(state.get("risk_assessment", {})),
        "checklist":       checklist_text,
        "evidence":        evidence_text,
    }).content

    trace_entry = {
        "node":      "answer_composer",
        "timestamp": time.time(),
        "length":    len(response),
    }
    logger.info("answer_composer: answer_length=%d", len(response))

    return {
        "final_answer": response,
        "trace":        [trace_entry],
    }
