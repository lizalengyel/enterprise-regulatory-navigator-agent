from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from ..llm import get_llm
from ..prompts import CHECKLIST_SYSTEM
from ..state import AgentState

logger = logging.getLogger(__name__)


def checklist_generator_node(state: AgentState) -> dict[str, Any]:
    """Generate an actionable compliance checklist from the obligations using LLM."""
    t_start     = time.time()
    obligations = "\n".join(f"- {o}" for o in state.get("obligations", []))

    prompt   = ChatPromptTemplate.from_messages([
        ("system", CHECKLIST_SYSTEM),
        ("human",  "/no_think\nContext: {profile}\n\nObligations:\n{obligations}"),
    ])
    chain    = prompt | get_llm()
    response = chain.invoke({
        "profile":     str(state.get("profile", {})),
        "obligations": obligations,
    }).content

    lines     = response.splitlines()
    checklist = [l.strip() for l in lines if l.strip().startswith("- [ ]")]

    if not checklist:
        checklist = [
            f"- [ ] {l.strip().lstrip('-•*').strip()}"
            for l in lines
            if l.strip() and not l.strip().startswith("#")
        ]

    trace_entry = {
        "node":      "checklist_generator",
        "timestamp": t_start,
        "items":     len(checklist),
    }
    logger.info("checklist_generator: %d items", len(checklist))

    return {
        "checklist": checklist,
        "trace":     [trace_entry],
    }
