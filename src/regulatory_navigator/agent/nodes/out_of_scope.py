from __future__ import annotations

import time
from typing import Any

from ..state import AgentState

OUT_OF_SCOPE_ANSWER = (
    "Your question appears to be outside the scope of this system.\n\n"
    "I specialise exclusively in EU regulatory compliance across:\n"
    "- **GDPR** — General Data Protection Regulation\n"
    "- **DORA** — Digital Operational Resilience Act\n"
    "- **NIS2** — Network and Information Security Directive\n"
    "- **EU AI Act** — Artificial Intelligence Act\n\n"
    "Please ask a question related to obligations, risks, or compliance "
    "requirements under one of these frameworks."
)


def out_of_scope_node(state: AgentState) -> dict[str, Any]:
    """Return a fixed out-of-scope response without invoking any LLM or RAG."""
    return {
        "final_answer":     OUT_OF_SCOPE_ANSWER,
        "validation_result": {"valid": True, "issues": [], "corrected_answer": None},
        "trace": [{
            "node":      "out_of_scope",
            "timestamp": time.time(),
        }],
    }
