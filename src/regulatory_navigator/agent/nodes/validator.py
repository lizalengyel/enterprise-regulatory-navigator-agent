from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..llm import get_llm
from ..prompts import VALIDATOR_SYSTEM
from ..state import AgentState

logger = logging.getLogger(__name__)


class ValidationOutput(BaseModel):
    valid:            bool      = Field(description="True if the answer is factually grounded in the evidence")
    issues:           list[str] = Field(description="List of unsupported claims or citation errors found")
    corrected_answer: str | None = Field(None, description="Corrected answer if issues were found, else null")


def citation_guardrail_validator_node(state: AgentState) -> dict[str, Any]:
    """Validate that the final answer is grounded in the retrieved evidence."""
    evidence_text = "\n\n".join(
        f"[{i+1}] {e.get('title', 'Source')}\n{e['text']}"
        for i, e in enumerate(state.get("evidence", []))
    )

    prompt  = ChatPromptTemplate.from_messages([
        ("system", VALIDATOR_SYSTEM),
        ("human",  "/no_think\nAnswer:\n{answer}\n\nEvidence:\n{evidence}"),
    ])
    chain   = prompt | get_llm().with_structured_output(ValidationOutput)
    result: ValidationOutput = chain.invoke({
        "answer":   state.get("final_answer", ""),
        "evidence": evidence_text,
    })

    final = result.corrected_answer if not result.valid and result.corrected_answer else state.get("final_answer", "")

    trace_entry = {
        "node":      "citation_guardrail_validator",
        "timestamp": time.time(),
        "valid":     result.valid,
        "issues":    result.issues,
    }
    logger.info("citation_guardrail_validator: valid=%s  issues=%d", result.valid, len(result.issues))

    return {
        "final_answer":      final,
        "validation_result": result.model_dump(),
        "trace":             [trace_entry],
    }
