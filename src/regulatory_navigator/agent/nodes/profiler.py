from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..llm import get_llm
from ..prompts import USE_CASE_PROFILER_SYSTEM
from ..state import AgentState
from ._constants import MAX_WEB_SEARCH_ITERATIONS

logger = logging.getLogger(__name__)


class UseCaseProfile(BaseModel):
    intent:       str        = Field(description="Regulatory intent, e.g. 'compliance check', 'risk assessment'")
    industry:     str        = Field(description="Industry sector, e.g. 'financial services', 'healthcare'")
    jurisdiction: str        = Field(description="Jurisdiction, e.g. 'EU', 'UK', 'global'")
    ai_use_case:  str | None = Field(None, description="Specific AI use case if relevant to the AI Act, else null")


def format_history(history: list[dict[str, str]], max_turns: int = 3) -> str:
    if not history:
        return ""
    recent = history[-(max_turns * 2):]
    lines  = ["Previous conversation:"]
    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"][:300] + "…" if len(msg["content"]) > 300 else msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def use_case_profiler_node(state: AgentState) -> dict[str, Any]:
    """Normalize the query then extract intent, industry, jurisdiction and AI use case."""
    t_start = time.time()

    raw   = state.get("user_query", "").strip()
    query = re.sub(r"\s+", " ", raw)

    history     = state.get("conversation_history", [])
    history_str = format_history(history)

    human_content = "/no_think\n"
    if history_str:
        human_content += f"{history_str}\n\nCurrent query: {{query}}"
    else:
        human_content += "{query}"

    prompt  = ChatPromptTemplate.from_messages([
        ("system", USE_CASE_PROFILER_SYSTEM),
        ("human",  human_content),
    ])
    chain   = prompt | get_llm().with_structured_output(UseCaseProfile)
    profile: UseCaseProfile = chain.invoke({"query": query})

    ai_use_case = profile.ai_use_case
    if ai_use_case and (
        "?" in ai_use_case
        or len(ai_use_case.split()) > 8
        or query.lower()[:30] in ai_use_case.lower()
    ):
        ai_use_case = None

    profile_dict = {**profile.model_dump(), "ai_use_case": ai_use_case}

    trace_entry = {
        "node":      "use_case_profiler",
        "timestamp": t_start,
        "profile":   profile_dict,
    }
    logger.info("use_case_profiler: %s", profile_dict)

    return {
        "user_query":                query,           # normalized query written back to state
        "web_search_iterations":     0,
        "max_web_search_iterations": MAX_WEB_SEARCH_ITERATIONS,
        "web_search_required":       False,
        "external_evidence":         [],
        "intent":                    profile.intent,
        "industry":                  profile.industry,
        "jurisdiction":              profile.jurisdiction,
        "ai_use_case":               ai_use_case,
        "profile":                   profile_dict,
        "trace":                     [trace_entry],
    }
