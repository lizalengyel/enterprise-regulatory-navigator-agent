from __future__ import annotations

from operator import add
from typing import Annotated, Any

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # input
    user_query:               str
    conversation_history:     list[dict[str, str]]

    # use case profiling
    intent:                   str
    industry:                 str
    jurisdiction:             str
    ai_use_case:              str
    profile:                  dict[str, Any]

    # regulation routing
    selected_regulations:     list[str]

    # RAG evidence
    evidence:                 list[dict[str, Any]]
    evidence_score:           float
    evidence_gaps:            list[str]
    regulations_covered:      list[str]
    sufficient:               bool

    # web search fallback
    web_search_required:      bool
    web_search_iterations:    int
    max_web_search_iterations: int
    external_evidence:        list[dict[str, Any]]

    # analysis
    risk_assessment:          dict[str, Any]
    obligations:              list[str]
    risk_score:               dict[str, Any]   # from risk_scoring_tool (parallel node)
    checklist:                list[str]

    # output
    final_answer:             str
    validation_result:        dict[str, Any]

    # observability
    trace:                    Annotated[list[dict[str, Any]], add]
