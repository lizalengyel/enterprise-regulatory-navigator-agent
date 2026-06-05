from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    answer_composer_node,
    checklist_generator_node,
    citation_guardrail_validator_node,
    evidence_sufficiency_node,
    external_evidence_summarizer_node,
    out_of_scope_node,
    rag_node,
    regulation_router_node,
    risk_obligation_analyzer_node,
    risk_scorer_node,
    use_case_profiler_node,
    web_search_node,
)
from .routing import route_after_profiling, route_after_sufficiency
from .state import AgentState


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("use_case_profiler",            use_case_profiler_node)
    graph.add_node("out_of_scope",                 out_of_scope_node)
    graph.add_node("regulation_router",            regulation_router_node)
    graph.add_node("rag",                          rag_node)
    graph.add_node("evidence_sufficiency",         evidence_sufficiency_node)
    graph.add_node("web_search",                   web_search_node)
    graph.add_node("external_evidence_summarizer", external_evidence_summarizer_node)
    graph.add_node("risk_obligation_analyzer",     risk_obligation_analyzer_node)
    # Fan-out
    graph.add_node("checklist_generator",          checklist_generator_node)
    graph.add_node("risk_scorer",                  risk_scorer_node)
    # Fan-in
    graph.add_node("answer_composer",              answer_composer_node)
    graph.add_node("citation_guardrail_validator", citation_guardrail_validator_node)

    # Linear pipeline
    graph.add_edge(START,               "use_case_profiler")
    graph.add_edge("out_of_scope",      END)

    # Scope check after profiling
    graph.add_conditional_edges(
        "use_case_profiler",
        route_after_profiling,
        {"in_scope": "regulation_router", "out_of_scope": "out_of_scope"},
    )
    graph.add_edge("regulation_router", "rag")
    graph.add_edge("rag",               "evidence_sufficiency")

    # Conditional — enough evidence?
    graph.add_conditional_edges(
        "evidence_sufficiency",
        route_after_sufficiency,
        {
            "web_search":    "web_search",
            "risk_analyzer": "risk_obligation_analyzer",
        },
    )

    # Web search loop
    graph.add_edge("web_search",                   "external_evidence_summarizer")
    graph.add_edge("external_evidence_summarizer", "evidence_sufficiency")

    # Fan-out: risk_obligation_analyzer → checklist_generator AND risk_scorer (parallel)
    graph.add_edge("risk_obligation_analyzer", "checklist_generator")
    graph.add_edge("risk_obligation_analyzer", "risk_scorer")

    # Fan-in: both must complete → answer_composer → validator → END
    graph.add_edge("checklist_generator",          "answer_composer")
    graph.add_edge("risk_scorer",                  "answer_composer")
    graph.add_edge("answer_composer",              "citation_guardrail_validator")
    graph.add_edge("citation_guardrail_validator", END)

    return graph


agent_graph = build_graph().compile()
