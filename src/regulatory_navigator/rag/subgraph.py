from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from .evidence import EvidencePack, build_evidence_pack, format_citations, format_context
from .reranker import rerank
from .retriever import retrieve
from .state import RAGState

logger = logging.getLogger(__name__)

TOP_K_RETRIEVE = 20
TOP_N_RERANK   = 5


# Nodes

def retrieve_node(state: RAGState) -> dict:
    candidates = retrieve(
        state["query"],
        regulations=state.get("regulations"),
        top_k=TOP_K_RETRIEVE,
    )
    logger.info("retrieve: %d candidates", len(candidates))
    return {"candidates": candidates}


def rerank_node(state: RAGState) -> dict:
    reranked = rerank(state["query"], state["candidates"], top_n=TOP_N_RERANK)
    logger.info("rerank: %d results kept", len(reranked))
    return {"reranked": reranked}


def build_evidence_node(state: RAGState) -> dict:
    pack = build_evidence_pack(
        query=state["query"],
        results=state["reranked"],
        required_regulations=state.get("required_regulations"),
    )
    logger.info(
        "evidence: score=%.3f  sufficient=%s  gaps=%s",
        pack.evidence_score, pack.sufficient, pack.gaps,
    )
    return {"evidence_pack": pack}


# Graph assembly

def build_rag_subgraph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve",       retrieve_node)
    graph.add_node("rerank",         rerank_node)
    graph.add_node("build_evidence", build_evidence_node)

    graph.add_edge(START,            "retrieve")
    graph.add_edge("retrieve",       "rerank")
    graph.add_edge("rerank",         "build_evidence")
    graph.add_edge("build_evidence", END)

    return graph


rag_subgraph = build_rag_subgraph().compile()

def run_rag(
    query: str,
    regulations: list[str] | None = None,
    required_regulations: list[str] | None = None,
) -> EvidencePack:
    """Run the RAG subgraph and return the final EvidencePack."""
    final_state = rag_subgraph.invoke({
        "query":                query,
        "regulations":          regulations,
        "required_regulations": required_regulations or regulations,
        "candidates":           [],
        "reranked":             [],
        "evidence_pack":        None,
    })
    return final_state["evidence_pack"]
