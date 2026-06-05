from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

TRUSTED_DOMAINS = [
    "eur-lex.europa.eu",
    "digital-strategy.ec.europa.eu",
    "eba.europa.eu",
    "esma.europa.eu",
    "eiopa.europa.eu",
]


@tool
def web_search_tool(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """
    Search the web for regulatory information using Tavily.
    Restricted to trusted EU regulatory authority domains only.
    Use this tool when the vector store evidence is insufficient to answer
    the user's question about EU regulations (GDPR, DORA, NIS2, AI Act).

    Args:
        query:       Search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of results with title, content, url, and score.
    """
    search   = TavilySearch(max_results=max_results, include_domains=TRUSTED_DOMAINS)
    response = search.invoke(query)

    if isinstance(response, dict):
        hits = response.get("results", [])
    elif isinstance(response, list):
        hits = response
    else:
        # Tavily returned a plain string — no structured results available
        return []

    return [
        {
            "title":   hit.get("title"),
            "content": hit.get("content", ""),
            "url":     hit.get("url", ""),
            "score":   hit.get("score"),
        }
        for hit in hits
        if isinstance(hit, dict)
    ]
