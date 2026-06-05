from __future__ import annotations

import logging
import time
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

_REGULATION_KEYWORDS: dict[str, list[str]] = {
    "GDPR": [
        "gdpr", "general data protection", "personal data", "data subject",
        "data controller", "data processor", "right to erasure", "right to be forgotten",
        "data breach", "dpo", "data protection officer", "consent", "lawful basis",
    ],
    "DORA": [
        "dora", "digital operational resilience", "ict risk", "ict third-party",
        "financial entity", "operational resilience", "incident reporting",
        "penetration testing", "tlpt",
    ],
    "NIS2": [
        "nis2", "nis 2", "network and information security", "essential entity",
        "important entity", "csirt", "cyber incident", "cybersecurity measures",
        "supply chain security",
    ],
    "AI Act": [
        "ai act", "artificial intelligence act", "high-risk ai", "gpai",
        "general purpose ai", "foundation model", "prohibited ai", "ai system",
        "conformity assessment", "ce marking", "transparency obligation",
    ],
}

_INDUSTRY_REGULATION_MAP: dict[str, list[str]] = {
    "financial":  ["DORA", "GDPR"],
    "bank":       ["DORA", "GDPR"],
    "insurance":  ["DORA", "GDPR"],
    "healthcare": ["GDPR", "NIS2"],
    "health":     ["GDPR", "NIS2"],
    "technology": ["GDPR", "NIS2", "AI Act"],
    "telecom":    ["NIS2", "GDPR"],
    "energy":     ["NIS2"],
    "transport":  ["NIS2"],
    "public":     ["NIS2", "GDPR"],
}


def regulation_router_node(state: AgentState) -> dict[str, Any]:
    """
    Detect relevant regulations using keyword matching on the query
    AND profile fields (industry, intent, ai_use_case).
    Falls back to all regulations if nothing is detected.
    """
    t_start = time.time()
    query_lower = state["user_query"].lower()
    profile     = state.get("profile", {})
    matched: set[str] = set()

    for regulation, keywords in _REGULATION_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            matched.add(regulation)

    industry = (profile.get("industry") or "").lower()
    for industry_kw, regs in _INDUSTRY_REGULATION_MAP.items():
        if industry_kw in industry:
            matched.update(regs)

    if profile.get("ai_use_case"):
        matched.add("AI Act")

    intent = (profile.get("intent") or "").lower()
    if any(kw in intent for kw in ("data breach", "personal data", "privacy")):
        matched.add("GDPR")
    if any(kw in intent for kw in ("cyber", "incident", "resilience")):
        matched.update(["NIS2", "DORA"])

    selected = sorted(matched) if matched else list(_REGULATION_KEYWORDS.keys())

    trace_entry = {
        "node":       "regulation_router",
        "timestamp":  t_start,
        "matched":    sorted(matched),
        "selected":   selected,
        "fallback":   not matched,
        "profile_signals": {
            "industry":    industry,
            "ai_use_case": profile.get("ai_use_case"),
            "intent":      intent,
        },
    }

    logger.info("regulation_router: selected=%s (fallback=%s)", selected, not matched)

    return {
        "selected_regulations": selected,
        "trace": [trace_entry],
    }
