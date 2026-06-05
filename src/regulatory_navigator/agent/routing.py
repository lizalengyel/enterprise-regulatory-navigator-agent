from __future__ import annotations

from .state import AgentState

_REGULATORY_KEYWORDS = [
    "gdpr", "dora", "nis2", "ai act", "regulation", "compliance", "obligation",
    "data protection", "data breach", "cybersecurity", "ict risk", "personal data",
    "supervisory", "controller", "processor", "high-risk", "incident", "notification",
    "third-party", "outsourc", "consent", "lawful", "privacy", "penalty", "fine",
    "directive", "framework", "assess", "audit", "certif", "tlpt", "csirt",
]

_REGULATORY_INTENTS = [
    "compliance", "regulation", "risk", "legal", "obligation", "assessment",
    "check", "analysis", "gap", "audit",
]

_KNOWN_INDUSTRIES = [
    "financial", "bank", "insurance", "healthcare", "health", "hospital",
    "technology", "telecom", "energy", "transport", "public",
]


def route_after_profiling(state: AgentState) -> str:
    """
    After the profiler:
    - in scope  → regulation_router
    - out of scope → out_of_scope
    """
    query   = (state.get("user_query") or "").lower()
    profile = state.get("profile") or {}
    intent  = (profile.get("intent") or "").lower()
    industry = (profile.get("industry") or "").lower()

    if any(kw in query for kw in _REGULATORY_KEYWORDS):
        return "in_scope"
    if any(kw in intent for kw in _REGULATORY_INTENTS):
        return "in_scope"
    if any(kw in industry for kw in _KNOWN_INDUSTRIES):
        return "in_scope"
    if profile.get("ai_use_case"):
        return "in_scope"

    return "out_of_scope"


def route_after_sufficiency(state: AgentState) -> str:
    """
    After the evidence sufficiency check:
    - web_search_required=True  → go to web search
    - web_search_required=False → go to risk analyzer
      (covers both: evidence is sufficient AND max iterations reached)
    """
    if state.get("web_search_required"):
        return "web_search"
    return "risk_analyzer"
