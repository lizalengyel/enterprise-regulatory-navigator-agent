from __future__ import annotations

from langchain_core.tools import tool

# Base risk weights per regulation
_REGULATION_BASE_RISK: dict[str, float] = {
    "GDPR":    0.8,   # High — fines up to 4% global turnover
    "DORA":    0.85,  # High — financial sector operational resilience
    "NIS2":    0.75,  # Medium-High — cybersecurity obligations
    "AI Act":  0.9,   # Very High — high-risk AI systems
}

_SEVERITY_MULTIPLIER: dict[str, float] = {
    "high":    1.0,
    "medium":  0.65,
    "low":     0.35,
    "unknown": 0.5,
}


@tool
def risk_scoring_tool(
    regulations:      list[str],
    severity:         str  = "medium",
    obligations_count: int = 0,
    evidence_score:   float = 0.5,
) -> dict:
    """
    Compute a composite regulatory risk score for a given compliance scenario.
    The score is deterministic and explainable — suitable for reporting.

    Args:
        regulations:       List of applicable regulations (e.g. ["GDPR", "DORA"]).
        severity:          Overall severity level: 'high', 'medium', 'low', or 'unknown'.
        obligations_count: Number of identified regulatory obligations.
        evidence_score:    RAG evidence quality score (0–1).

    Returns:
        Dictionary with composite_score (0–1), risk_level, per-regulation scores,
        and a short explanation.
    """
    if not regulations:
        return {
            "composite_score": 0.0,
            "risk_level":      "unknown",
            "per_regulation":  {},
            "explanation":     "No regulations identified.",
        }

    sev_mult = _SEVERITY_MULTIPLIER.get(severity.lower(), 0.5)

    # Per-regulation weighted score
    per_reg: dict[str, float] = {}
    for reg in regulations:
        base     = _REGULATION_BASE_RISK.get(reg, 0.5)
        per_reg[reg] = round(base * sev_mult, 3)

    # Composite: weighted average + obligation density bonus
    avg_reg_score    = sum(per_reg.values()) / len(per_reg)
    obligation_bonus = min(obligations_count * 0.02, 0.15)  # cap at +15%
    evidence_penalty = (1.0 - evidence_score) * 0.1         # low evidence → slight penalty
    composite        = round(min(avg_reg_score + obligation_bonus - evidence_penalty, 1.0), 3)

    if composite >= 0.75:
        risk_level = "critical"
    elif composite >= 0.55:
        risk_level = "high"
    elif composite >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    explanation = (
        f"Composite score {composite:.2f} based on {len(regulations)} regulation(s) "
        f"({', '.join(regulations)}), severity '{severity}', "
        f"{obligations_count} obligation(s), evidence quality {evidence_score:.2f}."
    )

    return {
        "composite_score": composite,
        "risk_level":      risk_level,
        "per_regulation":  per_reg,
        "explanation":     explanation,
    }
