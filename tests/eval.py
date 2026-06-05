"""
Functional evaluation of the Enterprise Regulatory Navigator.

Evaluates two nodes:
  1. Use Case Profiler (LLM) — industry match, AI use case detection, intent match
  2. RAG Retrieval           — regulation hit, article hit, keyword hit, evidence score

Also tests robustness with out-of-scope (noise) queries.

Usage:
    uv run python tests/eval.py
    uv run python tests/eval.py --output results/eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.disable(logging.CRITICAL)

from regulatory_navigator.agent.nodes.profiler import use_case_profiler_node
from regulatory_navigator.agent.routing import route_after_profiling
from regulatory_navigator.rag.subgraph import run_rag

# Validation set — 20 queries
# Fields:
#   query                   — the input question
#   expected_industry       — expected industry keyword (substring match, None = noise)
#   expected_ai_use_case    — True if AI use case should be detected, False if not, None = noise
#   expected_intent         — expected intent keyword (substring match, None = noise)
#   expected_regulations    — correct regulation set for RAG (empty = noise)
#   expected_article        — article that must appear in retrieved chunks (None = skip)
#   expected_keywords       — words that should appear in retrieved text ([] = noise)
#   noise                   — True if out-of-scope (robustness test)

EVAL_SET = [
    # GDPR — 4 questions
    {
        "query": "A company suffers a personal data breach. What are the notification obligations?",
        "expected_industry":    "technology",
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR"],
        "expected_article":     "Article 33",
        "expected_keywords":    ["72 hours", "supervisory authority", "notification"],
        "noise": False,
    },
    {
        "query": "What rights does a data subject have under GDPR?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR"],
        "expected_article":     "Article 15",
        "expected_keywords":    ["access", "erasure", "rectification", "data subject"],
        "noise": False,
    },
    {
        "query": "When is a Data Protection Officer required under GDPR?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR"],
        "expected_article":     "Article 37",
        "expected_keywords":    ["data protection officer", "DPO", "public authority"],
        "noise": False,
    },
    {
        "query": "Can personal data be transferred to a third country under GDPR?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR"],
        "expected_article":     "Article 46",
        "expected_keywords":    ["third country", "transfer", "adequacy", "safeguards"],
        "noise": False,
    },

    # DORA — 4 questions
    {
        "query": "A bank outsources critical services to a cloud provider. What DORA obligations apply?",
        "expected_industry":    "bank",
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["DORA"],
        "expected_article":     "Article 28",
        "expected_keywords":    ["third-party", "ICT", "contractual", "register"],
        "noise": False,
    },
    {
        "query": "What incident reporting timelines are required under DORA?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["DORA"],
        "expected_article":     "Article 19",
        "expected_keywords":    ["incident", "reporting", "classification", "significant"],
        "noise": False,
    },
    {
        "query": "What is Threat-Led Penetration Testing under DORA?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["DORA"],
        "expected_article":     "Article 26",
        "expected_keywords":    ["TLPT", "penetration testing", "threat-led"],
        "noise": False,
    },
    {
        "query": "What ICT risk management obligations does DORA impose on financial entities?",
        "expected_industry":    "financial",
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["DORA"],
        "expected_article":     "Article 6",
        "expected_keywords":    ["ICT risk", "framework", "financial entity", "management"],
        "noise": False,
    },

    # NIS2 — 3 questions
    {
        "query": "What cybersecurity measures are required for essential entities under NIS2?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["NIS2"],
        "expected_article":     "Article 21",
        "expected_keywords":    ["cybersecurity", "risk management", "essential", "measures"],
        "noise": False,
    },
    {
        "query": "What are the incident notification obligations under NIS2?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["NIS2"],
        "expected_article":     "Article 23",
        "expected_keywords":    ["significant incident", "CSIRT", "24 hours", "notification"],
        "noise": False,
    },
    {
        "query": "Which sectors fall under NIS2 as essential entities?",
        "expected_industry":    None,
        "expected_ai_use_case": False,
        "expected_intent":      "compliance",
        "expected_regulations": ["NIS2"],
        "expected_article":     "Article 3",
        "expected_keywords":    ["essential", "energy", "transport", "banking", "sector"],
        "noise": False,
    },

    # AI Act — 3 questions
    {
        "query": "A bank uses AI to assess loan applications. Is this a high-risk AI system?",
        "expected_industry":    "bank",
        "expected_ai_use_case": True,
        "expected_intent":      "risk",
        "expected_regulations": ["AI Act"],
        "expected_article":     "Article 6",
        "expected_keywords":    ["high-risk", "creditworthiness", "Annex III", "classification"],
        "noise": False,
    },
    {
        "query": "What transparency obligations apply to general-purpose AI models?",
        "expected_industry":    None,
        "expected_ai_use_case": True,
        "expected_intent":      "compliance",
        "expected_regulations": ["AI Act"],
        "expected_article":     "Article 53",
        "expected_keywords":    ["transparency", "general-purpose", "GPAI", "disclosure"],
        "noise": False,
    },
    {
        "query": "What AI practices are prohibited under the EU AI Act?",
        "expected_industry":    None,
        "expected_ai_use_case": True,
        "expected_intent":      "compliance",
        "expected_regulations": ["AI Act"],
        "expected_article":     "Article 5",
        "expected_keywords":    ["prohibited", "manipulation", "social scoring", "biometric"],
        "noise": False,
    },

    # Cross-regulation — 3 questions
    {
        "query": "A bank wants to deploy a GenAI chatbot for customer complaints. What obligations apply?",
        "expected_industry":    "bank",
        "expected_ai_use_case": True,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR", "DORA", "AI Act"],
        "expected_article":     None,
        "expected_keywords":    ["personal data", "ICT risk", "transparency", "third-party"],
        "noise": False,
    },
    {
        "query": "An employer uses AI to screen job applicants.",
        "expected_industry":    None,
        "expected_ai_use_case": True,
        "expected_intent":      "compliance",
        "expected_regulations": ["AI Act", "GDPR"],
        "expected_article":     None,
        "expected_keywords":    ["employment", "high-risk", "candidate", "human oversight"],
        "noise": False,
    },
    {
        "query": "A hospital uses AI for patient triage.",
        "expected_industry":    "health",
        "expected_ai_use_case": True,
        "expected_intent":      "compliance",
        "expected_regulations": ["GDPR", "AI Act", "NIS2"],
        "expected_article":     None,
        "expected_keywords":    ["health data", "high-risk", "cybersecurity", "human oversight"],
        "noise": False,
    },

    # Noise / out-of-scope — 3 questions
    {
        "query": "How do you play Uno?",
        "expected_industry":    None,
        "expected_ai_use_case": None,
        "expected_intent":      None,
        "expected_regulations": [],
        "expected_article":     None,
        "expected_keywords":    [],
        "noise": True,
    },
    {
        "query": "What is the best recipe for chocolate cake?",
        "expected_industry":    None,
        "expected_ai_use_case": None,
        "expected_intent":      None,
        "expected_regulations": [],
        "expected_article":     None,
        "expected_keywords":    [],
        "noise": True,
    },
    {
        "query": "Who won the FIFA World Cup in 2022?",
        "expected_industry":    None,
        "expected_ai_use_case": None,
        "expected_intent":      None,
        "expected_regulations": [],
        "expected_article":     None,
        "expected_keywords":    [],
        "noise": True,
    },
]


def fuzzy_match(expected: str | None, actual: str | None) -> bool:
    """Case-insensitive substring match."""
    if expected is None:
        return True
    if actual is None:
        return False
    return expected.lower() in actual.lower()


def evaluate_profiler(item: dict, profile: dict) -> dict:
    if item["noise"]:
        return {"skipped": True}

    industry_ok     = fuzzy_match(item["expected_industry"], profile.get("industry"))
    ai_use_case_ok  = (
        (item["expected_ai_use_case"] is True  and profile.get("ai_use_case") is not None) or
        (item["expected_ai_use_case"] is False and profile.get("ai_use_case") is None)
    )
    intent_ok       = fuzzy_match(item["expected_intent"], profile.get("intent"))

    return {
        "industry_match":      industry_ok,
        "ai_use_case_correct": ai_use_case_ok,
        "intent_match":        intent_ok,
        "overall":             all([
            industry_ok if item["expected_industry"] else True,
            ai_use_case_ok,
            intent_ok if item["expected_intent"] else True,
        ]),
        "extracted": {
            "industry":    profile.get("industry"),
            "ai_use_case": profile.get("ai_use_case"),
            "intent":      profile.get("intent"),
            "jurisdiction": profile.get("jurisdiction"),
        },
    }


def evaluate_rag(item: dict, pack) -> dict:
    if item["noise"]:
        return {
            "evidence_score":    pack.evidence_score,
            "regulation_hit":    None,
            "article_hit":       None,
            "keyword_hit":       None,
            "web_search_needed": not pack.sufficient,
            "noise_score_ok":    pack.evidence_score < 0.65,
        }

    all_text = " ".join(e.text.lower() for e in pack.evidence)
    covered  = pack.regulations_covered

    regulation_hit = any(r in covered for r in item["expected_regulations"]) if item["expected_regulations"] else True
    article_hit    = (
        item["expected_article"].lower() in all_text
        if item["expected_article"] else True
    )
    keyword_hit    = any(kw.lower() in all_text for kw in item["expected_keywords"]) if item["expected_keywords"] else True

    return {
        "evidence_score":      pack.evidence_score,
        "regulation_hit":      regulation_hit,
        "article_hit":         article_hit,
        "keyword_hit":         keyword_hit,
        "regulations_covered": covered,
        "web_search_needed":   not pack.sufficient,
    }


def run_eval(output_path: Path) -> None:
    print(f"\n{'='*65}")
    print("  Enterprise Regulatory Navigator -- Functional Evaluation")
    print(f"  Queries: {len(EVAL_SET)}  ({sum(1 for q in EVAL_SET if q['noise'])} noise)")
    print(f"{'='*65}\n")

    records = []

    for i, item in enumerate(EVAL_SET, 1):
        tag   = "[NOISE]" if item["noise"] else "       "
        print(f"  [{i:>2}/{len(EVAL_SET)}] {tag} {item['query'][:58]}", flush=True)

        t0 = time.perf_counter()

        # Profiler evaluation (LLM call)
        profiler_state  = {"user_query": item["query"], "conversation_history": []}
        profiler_output = use_case_profiler_node(profiler_state)
        profile         = profiler_output.get("profile", {})
        profiler_result = evaluate_profiler(item, profile)

        # Scope routing check
        routing_state  = {**profiler_output, "user_query": item["query"]}
        route_decision = route_after_profiling(routing_state)

        if item["noise"]:
            # Robustness: router should classify as out_of_scope
            routed_correctly = (route_decision == "out_of_scope")
            rag_result = {"skipped": True, "routed_correctly": routed_correctly}
            elapsed = time.perf_counter() - t0
            result_str = "ok" if routed_correctly else "FAIL"
            print(f"           scope_route={result_str} (got '{route_decision}')  {elapsed:.1f}s")
        else:
            # RAG evaluation (only for in-scope queries)
            pack = run_rag(
                query=item["query"],
                regulations=item["expected_regulations"],
                required_regulations=item["expected_regulations"] or None,
            )
            rag_result = evaluate_rag(item, pack)
            elapsed = time.perf_counter() - t0
            p = profiler_result
            print(
                f"           profiler: ind={'ok' if p['industry_match'] else 'FAIL'} "
                f"ai={'ok' if p['ai_use_case_correct'] else 'FAIL'} "
                f"intent={'ok' if p['intent_match'] else 'FAIL'}  |  "
                f"rag: reg={'ok' if rag_result['regulation_hit'] else 'FAIL'} "
                f"art={'ok' if rag_result['article_hit'] else 'FAIL'} "
                f"kw={'ok' if rag_result['keyword_hit'] else 'FAIL'} "
                f"score={rag_result['evidence_score']:.3f}  {elapsed:.1f}s"
            )

        record = {
            "idx":       i,
            "query":     item["query"],
            "noise":     item["noise"],
            "elapsed_s": round(elapsed, 2),
            "profiler":  profiler_result,
            "rag":       rag_result,
        }
        records.append(record)

    # Aggregate
    in_scope = [r for r in records if not r["noise"]]
    noise    = [r for r in records if r["noise"]]
    n        = len(in_scope)

    prof_industry = [r for r in in_scope if r["profiler"]["industry_match"] and EVAL_SET[r["idx"]-1]["expected_industry"]]
    prof_industry_total = sum(1 for item in EVAL_SET if not item["noise"] and item["expected_industry"])
    prof_industry_ok    = sum(1 for r in in_scope if r["profiler"]["industry_match"] and EVAL_SET[r["idx"]-1]["expected_industry"])

    prof_ai_ok     = sum(1 for r in in_scope if r["profiler"]["ai_use_case_correct"])
    prof_intent_ok = sum(1 for r in in_scope if r["profiler"]["intent_match"] and EVAL_SET[r["idx"]-1]["expected_intent"])
    prof_intent_total = sum(1 for item in EVAL_SET if not item["noise"] and item["expected_intent"])
    prof_overall   = sum(1 for r in in_scope if r["profiler"]["overall"])

    rag_reg_hit   = sum(1 for r in in_scope if r["rag"]["regulation_hit"])
    rag_art_hit   = sum(1 for r in in_scope if r["rag"]["article_hit"])
    rag_kw_hit    = sum(1 for r in in_scope if r["rag"]["keyword_hit"])
    rag_avg_score = sum(r["rag"]["evidence_score"] for r in in_scope) / n
    rag_web_rate  = sum(1 for r in in_scope if r["rag"]["web_search_needed"]) / n
    noise_routed_ok = sum(1 for r in noise if r["rag"].get("routed_correctly", False))

    print(f"\n{'='*65}")
    print("  USE CASE PROFILER EVALUATION  (LLM node, in-scope queries)")
    print(f"{'='*65}")
    print(f"  Overall accuracy    : {prof_overall}/{n}  ({prof_overall/n*100:.0f}%)")
    print(f"  Industry match      : {prof_industry_ok}/{prof_industry_total}  ({prof_industry_ok/prof_industry_total*100:.0f}% of queries with expected industry)")
    print(f"  AI use case correct : {prof_ai_ok}/{n}  ({prof_ai_ok/n*100:.0f}%)")
    print(f"  Intent match        : {prof_intent_ok}/{prof_intent_total}  ({prof_intent_ok/prof_intent_total*100:.0f}% of queries with expected intent)")

    print(f"\n{'='*65}")
    print("  RAG EVALUATION  (in-scope queries)")
    print(f"{'='*65}")
    print(f"  Regulation hit      : {rag_reg_hit}/{n}  ({rag_reg_hit/n*100:.0f}%)")
    print(f"  Article hit         : {rag_art_hit}/{n}  ({rag_art_hit/n*100:.0f}%)")
    print(f"  Keyword hit         : {rag_kw_hit}/{n}  ({rag_kw_hit/n*100:.0f}%)")
    print(f"  Mean evidence score : {rag_avg_score:.3f}")
    print(f"  Web search rate     : {rag_web_rate*100:.0f}%")

    print(f"\n{'='*65}")
    print("  ROBUSTNESS  (noise queries — scope routing)")
    print(f"{'='*65}")
    print(f"  Correctly routed out-of-scope : {noise_routed_ok}/{len(noise)}  ({noise_routed_ok/len(noise)*100:.0f}%)")

    summary = {
        "n_total":    len(records),
        "n_in_scope": n,
        "n_noise":    len(noise),
        "profiler": {
            "overall_accuracy":       round(prof_overall / n, 3),
            "industry_match_rate":    round(prof_industry_ok / prof_industry_total, 3) if prof_industry_total else None,
            "ai_use_case_accuracy":   round(prof_ai_ok / n, 3),
            "intent_match_rate":      round(prof_intent_ok / prof_intent_total, 3) if prof_intent_total else None,
        },
        "rag": {
            "regulation_hit_rate":    round(rag_reg_hit / n, 3),
            "article_hit_rate":       round(rag_art_hit / n, 3),
            "keyword_hit_rate":       round(rag_kw_hit / n, 3),
            "mean_evidence_score":    round(rag_avg_score, 3),
            "web_search_rate":        round(rag_web_rate, 3),
        },
        "robustness": {
            "noise_routed_correctly": round(noise_routed_ok / len(noise), 3),
        },
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Results saved to {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent.parent / "results" / "eval.json")
    args = parser.parse_args()

    run_eval(output_path=args.output)
