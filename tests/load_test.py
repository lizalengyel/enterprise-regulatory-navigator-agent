"""
Load test for the Enterprise Regulatory Navigator agent.

Setup:
    OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_QUEUE=512 OLLAMA_KEEP_ALIVE=24h ollama serve

Usage:
    uv run python load_test.py                           # 50 queries, 1 worker
    uv run python load_test.py --repeat 50 --workers 4  # 50 queries, 4 concurrent
    uv run python load_test.py --repeat 10 --workers 1  # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.disable(logging.CRITICAL)

from regulatory_navigator.agent.graph import agent_graph
from regulatory_navigator.rag.embeddings import embed_texts
from regulatory_navigator.rag.reranker import get_reranker

# 10 queries covering all regulation areas — each repeated --repeat times
# Default: --repeat 5  →  10 × 5 = 50 total requests
QUERIES = [
    # GDPR
    "A company suffers a personal data breach. What are the notification obligations?",
    "What rights does a data subject have under GDPR?",
    # NIS2
    "A telecom operator suffers a major cyber incident. What obligations arise under NIS2?",
    "What cybersecurity risk management measures are required by NIS2?",
    # DORA
    "A bank outsources critical services to a cloud provider. What DORA obligations apply?",
    "What incident reporting obligations exist under DORA?",
    # AI Act
    "A bank wants to use AI to assess loan applications. Is this considered a high-risk AI system?",
    "What transparency obligations apply to generative AI under the AI Act?",
    # Cross-regulation
    "A bank wants to deploy a GenAI chatbot for customer complaints. What regulatory obligations apply?",
    "A hospital uses AI for patient triage. What regulatory requirements apply?",
]

MAX_RETRIES = 4
RETRY_BASE  = 2.0


# Helpers

def percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    s   = sorted(data)
    idx = (p / 100) * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def node_durations(trace: list[dict]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for i, entry in enumerate(trace[:-1]):
        node     = entry["node"]
        duration = trace[i + 1]["timestamp"] - entry["timestamp"]
        durations[node] = durations.get(node, 0.0) + duration
    return durations


# Single query runner

def run_query(idx: int, query: str) -> dict:
    t0 = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            state   = agent_graph.invoke({"user_query": query, "conversation_history": []})
            elapsed = time.perf_counter() - t0
            trace   = state.get("trace", [])
            return {
                "idx":            idx,
                "query":          query,
                "total_s":        round(elapsed, 3),
                "attempts":       attempt + 1,
                "evidence_score": state.get("evidence_score"),
                "web_search":     any(e["node"] == "web_search" for e in trace),
                "node_durations": {k: round(v, 3) for k, v in node_durations(trace).items()},
                "error":          None,
            }
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                time.sleep(RETRY_BASE ** (attempt + 1))
            else:
                break

    return {
        "idx":      idx,
        "query":    query,
        "total_s":  round(time.perf_counter() - t0, 3),
        "attempts": MAX_RETRIES,
        "error":    str(last_exc),
    }


def run_load_test(repeat: int, workers: int, output_path: Path) -> None:
    n_queries = len(QUERIES) * repeat

    print(f"\n{'='*65}")
    print(f"  Enterprise Regulatory Navigator -- Load Test")
    print(f"  Queries: {len(QUERIES)} x {repeat} repeats = {n_queries} total")
    print(f"  Workers: {workers}")
    print(f"  Model  : {os.getenv('LLM_MODEL', 'qwen3:1.7b')}   Output: {output_path}")
    print(f"{'='*65}")

    print("\n  Pre-warming models ...", end="", flush=True)
    embed_texts(["warm-up"])
    get_reranker()
    print(" done\n")

    records:      list[dict] = []
    completed:    int        = 0
    t_wall_start = time.perf_counter()
    query_list   = [q for q in QUERIES for _ in range(repeat)]

    if workers == 1:
        for i, q in enumerate(query_list):
            print(f"  [{i+1:>3}/{n_queries}] {q[:55]:<55}", end="  ", flush=True)
            record = run_query(i + 1, q)
            records.append(record)
            status = "ERR" if record.get("error") else ("WEB" if record.get("web_search") else "OK")
            print(f"{record['total_s']:>7.1f}s  {status}")
    else:
        print(f"  Submitting {n_queries} queries across {workers} workers ...\n")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_query, i + 1, q): i for i, q in enumerate(query_list)}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed += 1
                status = "ERR" if record.get("error") else ("WEB" if record.get("web_search") else "OK")
                print(
                    f"  [{completed:>3}/{n_queries}] "
                    f"{record['query'][:50]:<50}  "
                    f"{record['total_s']:>7.1f}s  {status}"
                )

    wall_elapsed = time.perf_counter() - t_wall_start

    # Aggregate

    successful      = [r for r in records if not r.get("error")]
    errors          = len(records) - len(successful)
    total_times     = [r["total_s"] for r in successful]

    first_error = next((r for r in records if r.get("error")), None)
    if first_error:
        print(f"\n  Sample error: {first_error['error'][:200]}")

    if not total_times:
        print("\n  All queries failed -- check Ollama status.")
        return

    throughput      = len(successful) / wall_elapsed
    web_search_rate = sum(1 for r in successful if r.get("web_search")) / len(successful)

    all_node_times: dict[str, list[float]] = {}
    for r in successful:
        for node, t in r.get("node_durations", {}).items():
            all_node_times.setdefault(node, []).append(t)

    node_means      = {n: statistics.mean(ts) for n, ts in all_node_times.items()}
    top_bottlenecks = sorted(node_means.items(), key=lambda x: -x[1])

    print(f"\n{'='*65}")
    print("  LATENCY SUMMARY")
    print(f"{'='*65}")
    print(f"  Successful / Total : {len(successful)} / {n_queries}  (errors: {errors})")
    print(f"  Workers            : {workers}")
    print(f"  Wall-clock time    : {wall_elapsed:.1f}s")
    print(f"  Throughput         : {throughput:.3f} queries/s")
    print(f"  Web search rate    : {web_search_rate*100:.0f}%")
    print()
    print(f"  {'Metric':<10}  {'Value':>8}")
    print(f"  {'-'*22}")
    print(f"  {'Mean':<10}  {statistics.mean(total_times):>7.1f}s")
    print(f"  {'Median':<10}  {statistics.median(total_times):>7.1f}s")
    print(f"  {'p75':<10}  {percentile(total_times, 75):>7.1f}s")
    print(f"  {'p90':<10}  {percentile(total_times, 90):>7.1f}s")
    print(f"  {'p95':<10}  {percentile(total_times, 95):>7.1f}s")
    print(f"  {'p99':<10}  {percentile(total_times, 99):>7.1f}s")
    print(f"  {'Min':<10}  {min(total_times):>7.1f}s")
    print(f"  {'Max':<10}  {max(total_times):>7.1f}s")
    print(f"  {'StdDev':<10}  {statistics.stdev(total_times) if len(total_times) > 1 else 0:>7.1f}s")

    print(f"\n{'='*65}")
    print("  PER-NODE MEAN LATENCY  (bottleneck ranking)")
    print(f"{'='*65}")
    for node, mean_t in top_bottlenecks:
        print(f"  {node:<40} {mean_t:>7.2f}s")

    print(f"\n{'='*65}")
    print("  BOTTLENECK ANALYSIS")
    print(f"{'='*65}")
    if top_bottlenecks:
        b1_node, b1_time = top_bottlenecks[0]
        b2_node, b2_time = top_bottlenecks[1] if len(top_bottlenecks) > 1 else ("--", 0)
        print(f"\n  Primary bottleneck   : {b1_node} ({b1_time:.2f}s avg)")
        print(f"  Secondary bottleneck : {b2_node} ({b2_time:.2f}s avg)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "n_queries":     n_queries,
            "repeat":        repeat,
            "workers":       workers,
            "n_success":     len(successful),
            "n_errors":      errors,
        },
        "wall_s":          round(wall_elapsed, 3),
        "throughput_qps":  round(throughput, 3),
        "web_search_rate": round(web_search_rate, 3),
        "latency": {
            "mean_s":   round(statistics.mean(total_times), 3),
            "median_s": round(statistics.median(total_times), 3),
            "p75_s":    round(percentile(total_times, 75), 3),
            "p90_s":    round(percentile(total_times, 90), 3),
            "p95_s":    round(percentile(total_times, 95), 3),
            "p99_s":    round(percentile(total_times, 99), 3),
            "min_s":    round(min(total_times), 3),
            "max_s":    round(max(total_times), 3),
            "stddev_s": round(statistics.stdev(total_times) if len(total_times) > 1 else 0, 3),
        },
        "bottlenecks": [{"node": n, "mean_s": round(t, 3)} for n, t in top_bottlenecks],
        "records":     records,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"  Results saved to {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test for the Regulatory Navigator agent.")
    parser.add_argument("--repeat",  type=int,  default=5,    help="Repeats per query (default: 5 -> 10x5=50 total)")
    parser.add_argument("--workers", type=int,  default=1,    help="Concurrent workers (default: 1)")
    parser.add_argument("--output",  type=Path, default=Path(__file__).parent.parent / "results" / "load_test.json")
    args = parser.parse_args()

    run_load_test(repeat=args.repeat, workers=args.workers, output_path=args.output)
