import logging
import pprint
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from regulatory_navigator.agent.graph import agent_graph

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# Mermaid graph

mermaid = agent_graph.get_graph().draw_mermaid()
print("\n--- MERMAID GRAPH ---")
print(mermaid)
output = Path(__file__).parent.parent / "graph.mmd"
output.write_text(mermaid)
print(f"(saved to {output})")

# Query

query = "A bank wants to deploy a GenAI chatbot for customer complaints. What are the regulatory obligations?"

print("=" * 70)
print(f"QUERY: {query}")
print("=" * 70)

# Run

t0     = time.perf_counter()
result = agent_graph.invoke({"user_query": query})
elapsed = time.perf_counter() - t0

# Output

print(f"\n Inference time: {elapsed:.2f}s")

print("\n--- PROFILE ---")
pprint.pprint(result.get("profile"))

print("\n--- SELECTED REGULATIONS ---")
print(result.get("selected_regulations"))

print("\n--- EVIDENCE SCORE ---")
print(result.get("evidence_score"))

print("\n--- EVIDENCE GAPS ---")
print(result.get("evidence_gaps"))

print("\n--- RISK ASSESSMENT ---")
pprint.pprint(result.get("risk_assessment"))

print("\n--- CHECKLIST ---")
for item in result.get("checklist", []):
    print(item)

print("\n--- FINAL ANSWER ---")
print(result.get("final_answer"))

print("\n--- VALIDATION ---")
pprint.pprint(result.get("validation_result"))

print("\n--- TRACE ---")
for entry in result.get("trace", []):
    node = entry["node"]
    if "evidence_score" in entry:
        detail = f"score={entry['evidence_score']}  sufficient={entry.get('sufficient', '-')}"
    elif "new_score" in entry:
        detail = f"new_score={entry['new_score']}"
    elif "profile" in entry:
        detail = str(entry["profile"])
    elif "selected" in entry:
        detail = f"selected={entry['selected']}"
    elif "severity" in entry:
        detail = f"severity={entry['severity']}  risks={entry['risks']}  obligations={entry['obligations']}"
    elif "items" in entry:
        detail = f"items={entry['items']}"
    elif "length" in entry:
        detail = f"length={entry['length']}"
    elif "valid" in entry:
        detail = f"valid={entry['valid']}  issues={entry['issues']}"
    elif "queries" in entry:
        detail = f"queries={entry['queries']}  hits={entry['results']}"
    elif "summary_length" in entry:
        detail = f"summary_length={entry['summary_length']}"
    else:
        detail = "-"
    print(f"  [{node}]  {detail}")
