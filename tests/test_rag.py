from regulatory_navigator.rag.subgraph import run_rag
from regulatory_navigator.rag.evidence import format_context, format_citations

pack = run_rag(
    query="What are the obligations for data breach notification?",
    regulations=["GDPR", "NIS2"],
)

print(f"Score:      {pack.evidence_score}")
print(f"Sufficient: {pack.sufficient}")
print(f"Covered:    {pack.regulations_covered}")
print(f"Gaps:       {pack.gaps}")
print()
print(format_context(pack))
print()
print(format_citations(pack))
