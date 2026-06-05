from __future__ import annotations

import time

from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from regulatory_navigator.agent.graph import agent_graph

# Page config

st.set_page_config(
    page_title="Enterprise Regulatory Navigator",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants

NODE_LABELS: dict[str, str] = {
    "intake_parser":                "📥 Parsing query",
    "use_case_profiler":            "🔍 Profiling use case",
    "out_of_scope":                 "🚫 Out of scope",
    "regulation_router":            "🗺️  Routing to regulations",
    "rag":                          "📚 Retrieving evidence (RAG)",
    "evidence_sufficiency":         "⚖️  Checking evidence sufficiency",
    "web_search":                   "🌐 Searching the web",
    "external_evidence_summarizer": "📝 Summarising web evidence",
    "risk_obligation_analyzer":     "⚠️  Analysing risks & obligations",
    "checklist_generator":          "✅ Generating compliance checklist",
    "risk_scorer":                  "📊 Scoring regulatory risk",
    "answer_composer":              "✍️  Composing answer",
    "citation_guardrail_validator": "🔒 Validating citations",
}

REGULATION_COLORS: dict[str, str] = {
    "GDPR":    "#2E86AB",
    "DORA":    "#A23B72",
    "NIS2":    "#F18F01",
    "AI Act":  "#C73E1D",
}

SEVERITY_COLOR: dict[str, str] = {
    "high":    "#C73E1D",
    "medium":  "#F18F01",
    "low":     "#2E86AB",
    "unknown": "#888888",
}


# Helper renderers

def regulation_badge(reg: str) -> str:
    color = REGULATION_COLORS.get(reg, "#555555")
    return (
        f'<span style="background:{color};color:white;padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:bold;margin:2px">{reg}</span>'
    )


def render_regulation_badges(regulations: list[str]) -> None:
    st.markdown(" ".join(regulation_badge(r) for r in regulations), unsafe_allow_html=True)


def render_evidence(evidence: list[dict]) -> None:
    if not evidence:
        st.info("No evidence items.")
        return
    for i, e in enumerate(evidence, 1):
        title = e.get("title") or e.get("source", "Source")
        url   = e.get("url")
        score = e.get("score")
        with st.expander(f"[{i}] {title}" + (f"  —  score: {score:.3f}" if score else ""), expanded=False):
            st.markdown(e.get("text", ""))
            if url:
                st.markdown(f"🔗 [{url}]({url})")


def render_checklist(checklist: list[str]) -> None:
    if not checklist:
        st.info("No checklist items generated.")
        return
    for item in checklist:
        st.markdown(item)


def render_risk_assessment(risk: dict) -> None:
    severity = (risk.get("severity") or "unknown").lower()
    color    = SEVERITY_COLOR.get(severity, "#888")
    st.markdown(
        f'**Overall severity:** <span style="color:{color};font-weight:bold">'
        f'{severity.upper()}</span>',
        unsafe_allow_html=True,
    )
    if risk.get("risks"):
        st.markdown("**Risks identified:**")
        for r in risk["risks"]:
            st.markdown(f"- {r}")
    if risk.get("obligations"):
        st.markdown("**Key obligations:**")
        for o in risk["obligations"]:
            st.markdown(f"- {o}")


def node_step_detail(node_name: str, updates: dict, state: dict) -> str:
    """Return a short detail line shown below each node label during streaming."""
    if node_name == "intake_parser":
        return "Query normalised"

    if node_name == "use_case_profiler":
        p = updates.get("profile", {})
        industry  = (p.get("industry") or "—").title()
        use_case  = (p.get("ai_use_case") or p.get("intent") or "—").replace("_", " ").title()
        return f"Industry: **{industry}** · Use case: **{use_case}**"

    if node_name == "regulation_router":
        regs = updates.get("selected_regulations", [])
        return f"Selected: **{', '.join(regs)}**" if regs else ""

    if node_name == "rag":
        score      = updates.get("evidence_score", 0)
        covered    = updates.get("regulations_covered", [])
        sufficient = updates.get("sufficient", False)
        gaps       = updates.get("evidence_gaps", [])
        missing    = [g.split(":")[-1].strip() for g in gaps if ":" in g]
        flag       = "✅ Sufficient" if sufficient else "⚠️ Insufficient"
        lines      = [
            f"Score: **{score:.3f}** · {flag}",
            "Retrieved **20 chunks** → reranked to **5 chunks**",
        ]
        if covered:
            lines.append(f"Covered: **{', '.join(covered)}**")
        if missing:
            lines.append(f"Missing: **{', '.join(missing)}**")
        return "  \n".join(lines)

    if node_name == "evidence_sufficiency":
        score        = state.get("evidence_score", 0)
        web_required = updates.get("web_search_required", False)
        return f"Score: **{score:.3f}** · {'🌐 Web search required' if web_required else '✅ Evidence sufficient'}"

    if node_name == "web_search":
        n = len(updates.get("external_evidence", []))
        return f"Retrieved **{n}** additional sources"

    if node_name == "external_evidence_summarizer":
        score = updates.get("evidence_score", 0)
        return f"New score: **{score:.3f}**"

    if node_name == "risk_obligation_analyzer":
        risk     = updates.get("risk_assessment", {})
        severity = risk.get("severity", "—")
        n_risks  = len(risk.get("risks", []))
        return f"Risk level: **{severity}** · {n_risks} risk(s) identified"

    if node_name == "checklist_generator":
        n = len(updates.get("checklist", []))
        return f"Generated **{n}** checklist items"

    if node_name == "risk_scorer":
        rs    = updates.get("risk_score", {})
        score = rs.get("composite_score", "—")
        level = rs.get("risk_level", "—")
        return f"Composite score: **{score}** · Level: **{level}**"

    if node_name == "answer_composer":
        n = len(updates.get("final_answer", ""))
        return f"Completed · **{n}** characters"

    if node_name == "citation_guardrail_validator":
        valid = (updates.get("validation_result") or {}).get("valid", True)
        return "✅ Valid" if valid else "⚠️ Issues corrected"

    return ""


def render_trace(trace: list[dict]) -> None:
    for entry in trace:
        node   = entry.get("node", "")
        label  = NODE_LABELS.get(node, node.replace("_", " ").title())
        detail = ""
        if "evidence_score" in entry:
            detail = f"score={entry['evidence_score']}  sufficient={entry.get('sufficient')}"
        elif "new_score" in entry:
            detail = f"new_score={entry['new_score']}"
        elif "severity" in entry:
            detail = f"severity={entry['severity']}  risks={entry['risks']}"
        elif "items" in entry:
            detail = f"items={entry['items']}"
        elif "valid" in entry:
            detail = f"valid={entry['valid']}"
        st.markdown(f"**{label}**" + (f"  —  `{detail}`" if detail else ""))


def render_result(result: dict) -> None:
    """Render the full agent result below the final answer."""
    tabs = st.tabs(["📚 Evidence", "⚠️ Risk & Obligations", "✅ Checklist", "🔒 Validation", "🔎 Trace"])

    with tabs[0]:
        score = result.get("evidence_score")
        if score is not None:
            st.metric("Evidence Score", f"{score:.3f}", help="0–1 composite relevance score")
        covered = result.get("regulations_covered", [])
        if covered:
            st.markdown("**Regulations covered:**")
            render_regulation_badges(covered)
        st.divider()
        render_evidence(result.get("evidence", []))

    with tabs[1]:
        render_risk_assessment(result.get("risk_assessment") or {})

    with tabs[2]:
        render_checklist(result.get("checklist", []))

    with tabs[3]:
        validation = result.get("validation_result") or {}
        valid      = validation.get("valid", True)
        issues     = validation.get("issues", [])
        if valid:
            st.success("✅ Answer validated — no citation issues found.")
        else:
            st.warning("⚠️ Validator found issues and corrected the answer.")
            for issue in issues:
                st.markdown(f"- {issue}")

    with tabs[4]:
        render_trace(result.get("trace", []))


# Sidebar

with st.sidebar:
    st.title("⚖️ Regulatory Navigator")
    st.caption("Agentic RAG · EU Compliance")
    st.divider()

    st.markdown("""
**Covered regulations:**

🇪🇺 &nbsp;**GDPR** — General Data Protection Regulation
🏦 &nbsp;**DORA** — Digital Operational Resilience Act
🔒 &nbsp;**NIS2** — Network & Information Security Directive
🤖 &nbsp;**AI Act** — EU Artificial Intelligence Act
""")

    st.divider()

    st.markdown("""
**How it works:**
1. Your query is profiled for industry & intent
2. Relevant regulations are identified
3. Evidence is retrieved from regulatory PDFs
4. Web search fills any gaps
5. Risks, obligations & checklist are generated
6. Answer is validated against evidence
""")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages             = []
        st.session_state.results              = []
        st.session_state.conversation_history = []
        st.rerun()


# Main — chat interface

st.title("Enterprise Regulatory Navigator")
st.caption("Ask questions about EU regulatory compliance obligations (GDPR · DORA · NIS2 · AI Act)")

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Render chat history
result_idx = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and result_idx < len(st.session_state.results):
            render_result(st.session_state.results[result_idx])
            result_idx += 1

# Chat input
if query := st.chat_input("e.g. What are a bank's obligations under DORA for ICT risk management?"):

    # Display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent with live node streaming
    with st.chat_message("assistant"):
        final_state: dict = {}

        with st.status("🔄 Running regulatory analysis…", expanded=True) as status:
            t_start = time.perf_counter()

            for chunk in agent_graph.stream(
                {
                    "user_query":           query,
                    "conversation_history": st.session_state.conversation_history,
                },
                stream_mode="updates",
            ):
                for node_name, updates in chunk.items():
                    if isinstance(updates, dict):
                        for k, v in updates.items():
                            if k == "trace" and isinstance(v, list):
                                final_state.setdefault("trace", [])
                                final_state["trace"].extend(v)
                            else:
                                final_state[k] = v

                    label  = NODE_LABELS.get(node_name, node_name.replace("_", " ").title())
                    detail = node_step_detail(node_name, updates if isinstance(updates, dict) else {}, final_state)
                    status.write(f"{label}  \n{detail}" if detail else label)

            elapsed = time.perf_counter() - t_start
            status.write(f"⏱ Total execution time: **{elapsed:.1f}s**")
            status.update(label=f"✅ Analysis complete  ({elapsed:.1f}s)", state="complete", expanded=False)

        # Regulation badges
        selected = final_state.get("selected_regulations", [])
        if selected:
            render_regulation_badges(selected)
            st.markdown("")

        # Final answer
        answer = final_state.get("final_answer", "No answer generated.")
        st.markdown(answer)

        # Detailed results in tabs
        render_result(final_state)

    # Persist to session state
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.results.append(final_state)

    # Update conversation history for next turn (keep last 6 messages = 3 turns)
    st.session_state.conversation_history.append({"role": "user",      "content": query})
    st.session_state.conversation_history.append({"role": "assistant", "content": answer[:500]})
    st.session_state.conversation_history = st.session_state.conversation_history[-6:]
