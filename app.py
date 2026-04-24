"""Causal Advisor — Streamlit UI.

Run locally:
    PYTHONPATH=src streamlit run app.py

The app loads its keys from `.env` (or Streamlit Cloud's Secrets UI in deploy).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Allow `streamlit run app.py` without `pip install -e .`
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env if present (local dev).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import plotly.express as px

# Bridge Streamlit Cloud secrets → os.environ so `os.environ.get(...)` calls
# inside the Tracer / clients work identically in dev (.env) and prod (Cloud).
import streamlit as _bootstrap_st  # avoid shadowing the `st` import below
try:
    for _key in (
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ):
        if _key in _bootstrap_st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(_bootstrap_st.secrets[_key])
except (FileNotFoundError, Exception):
    # No secrets.toml — fine in local dev.
    pass

from financial_agent.data_loader import DataLoader
from financial_agent.evaluation import BriefingEvaluator
from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.observability import Tracer
from financial_agent.portfolio import PortfolioAnalyzer
from financial_agent.reasoning import (
    AnthropicClient,
    ChatAgent,
    ChatSession,
    GroqClient,
    MockLLMClient,
    ReasoningAgent,
    build_context,
)

DATA_DIR = ROOT / "data"


# ---------------------------------------------------------------------------
# Streamlit page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Causal Advisor",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resources (singletons across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_loader() -> DataLoader:
    return DataLoader(DATA_DIR)


@st.cache_resource(show_spinner=False)
def get_tracer() -> Tracer:
    return Tracer()


def make_client(provider: str, model: str | None):
    """Build a fresh LLM client (NOT cached — provider can change between runs)."""
    if provider == "groq":
        kwargs = {"model": model} if model else {}
        return GroqClient(**kwargs)
    if provider == "anthropic":
        kwargs = {"model": model} if model else {}
        return AnthropicClient(**kwargs)
    return None  # mock — built per-context below


def make_context(loader: DataLoader, portfolio_id: str):
    portfolio = loader.get_portfolio(portfolio_id)
    return build_context(
        portfolio,
        market_analyzer=MarketAnalyzer(loader.market),
        sector_analyzer=SectorAnalyzer(loader.market),
        news_processor=NewsProcessor(loader.news),
        portfolio_analyzer=PortfolioAnalyzer(
            portfolio,
            mutual_funds=loader.mutual_funds,
            rate_sensitive_sectors=loader.sector_map.rate_sensitive_sectors,
        ),
        macro_correlations=loader.sector_map.macro_correlations.correlations,
    )


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

loader = get_loader()
tracer = get_tracer()

with st.sidebar:
    st.title("💹 Causal Advisor")
    st.caption("Reasoning-first financial advisor agent.")

    portfolio_options = {
        pid: f"{pid} — {p.user_name} ({p.portfolio_type.replace('_', ' ').title()})"
        for pid, p in loader.portfolios.items()
    }
    portfolio_id = st.selectbox(
        "Portfolio",
        options=list(portfolio_options.keys()),
        format_func=lambda pid: portfolio_options[pid],
        index=1,  # default to PORTFOLIO_002 (the dramatic one)
    )

    st.divider()

    provider_choice = "groq" if os.environ.get("GROQ_API_KEY") else (
        "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock"
    )
    provider = st.selectbox(
        "LLM provider",
        options=["groq", "anthropic", "mock"],
        index=["groq", "anthropic", "mock"].index(provider_choice),
        help="'mock' uses a deterministic stub — no API key required.",
    )

    model_overrides = {
        "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        "anthropic": ["claude-sonnet-4-5", "claude-opus-4-5"],
        "mock": ["mock"],
    }
    model = st.selectbox("Model", options=model_overrides[provider], index=0)

    use_judge = st.checkbox(
        "Run LLM-as-judge during eval",
        value=(provider != "mock"),
        help="Adds one extra LLM call. Free on Groq.",
    )

    st.divider()

    if st.button("Generate briefing", type="primary", use_container_width=True):
        # Reset chat when generating a new briefing.
        st.session_state.pop("chat_session", None)
        st.session_state["regen"] = True
        st.session_state["pending_portfolio"] = portfolio_id
        st.session_state["pending_provider"] = provider
        st.session_state["pending_model"] = model
        st.session_state["pending_use_judge"] = use_judge

    # Status footer
    st.divider()
    st.caption(f"Tracing: {'🟢 on' if tracer.enabled else '⚪ off'}")
    st.caption("Set LANGFUSE_* in .env to enable.")


# ---------------------------------------------------------------------------
# Generate briefing on demand
# ---------------------------------------------------------------------------

def _generate(portfolio_id: str, provider: str, model: str, use_judge: bool):
    ctx = make_context(loader, portfolio_id)
    if provider == "mock":
        client = MockLLMClient.from_context(ctx)
    else:
        client = make_client(provider, model)

    agent = ReasoningAgent(client=client, tracer=tracer)
    run = agent.generate(ctx)

    judge_client = client if (use_judge and provider != "mock") else None
    evaluator = BriefingEvaluator(judge_client=judge_client, tracer=tracer)
    eval_result = evaluator.score(run.briefing, ctx, trace_span=run.trace)

    # Persist for chat + tabs.
    st.session_state["context"] = ctx
    st.session_state["run"] = run
    st.session_state["eval_result"] = eval_result
    st.session_state["client"] = client
    st.session_state["chat_session"] = ChatSession(briefing=run.briefing, context=ctx)
    st.session_state["chat_agent"] = ChatAgent(client=client, tracer=tracer)
    st.session_state["last_provider"] = provider
    st.session_state["last_model"] = model


if st.session_state.pop("regen", False):
    with st.spinner("Generating briefing… (1–3 s)"):
        try:
            _generate(
                st.session_state["pending_portfolio"],
                st.session_state["pending_provider"],
                st.session_state["pending_model"],
                st.session_state["pending_use_judge"],
            )
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()


# ---------------------------------------------------------------------------
# Main panel — empty state
# ---------------------------------------------------------------------------

if "run" not in st.session_state:
    st.title("Causal Advisor")
    st.markdown(
        """
Pick a portfolio in the sidebar and click **Generate briefing**.

Three sample portfolios from the assignment dataset are loaded:

- **PORTFOLIO_001** — Diversified (Rahul Sharma, well-balanced)
- **PORTFOLIO_002** — Sector-concentrated (Priya Patel, **91% banking + FS**)
- **PORTFOLIO_003** — Conservative (Arun Krishnamurthy, mutual-fund heavy)

Each briefing produces:
1. A **causal chain** linking macro news → sector → stock → ₹ portfolio impact
2. **Conflict resolution** for divergent signals (e.g., positive news + falling price)
3. A **5-dimension self-evaluation** with rule-based + LLM-as-judge scores

You can then ask follow-up questions in the chat box that appears at the bottom.
"""
    )
    st.stop()


# ---------------------------------------------------------------------------
# Main panel — briefing view
# ---------------------------------------------------------------------------

ctx = st.session_state["context"]
run = st.session_state["run"]
eval_result = st.session_state["eval_result"]
snap = ctx.portfolio_snapshot

# --- Header ---
st.title(f"💹 {snap.portfolio_id} — {snap.user_name}")
st.caption(
    f"{snap.risk_profile.title()} risk profile · "
    f"Provider: **{st.session_state['last_provider']}** "
    f"({st.session_state['last_model']}) · "
    f"Tracing: {'🟢 on' if tracer.enabled else '⚪ off'}"
)

# --- Metric tiles ---
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric(
    "Current value",
    f"₹{snap.current_value:,.0f}",
    f"₹{snap.overall_pnl:+,.0f} lifetime",
)
col_b.metric(
    "Day P&L",
    f"₹{snap.day_pnl:+,.0f}",
    f"{snap.day_pnl_percent:+.2f}%",
    delta_color="normal",
)
col_c.metric(
    "Self-reported confidence",
    f"{run.briefing.confidence:.2f}",
    help=run.briefing.confidence_rationale,
)
col_d.metric(
    "Rule-based confidence",
    f"{run.rule_based_confidence:.2f}",
)
col_e.metric(
    "Eval overall",
    f"{eval_result.overall_score:.2f}",
    help=eval_result.summary or "Hybrid rule + judge score across 5 dimensions.",
)

st.divider()

# --- Tabs ---
tab_brief, tab_holdings, tab_risks, tab_eval, tab_context = st.tabs(
    ["📝 Briefing", "📊 Holdings & Allocation", "⚠️ Risks", "🧪 Self-evaluation", "🔍 Raw context"]
)

with tab_brief:
    st.markdown(f"### {run.briefing.headline}")
    st.caption(f"_{run.briefing.confidence_rationale}_")
    st.markdown("#### Causal chain")
    if not run.briefing.causal_chain:
        st.info("No causal links produced.")
    for i, link in enumerate(run.briefing.causal_chain, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {link.macro_event}**")
            st.markdown(f"- **Sector:** {link.sector_impact}")
            st.markdown(f"- **Stocks:** {link.stock_impact}")
            st.markdown(f"- **Portfolio:** {link.portfolio_impact}")
            if link.evidence_ids:
                st.caption("Evidence: " + ", ".join(link.evidence_ids))

    if run.briefing.conflicts:
        st.markdown("#### Conflicting signals")
        for c in run.briefing.conflicts:
            with st.container(border=True):
                st.markdown(f"**{c.description}**")
                st.markdown(f"_Resolution:_ {c.resolution}")
                if c.evidence_ids:
                    st.caption("Evidence: " + ", ".join(c.evidence_ids))

    if run.briefing.recommendations:
        st.markdown("#### Recommendations")
        priority_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
        for r in run.briefing.recommendations:
            st.markdown(f"{priority_color.get(r.priority, '•')} **{r.priority}** — {r.text}")

with tab_holdings:
    left, right = st.columns(2)
    with left:
        st.markdown("#### Sector allocation (raw bucket)")
        sector_data = sorted(
            snap.sector_allocation.items(), key=lambda kv: -kv[1]
        )
        if sector_data:
            labels, values = zip(*sector_data)
            fig = px.pie(
                names=labels, values=values, hole=0.55,
            )
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("#### Sector allocation (look-through)")
        st.caption("Mutual-fund holdings decomposed into underlying sectors.")
        look_data = sorted(
            snap.sector_allocation_lookthrough.items(), key=lambda kv: -kv[1]
        )[:10]
        if look_data:
            labels, values = zip(*look_data)
            fig2 = px.bar(
                x=values, y=labels, orientation="h",
                labels={"x": "Weight (%)", "y": ""},
            )
            fig2.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=10),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Day movers")
    g_col, l_col = st.columns(2)
    with g_col:
        st.markdown("**Top gainers**")
        if not snap.top_gainers:
            st.caption("(none)")
        for m in snap.top_gainers:
            st.write(
                f"`{m.identifier}` {m.day_change_percent:+.2f}% — "
                f"₹{m.day_change_value:+,.0f} ({m.weight_percent:.1f}% weight)"
            )
    with l_col:
        st.markdown("**Top losers**")
        if not snap.top_losers:
            st.caption("(none)")
        for m in snap.top_losers:
            st.write(
                f"`{m.identifier}` {m.day_change_percent:+.2f}% — "
                f"₹{m.day_change_value:+,.0f} ({m.weight_percent:.1f}% weight)"
            )

    st.markdown("#### Asset type")
    asset_df = sorted(snap.asset_allocation.items(), key=lambda kv: -kv[1])
    for k, v in asset_df:
        st.progress(v / 100, text=f"{k.replace('_', ' ').title()} — {v:.1f}%")

with tab_risks:
    if not snap.risks:
        st.success("No risk flags triggered. Portfolio looks balanced.")
    else:
        st.caption(f"{len(snap.risks)} flag(s) raised against the threshold rules.")
        sev_emoji = {"CRITICAL": "🔴", "WARN": "🟡", "INFO": "🔵"}
        for r in snap.risks:
            with st.container(border=True):
                st.markdown(
                    f"{sev_emoji.get(r.severity.value, '•')} "
                    f"**{r.severity.value}** — `{r.code}`"
                )
                st.write(r.message)
                if r.metric is not None:
                    st.caption(f"Metric: {r.metric:.2f}")

with tab_eval:
    st.markdown(
        f"### Overall score: **{eval_result.overall_score:.2f}** "
        f"({'rule + judge' if eval_result.judge_used else 'rule-only'})"
    )
    if eval_result.summary:
        st.info(eval_result.summary)

    # Combined-score bar chart
    dim_names = [d.name for d in eval_result.dimensions]
    combined = [d.combined for d in eval_result.dimensions]
    fig = px.bar(
        x=combined, y=dim_names, orientation="h",
        labels={"x": "Combined score (0–1)", "y": ""},
        range_x=[0, 1],
        text=[f"{v:.2f}" for v in combined],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-dimension table
    rows = []
    for d in eval_result.dimensions:
        rows.append({
            "Dimension": d.name,
            "Weight": f"{d.weight:.0%}",
            "Rule": f"{d.rule_score:.2f}",
            "Judge": "—" if d.judge_score is None else f"{d.judge_score:.2f}",
            "Combined": f"{d.combined:.2f}",
            "Critique": d.judge_critique or d.rule_critique,
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

with tab_context:
    st.caption(
        "The exact JSON the LLM saw. Every figure in the briefing must trace back to this."
    )
    with st.expander("Reasoning context (full JSON)", expanded=False):
        st.json(ctx.to_dict())
    with st.expander("Raw LLM response (XML)", expanded=False):
        st.code(run.briefing.raw_response or "(none)", language="xml")


# ---------------------------------------------------------------------------
# Chat — flavor A briefing-grounded follow-up
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### 💬 Ask a follow-up")
st.caption(
    "Questions are answered using the same briefing + context above. "
    "The agent will refuse to invent data outside the context."
)

session: ChatSession = st.session_state["chat_session"]
chat_agent: ChatAgent = st.session_state["chat_agent"]

# Render existing history
for msg in session.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# Suggested prompts when chat is empty
if not session.messages:
    st.markdown("**Try one of these:**")
    cols = st.columns(3)
    suggestions = [
        "What was the single largest contributor to today's loss?",
        "How exposed am I to interest-rate moves?",
        "If RBI cut rates next week, which holdings would benefit?",
    ]
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True):
            st.session_state["pending_question"] = suggestion
            st.rerun()

# Resolve a queued suggestion (set by a button above)
question_to_ask = st.session_state.pop("pending_question", None) or st.chat_input(
    "Ask anything about this briefing…"
)

if question_to_ask:
    with st.chat_message("user"):
        st.markdown(question_to_ask)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                turn = chat_agent.ask(session, question_to_ask, trace_span=run.trace)
                st.markdown(turn.answer)
            except Exception as exc:
                st.error(f"Chat call failed: {exc}")
