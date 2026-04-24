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
        "LANGFUSE_BASE_URL",
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
# Streamlit page setup + CSS polish
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Causal Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Inline CSS — subtle dark polish, smaller typography, sidebar always visible.
# Targets Streamlit's stable data-testid selectors so it survives minor upgrades.
st.markdown(
    """
    <style>
      /* Hide Streamlit chrome (top-right hamburger, footer, top header bar) */
      #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}

      /* Keep the sidebar locked open on every load — disable the collapse button.
         Reviewers consistently miss the chevron and think the controls are gone. */
      [data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapsedControl"] {
          display: none !important;
      }
      [data-testid="stSidebar"][aria-expanded="false"] {
          transform: translateX(0) !important;
          visibility: visible !important;
          min-width: 290px !important;
      }
      [data-testid="stSidebar"] {
          background: rgba(15, 23, 42, 0.7);
          border-right: 1px solid rgba(148, 163, 184, 0.08);
      }
      [data-testid="stSidebar"] h1 {
          font-size: 1.05rem !important;
          font-weight: 600;
          margin-bottom: 0.15rem;
      }

      /* Main content padding */
      [data-testid="stMain"] [data-testid="stMainBlockContainer"] {
          padding-top: 1.5rem;
          padding-bottom: 3rem;
      }

      /* Tighter, smaller typography across the board */
      h1 { font-size: 1.55rem !important; font-weight: 600; letter-spacing: -0.012em; margin: 0.4rem 0 0.6rem; }
      h2 { font-size: 1.25rem !important; font-weight: 600; letter-spacing: -0.012em; margin: 0.8rem 0 0.4rem; }
      h3 { font-size: 1.05rem !important; font-weight: 600; letter-spacing: -0.01em; margin: 0.8rem 0 0.4rem; }
      h4 { font-size: 0.95rem !important; font-weight: 600; margin: 0.6rem 0 0.3rem; }
      h5 { font-size: 0.85rem !important; font-weight: 500; color: rgba(241, 245, 249, 0.75); margin: 0.4rem 0; }

      /* Body text */
      [data-testid="stMarkdownContainer"] p { font-size: 0.92rem; line-height: 1.5; }

      /* Metric tiles — quieter */
      [data-testid="stMetric"] {
          background: rgba(30, 41, 59, 0.4);
          border: 1px solid rgba(148, 163, 184, 0.08);
          border-radius: 10px;
          padding: 0.7rem 0.9rem;
      }
      [data-testid="stMetric"] [data-testid="stMetricLabel"] {
          color: rgba(241, 245, 249, 0.55);
          font-size: 0.7rem;
          letter-spacing: 0.04em;
          text-transform: uppercase;
          font-weight: 500;
      }
      [data-testid="stMetric"] [data-testid="stMetricValue"] {
          font-size: 1.15rem;
          font-weight: 600;
          letter-spacing: -0.01em;
      }
      [data-testid="stMetric"] [data-testid="stMetricDelta"] {
          font-size: 0.75rem;
      }

      /* Bordered containers — subtler */
      [data-testid="stVerticalBlockBorderWrapper"] {
          border-radius: 10px !important;
          border-color: rgba(148, 163, 184, 0.08) !important;
          background: rgba(30, 41, 59, 0.25);
      }

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {
          gap: 0.2rem;
          border-bottom: 1px solid rgba(148, 163, 184, 0.08);
      }
      .stTabs [data-baseweb="tab"] {
          padding: 0.45rem 0.85rem;
          font-size: 0.85rem;
          color: rgba(241, 245, 249, 0.6);
      }
      .stTabs [aria-selected="true"] {
          color: #f1f5f9;
          font-weight: 600;
      }

      /* Buttons — flat & smaller */
      [data-testid="stBaseButton-secondary"],
      [data-testid="stBaseButton-primary"] {
          border-radius: 8px;
          font-weight: 500;
          font-size: 0.85rem;
          padding: 0.35rem 0.85rem;
      }

      /* Chat */
      [data-testid="stChatInput"] { border-radius: 10px; }
      [data-testid="stChatMessage"] {
          background: rgba(30, 41, 59, 0.35);
          border: 1px solid rgba(148, 163, 184, 0.06);
          border-radius: 10px;
      }

      /* Dividers */
      hr {
          border-color: rgba(148, 163, 184, 0.10) !important;
          margin: 1rem 0;
      }

      /* Pill chips for severity / status */
      .pill {
          display: inline-flex;
          align-items: center;
          gap: 0.3rem;
          padding: 0.1rem 0.55rem;
          border-radius: 999px;
          font-size: 0.72rem;
          font-weight: 500;
          letter-spacing: 0.01em;
      }
      .pill-critical { background: rgba(239, 68, 68, 0.13); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.25); }
      .pill-warn     { background: rgba(234, 179, 8, 0.12); color: #fde68a; border: 1px solid rgba(234, 179, 8, 0.25); }
      .pill-info     { background: rgba(59, 130, 246, 0.12); color: #93c5fd; border: 1px solid rgba(59, 130, 246, 0.22); }
      .pill-high     { background: rgba(239, 68, 68, 0.10); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.20); }
      .pill-medium   { background: rgba(234, 179, 8, 0.10); color: #fde68a; border: 1px solid rgba(234, 179, 8, 0.20); }
      .pill-low      { background: rgba(34, 197, 94, 0.10); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.20); }
      .pill-on       { background: rgba(34, 197, 94, 0.12); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.25); }
      .pill-off      { background: rgba(148, 163, 184, 0.10); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.20); }
      .pill code, .mono {
          background: rgba(148, 163, 184, 0.10);
          padding: 0.08rem 0.4rem;
          border-radius: 5px;
          font-size: 0.78rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
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
# Provider-default helpers (used by sidebar AND empty-state quick-launch)
# ---------------------------------------------------------------------------

PROVIDER_OPTIONS = ["groq", "anthropic", "mock"]
MODEL_OPTIONS = {
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-5"],
    "mock": ["mock"],
}


def default_provider() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "mock"


def queue_briefing(portfolio_id: str, *, provider: str, model: str, use_judge: bool) -> None:
    """Stage a regen — picked up at the bottom of the script run."""
    st.session_state.pop("chat_session", None)
    st.session_state["regen"] = True
    st.session_state["pending_portfolio"] = portfolio_id
    st.session_state["pending_provider"] = provider
    st.session_state["pending_model"] = model
    st.session_state["pending_use_judge"] = use_judge


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

loader = get_loader()
tracer = get_tracer()

with st.sidebar:
    st.markdown("# :material/insights: Causal Advisor")
    st.caption("Reasoning-first financial advisor agent.")
    st.markdown("")

    portfolio_options = {
        pid: f"{pid} — {p.user_name}"
        for pid, p in loader.portfolios.items()
    }
    portfolio_id = st.selectbox(
        "Portfolio",
        options=list(portfolio_options.keys()),
        format_func=lambda pid: portfolio_options[pid],
        index=1,  # default to PORTFOLIO_002 (the dramatic one)
    )

    portfolio_obj = loader.portfolios[portfolio_id]
    st.caption(
        f"_{portfolio_obj.portfolio_type.replace('_', ' ').title()} · "
        f"{portfolio_obj.risk_profile.title()} risk_"
    )

    st.divider()

    provider = st.selectbox(
        "LLM provider",
        options=PROVIDER_OPTIONS,
        index=PROVIDER_OPTIONS.index(default_provider()),
        help="'mock' uses a deterministic stub — no API key required.",
    )
    model = st.selectbox("Model", options=MODEL_OPTIONS[provider], index=0)

    use_judge = st.checkbox(
        "Run LLM-as-judge during eval",
        value=(provider != "mock"),
        help="Adds one extra LLM call. Free on Groq.",
    )

    st.divider()

    if st.button(
        ":material/rocket_launch: Generate briefing",
        type="primary",
        use_container_width=True,
    ):
        queue_briefing(portfolio_id, provider=provider, model=model, use_judge=use_judge)

    st.divider()
    if tracer.enabled:
        st.markdown(
            '<span class="pill pill-on">'
            ':material/cloud_done: Tracing on</span>',
            unsafe_allow_html=True,
        )
        st.caption("Langfuse credentials detected.")
    else:
        st.markdown(
            '<span class="pill pill-off">'
            ':material/cloud_off: Tracing off</span>',
            unsafe_allow_html=True,
        )
        st.caption("Set `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` to enable.")


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

    st.session_state["context"] = ctx
    st.session_state["run"] = run
    st.session_state["eval_result"] = eval_result
    st.session_state["client"] = client
    st.session_state["chat_session"] = ChatSession(briefing=run.briefing, context=ctx)
    st.session_state["chat_agent"] = ChatAgent(client=client, tracer=tracer)
    st.session_state["last_provider"] = provider
    st.session_state["last_model"] = model


def _explain_llm_error(exc: Exception) -> tuple[str, str]:
    """Map a low-level LLM exception to a (icon, message) pair for st.error."""
    name = type(exc).__name__
    text = str(exc)
    if "RateLimit" in name or "429" in text:
        return (
            ":material/timer:",
            "**Groq free-tier rate limit hit.**\n\n"
            "Llama 3.3 70B allows ~12,000 tokens/minute, and one briefing + LLM-as-judge "
            "burns through ~14,000 tokens. You can:\n"
            "- **Wait ~30 seconds** and click Generate again, or\n"
            "- Switch to **llama-3.1-8b-instant** in the sidebar (higher rate limit, faster), or\n"
            "- Uncheck **Run LLM-as-judge during eval** to halve token usage, or\n"
            "- Switch the provider to **mock** to demo the pipeline without any LLM call."
        )
    if "Authentication" in name or "401" in text:
        return (
            ":material/key:",
            "**LLM authentication failed.** Check that the API key in your `.env` "
            "(or Streamlit Cloud Secrets) matches the provider you selected.",
        )
    if "BadRequest" in name or "400" in text:
        return (
            ":material/report:",
            f"**Bad request to the LLM:** {text}",
        )
    if "Connection" in name or "Timeout" in name:
        return (
            ":material/wifi_off:",
            "**Network error reaching the LLM provider.** Check your connection and retry.",
        )
    return (
        ":material/error:",
        f"**Failed to generate briefing:** `{name}` — {text}",
    )


if st.session_state.pop("regen", False):
    with st.spinner("Generating briefing… (1–3 s)"):
        try:
            _generate(
                st.session_state["pending_portfolio"],
                st.session_state["pending_provider"],
                st.session_state["pending_model"],
                st.session_state["pending_use_judge"],
            )
        except Exception as exc:  # noqa: BLE001 — UI must always render something
            icon, message = _explain_llm_error(exc)
            st.error(message, icon=icon)
            st.stop()


# ---------------------------------------------------------------------------
# Main panel — empty state
# ---------------------------------------------------------------------------

if "run" not in st.session_state:
    st.markdown("# Causal Advisor")
    st.caption(
        "Reasoning-first financial advisor agent for Indian equity portfolios. "
        "Links macro news → sector → stock → ₹ portfolio impact, flags conflicting "
        "signals, and grades its own output."
    )

    st.info(
        "Pick a portfolio in the sidebar (left), choose a provider, then click "
        "**Generate briefing**. Use **mock** if Groq's free tier is rate-limited.",
        icon=":material/keyboard_arrow_left:",
    )

    st.markdown("##### Sample portfolios")
    for pid, p in loader.portfolios.items():
        with st.container(border=True):
            st.markdown(f"**{pid} — {p.user_name}**")
            st.caption(
                f"{p.description} · "
                f"{len(p.stocks) + len(p.mutual_funds)} holdings · "
                f"{p.risk_profile.title()} risk"
            )

    st.stop()


# ---------------------------------------------------------------------------
# Main panel — briefing view
# ---------------------------------------------------------------------------

ctx = st.session_state["context"]
run = st.session_state["run"]
eval_result = st.session_state["eval_result"]
snap = ctx.portfolio_snapshot

# --- Top action bar (back + rerun + heading) ---
nav_col, title_col, rerun_col = st.columns([1.2, 5, 1.5])
with nav_col:
    if st.button(
        ":material/arrow_back: Portfolios",
        use_container_width=True,
        help="Clear this briefing and return to the portfolio picker.",
    ):
        for _k in ("run", "context", "eval_result", "client",
                   "chat_session", "chat_agent", "last_provider", "last_model"):
            st.session_state.pop(_k, None)
        st.rerun()
with rerun_col:
    if st.button(
        ":material/refresh: Re-run",
        use_container_width=True,
        type="primary",
        help="Regenerate the briefing for this portfolio with the same settings.",
    ):
        queue_briefing(
            snap.portfolio_id,
            provider=st.session_state["last_provider"],
            model=st.session_state["last_model"],
            use_judge=st.session_state.get("pending_use_judge", True),
        )
        st.rerun()

with title_col:
    st.markdown(
        f"### :material/account_balance_wallet: {snap.portfolio_id} · {snap.user_name}"
    )

# Status row (risk · provider · tracing pill) + sidebar hint
tracing_pill = (
    '<span class="pill pill-on">:material/cloud_done: Tracing on</span>'
    if tracer.enabled else
    '<span class="pill pill-off">:material/cloud_off: Tracing off</span>'
)
st.markdown(
    f"<div style='display:flex;gap:0.5rem;align-items:center;flex-wrap:wrap;"
    f"margin-bottom:1rem;'>"
    f"<span class='pill pill-info'>:material/account_circle: {snap.risk_profile.title()}</span>"
    f"<span class='pill pill-info'>:material/bolt: "
    f"{st.session_state['last_provider']} · {st.session_state['last_model']}</span>"
    f"{tracing_pill}"
    f"<span class='pill pill-off'>:material/menu: Sidebar (left) "
    f"to switch provider / model / portfolio</span>"
    f"</div>",
    unsafe_allow_html=True,
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
    "Self-reported",
    f"{run.briefing.confidence:.2f}",
    help=run.briefing.confidence_rationale,
)
col_d.metric(
    "Rule-based",
    f"{run.rule_based_confidence:.2f}",
)
col_e.metric(
    "Eval overall",
    f"{eval_result.overall_score:.2f}",
    help=eval_result.summary or "Hybrid rule + judge across 5 dimensions.",
)

st.divider()

# --- Tabs ---
tab_brief, tab_holdings, tab_risks, tab_eval, tab_context = st.tabs(
    [
        ":material/article: Briefing",
        ":material/pie_chart: Holdings & Allocation",
        ":material/warning: Risks",
        ":material/fact_check: Self-evaluation",
        ":material/code: Raw context",
    ]
)

with tab_brief:
    st.markdown(f"### {run.briefing.headline}")
    st.caption(f"_{run.briefing.confidence_rationale}_")
    st.markdown("#### Causal chain")
    if not run.briefing.causal_chain:
        st.info("No causal links produced.", icon=":material/info:")
    for i, link in enumerate(run.briefing.causal_chain, 1):
        with st.container(border=True):
            st.markdown(f"##### {i}. {link.macro_event}")
            st.markdown(f":material/category: **Sector** &nbsp; {link.sector_impact}")
            st.markdown(f":material/show_chart: **Stocks** &nbsp; {link.stock_impact}")
            st.markdown(f":material/account_balance_wallet: **Portfolio** &nbsp; {link.portfolio_impact}")
            if link.evidence_ids:
                st.caption(
                    ":material/link: Evidence — "
                    + ", ".join(f"`{e}`" for e in link.evidence_ids)
                )

    if run.briefing.conflicts:
        st.markdown("#### Conflicting signals")
        for c in run.briefing.conflicts:
            with st.container(border=True):
                st.markdown(f":material/compare_arrows: **{c.description}**")
                st.markdown(f"_Resolution:_ {c.resolution}")
                if c.evidence_ids:
                    st.caption(
                        ":material/link: Evidence — "
                        + ", ".join(f"`{e}`" for e in c.evidence_ids)
                    )

    if run.briefing.recommendations:
        st.markdown("#### Recommendations")
        priority_class = {"HIGH": "pill-high", "MEDIUM": "pill-medium", "LOW": "pill-low"}
        priority_icon = {
            "HIGH": ":material/priority_high:",
            "MEDIUM": ":material/drag_handle:",
            "LOW": ":material/low_priority:",
        }
        for r in run.briefing.recommendations:
            cls = priority_class.get(r.priority, "pill-info")
            icon = priority_icon.get(r.priority, ":material/circle:")
            st.markdown(
                f'<span class="pill {cls}">{icon} {r.priority}</span> &nbsp; {r.text}',
                unsafe_allow_html=True,
            )

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
                color_discrete_sequence=px.colors.sequential.Greens_r,
            )
            fig.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
            )
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
                color=values,
                color_continuous_scale="Greens",
            )
            fig2.update_layout(
                height=350, margin=dict(l=0, r=0, t=10, b=10),
                yaxis=dict(autorange="reversed"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Day movers")
    g_col, l_col = st.columns(2)
    with g_col:
        st.markdown(":material/trending_up: **Top gainers**")
        if not snap.top_gainers:
            st.caption("(none)")
        for m in snap.top_gainers:
            st.markdown(
                f"`{m.identifier}` &nbsp; **{m.day_change_percent:+.2f}%** &nbsp; "
                f"₹{m.day_change_value:+,.0f} &nbsp; ({m.weight_percent:.1f}% weight)"
            )
    with l_col:
        st.markdown(":material/trending_down: **Top losers**")
        if not snap.top_losers:
            st.caption("(none)")
        for m in snap.top_losers:
            st.markdown(
                f"`{m.identifier}` &nbsp; **{m.day_change_percent:+.2f}%** &nbsp; "
                f"₹{m.day_change_value:+,.0f} &nbsp; ({m.weight_percent:.1f}% weight)"
            )

    st.markdown("#### Asset type")
    asset_df = sorted(snap.asset_allocation.items(), key=lambda kv: -kv[1])
    for k, v in asset_df:
        st.progress(v / 100, text=f"{k.replace('_', ' ').title()} — {v:.1f}%")

with tab_risks:
    if not snap.risks:
        st.success(
            "No risk flags triggered. Portfolio looks balanced.",
            icon=":material/check_circle:",
        )
    else:
        st.caption(f"{len(snap.risks)} flag(s) raised against the threshold rules.")
        sev_class = {
            "CRITICAL": "pill-critical",
            "WARN": "pill-warn",
            "INFO": "pill-info",
        }
        sev_icon = {
            "CRITICAL": ":material/error:",
            "WARN": ":material/warning:",
            "INFO": ":material/info:",
        }
        for r in snap.risks:
            with st.container(border=True):
                cls = sev_class.get(r.severity.value, "pill-info")
                icon = sev_icon.get(r.severity.value, ":material/circle:")
                st.markdown(
                    f'<span class="pill {cls}">{icon} {r.severity.value}</span> '
                    f'&nbsp; <code class="mono">{r.code}</code>',
                    unsafe_allow_html=True,
                )
                st.write(r.message)
                if r.metric is not None:
                    st.caption(f":material/percent: Metric — {r.metric:.2f}")

with tab_eval:
    st.markdown(
        f"### Overall score: **{eval_result.overall_score:.2f}** "
        f"({'rule + judge' if eval_result.judge_used else 'rule-only'})"
    )
    if eval_result.summary:
        st.info(eval_result.summary, icon=":material/lightbulb:")

    dim_names = [d.name for d in eval_result.dimensions]
    combined = [d.combined for d in eval_result.dimensions]
    fig = px.bar(
        x=combined, y=dim_names, orientation="h",
        labels={"x": "Combined score (0–1)", "y": ""},
        range_x=[0, 1],
        text=[f"{v:.2f}" for v in combined],
        color=combined,
        color_continuous_scale="Greens",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

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
    with st.expander(":material/data_object: Reasoning context (full JSON)", expanded=False):
        st.json(ctx.to_dict())
    with st.expander(":material/code: Raw LLM response (XML)", expanded=False):
        st.code(run.briefing.raw_response or "(none)", language="xml")


# ---------------------------------------------------------------------------
# Chat — flavor A briefing-grounded follow-up
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### :material/forum: Ask a follow-up")
st.caption(
    "Questions are answered using the same briefing + context above. "
    "The agent will refuse to invent data outside the context."
)

session: ChatSession = st.session_state["chat_session"]
chat_agent: ChatAgent = st.session_state["chat_agent"]

# Avatars
USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"

# Render existing history
for msg in session.messages:
    avatar = USER_AVATAR if msg.role == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg.role, avatar=avatar):
        st.markdown(msg.content)

# Suggested prompts when chat is empty
if not session.messages:
    st.markdown(":material/lightbulb: **Try one of these:**")
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
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question_to_ask)
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Thinking…"):
            try:
                turn = chat_agent.ask(session, question_to_ask, trace_span=run.trace)
                st.markdown(turn.answer)
            except Exception as exc:  # noqa: BLE001 — chat must never crash the page
                icon, message = _explain_llm_error(exc)
                st.error(message, icon=icon)
