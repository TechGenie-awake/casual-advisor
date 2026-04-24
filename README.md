# Causal Advisor

> An autonomous financial-advisor agent for Indian equity portfolios that **reasons in causal chains**, not data dumps.
> Given a portfolio, it links *macro news → sector trends → individual stocks → portfolio impact* with quantified ₹ contributions, flags conflicting signals (positive news vs. negative price action), and grades its own output against a 5-dimension rubric.

**[▶ Live app — casual-advisor.streamlit.app](https://casual-advisor.streamlit.app/)** &nbsp; · &nbsp; **[GitHub](https://github.com/TechGenie-awake/casual-advisor)**

Built for the AI Engineering Challenge brief in [Agent Assignment/](Agent%20Assignment/).

---

## TL;DR

```bash
git clone https://github.com/TechGenie-awake/casual-advisor.git
cd casual-advisor
pip install -e .
cp .env.example .env   # paste your free Groq key from https://console.groq.com/keys
PYTHONPATH=src python3 scripts/run_briefing.py PORTFOLIO_002
```

You'll see a quantified causal-chain briefing for a banking-heavy portfolio, followed by a 5-dimension self-evaluation. Total cost: **₹0** (Groq free tier).

A live Streamlit demo is linked at the bottom of this README.

---

## What it does

For each of the three sample portfolios:

1. **Phase 1 — Market Intelligence**: classifies index sentiment (NIFTY 50 -1.00% → BEARISH), recomputes sector trends from constituent stocks (BANKING -3.11% weighted), and ranks news by impact × scope × sentiment magnitude.
2. **Phase 2 — Portfolio Analytics**: recomputes daily P&L, asset/sector allocation (with mutual-fund look-through), and flags concentration risks against threshold rules.
3. **Phase 3 — Reasoning**: builds a *minimal relevant context* (only news / sectors / divergences that touch the portfolio's exposure), pre-computes ₹ contribution attribution, and produces a structured `<briefing>` with a 2-4 link causal chain, conflict notes, and a confidence score.
4. **Phase 4 — Observability + Self-evaluation**: every LLM call is traced to Langfuse. Each briefing is scored on 5 dimensions (Causal Depth, Evidence Accuracy, Conflict Handling, Prioritization, Quantification) using a hybrid of deterministic rule-based scorers + an LLM-as-judge.

The reasoning isn't "the LLM thinks": every quantitative claim in the output (e.g. *"Banking holdings drove ₹46,850 of the ₹57,390 day loss (81.6%)"*) is anchored to a number that Phase 2 pre-computed. The LLM's job is narrative; the math happens before the LLM is even called.

---

## Architecture

```
                   ┌──────────────────────────────────────────┐
                   │   data/  (6 mock JSON files)             │
                   │   market, news, portfolios, MFs,         │
                   │   sectors, history                       │
                   └────────────────┬─────────────────────────┘
                                    │
                                    ▼
                   ┌──────────────────────────────────────────┐
                   │   DataLoader  (pydantic-typed accessors) │
                   └────────────────┬─────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌────────────────────┐  ┌────────────────┐  ┌────────────────────┐
   │  Phase 1           │  │  Phase 2       │  │  Reference data    │
   │  MarketAnalyzer    │  │  Portfolio     │  │  Sector taxonomy,  │
   │  SectorAnalyzer    │  │  Analyzer      │  │  macro correl.     │
   │  NewsProcessor     │  │  (P&L, alloc,  │  │                    │
   │                    │  │   risk flags)  │  │                    │
   └─────────┬──────────┘  └────────┬───────┘  └─────────┬──────────┘
             │                      │                    │
             └────────────┬─────────┴────────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │  Phase 3                     │
             │  build_context()             │
             │  pre-filter to held sectors, │
             │  compute ₹ attribution,      │
             │  filter macro correlations   │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │  ReasoningAgent              │
             │  prompt → LLM (Groq /        │
             │  Anthropic / Mock) → parse   │
             │  → rule-based confidence     │
             │  reconciliation              │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │  Phase 4                     │
             │  BriefingEvaluator           │
             │  rule scorers + LLM-as-judge │
             │  → 5-dim scores + overall    │
             └────────────┬─────────────────┘
                          ▼
             ┌──────────────────────────────┐
             │  CLI / Streamlit UI          │
             │  + Langfuse Tracer (optional)│
             └──────────────────────────────┘
```

**Modularity claim, in code:**

| Concern | Where it lives |
|---|---|
| Data ingestion | [`src/financial_agent/data_loader.py`](src/financial_agent/data_loader.py) |
| Typed entities | [`src/financial_agent/models/`](src/financial_agent/models/) |
| Market intelligence | [`src/financial_agent/market/`](src/financial_agent/market/) |
| Portfolio analytics | [`src/financial_agent/portfolio/`](src/financial_agent/portfolio/) |
| Reasoning + LLM clients | [`src/financial_agent/reasoning/`](src/financial_agent/reasoning/) |
| Observability | [`src/financial_agent/observability/`](src/financial_agent/observability/) |
| Self-evaluation | [`src/financial_agent/evaluation/`](src/financial_agent/evaluation/) |

Every module has a focused responsibility and is wired through dependency injection — no globals, no implicit state, no framework lock-in. Swapping Groq for Anthropic is a one-line config change ([`scripts/run_briefing.py:48`](scripts/run_briefing.py)).

---

## How each rubric line is met

| Rubric criterion (weight) | Where it's earned |
|---|---|
| **Reasoning Quality (35%)** | Causal chain enforced by XML schema in [`prompts.py`](src/financial_agent/reasoning/prompts.py); ₹ figures pre-computed in [`context.py:_compute_attribution`](src/financial_agent/reasoning/context.py); macro mechanism cited via [`sector_mapping.json`](data/sector_mapping.json) `macro_correlations`; few-shot example anchors quality bar; `<thinking>` block forces enumeration before output |
| **Code Design (20%)** | One module per concern, pydantic models everywhere, type hints throughout, `Protocol`-based `BaseLLMClient` interface for trivial provider swap, dataclasses for output models, no globals |
| **Observability (15%)** | Langfuse v3 wrapped in [`observability/tracer.py`](src/financial_agent/observability/tracer.py); auto-detects keys from env, gracefully no-ops when absent. Every briefing produces a parent span + nested generation event + 7 attached scores |
| **Edge Case Handling (15%)** | Stock-vs-sector divergences detected in [`SectorAnalyzer.divergences`](src/financial_agent/market/sectors.py); conflict news identified in [`NewsProcessor.conflict_candidates`](src/financial_agent/market/news.py); both surfaced in the LLM context; system prompt requires explicit `<conflict>` reconciliation; rule scorer **penalizes** unaddressed conflicts |
| **Evaluation Layer (15%)** | [`BriefingEvaluator`](src/financial_agent/evaluation/evaluator.py) scores 5 dimensions with rule-based scorers + LLM-as-judge, blended `0.6 × rule + 0.4 × judge` per dimension. Degrades to rule-only on judge failure. All scores pushed to Langfuse |

---

## Quick start

### Prerequisites
- Python 3.11+
- A free Groq API key (no credit card): https://console.groq.com/keys
- *Optional:* a free Langfuse account for tracing: https://cloud.langfuse.com

### Install

```bash
git clone https://github.com/TechGenie-awake/casual-advisor.git
cd casual-advisor
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure secrets

```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
# Optionally paste LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY for tracing
```

### Run a briefing (CLI)

```bash
PYTHONPATH=src python3 scripts/run_briefing.py PORTFOLIO_002
```

Other portfolios: `PORTFOLIO_001` (Diversified), `PORTFOLIO_003` (Conservative).

Useful flags:

| Flag | Purpose |
|---|---|
| `--dry-run` | Use `MockLLMClient` — no API call, deterministic output, useful for CI / pipeline checks |
| `--provider {auto,groq,anthropic,mock}` | Force a specific LLM provider |
| `--model llama-3.3-70b-versatile` | Override the model name |
| `--no-eval` | Skip the self-evaluation pass |
| `--no-judge` | Run rule-based eval only (no extra LLM call) |
| `--json` | Emit structured JSON instead of Markdown |
| `--show-prompt` | Print the rendered prompt and exit (debugging) |

### Run the Streamlit UI

```bash
PYTHONPATH=src streamlit run app.py
```

Opens at http://localhost:8501. Pick a portfolio in the sidebar, hit "Generate briefing", then ask follow-up questions in the chat box at the bottom.

### Run the tests

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

Expected: **116 passing in <1s** (no API calls — all LLM-dependent tests use mocks).

---

## Project structure

```
financial-agent/
├── data/                              # 6 mock JSON files (provided by the brief)
├── src/financial_agent/
│   ├── models/                        # pydantic schemas (Stock, Portfolio, NewsArticle, …)
│   ├── data_loader.py                 # DataLoader — single entry to typed data
│   ├── market/                        # Phase 1
│   │   ├── analyzer.py                #   MarketAnalyzer — index sentiment
│   │   ├── sectors.py                 #   SectorAnalyzer — sector trends + divergences
│   │   └── news.py                    #   NewsProcessor — filtering + prioritization
│   ├── portfolio/                     # Phase 2
│   │   └── analytics.py               #   PortfolioAnalyzer — P&L, allocation, risk flags
│   ├── reasoning/                     # Phase 3
│   │   ├── briefing.py                #   typed Briefing output models
│   │   ├── context.py                 #   build_context — pre-filter + ₹ attribution
│   │   ├── prompts.py                 #   system + user prompts (with few-shot example)
│   │   ├── parser.py                  #   tolerant XML → typed Briefing
│   │   ├── client.py                  #   AnthropicClient, GroqClient, MockLLMClient
│   │   └── agent.py                   #   ReasoningAgent — orchestrator + confidence reconciliation
│   ├── observability/                 # Phase 4a
│   │   └── tracer.py                  #   Tracer — Langfuse wrapper (graceful no-op)
│   └── evaluation/                    # Phase 4b
│       ├── result.py                  #   EvaluationResult, DimensionScore
│       ├── rubric.py                  #   5 rule-based scorers + dimension weights
│       ├── judge.py                   #   LLM-as-judge prompt + parser
│       └── evaluator.py               #   BriefingEvaluator orchestrator
├── scripts/
│   ├── smoke_test.py                  # Phase 1 + 2 end-to-end on all 3 portfolios
│   └── run_briefing.py                # main CLI (Phase 1 → 4 + tracing + eval)
├── tests/                             # 116 pytest tests, no API calls
├── app.py                             # Streamlit UI entry point
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Sample output (PORTFOLIO_002 — Priya Patel, banking-heavy)

```
**Portfolio fell 2.73% as banking sector weighed down by RBI's hawkish stance and global risk-off sentiment.**

_Confidence: 0.80 — Macro drivers and sector trends align with the portfolio's move,
but the minimal IT allocation reduced the positive impact of INFY's deal win._

## Causal chain
1. **RBI's hawkish stance pressured banks (NEWS001).**
   - Sector: BANKING -3.11% weighted, top loser of the day.
   - Stocks: HDFCBANK -3.51% (22.62% weight, ₹-16,845), ICICIBANK -3.13%
            (13.79% weight, ₹-9,112), SBIN -3.02% (14.71% weight, ₹-9,360).
   - Portfolio: ₹46,850 of the ₹57,390 day loss (81.6%).
2. **Global risk-off sentiment drives selling (NEWS022).**
   - Sector: BANKING and FINANCIAL_SERVICES sectors fell due to high-beta nature.
   - Stocks: BAJFINANCE -2.01% contributing ₹-5,612 to the day's loss.
   - Portfolio: ₹8,172 of the ₹57,390 day loss (14.2%).

## Conflicting signals
- **INFY won a $1.5 billion deal (NEWS008) but the portfolio's IT allocation is
  minimal (1.0%).** Resolution: positive news did not significantly impact the
  portfolio due to low allocation.

## Recommendations
- [HIGH] Rebalance to reduce concentration in banking, given the CRITICAL
        sector concentration risk flag.

## Self-evaluation — Overall: 0.86 (rule + judge)
| Dimension          | Rule | Judge | Combined |
|--------------------|------|-------|----------|
| Causal Depth       | 1.00 | 0.80  | 0.92     |
| Evidence Accuracy  | 1.00 | 0.90  | 0.96     |
| Conflict Handling  | 0.20 | 0.70  | 0.40     |
| Prioritization     | 1.00 | 0.80  | 0.92     |
| Quantification     | 1.00 | 0.90  | 0.96     |
```

---

## Design decisions worth calling out

1. **The LLM does narrative, not arithmetic.** Every ₹ figure and % weight is pre-computed in Phase 1 / 2 and handed to the LLM in `contribution_attribution`. The model never multiplies prices; it just connects evidence into a story.
2. **The LLM is interchangeable.** `BaseLLMClient` is a `Protocol`. Switching from Groq (free Llama 3.3 70B) to Anthropic (Claude Sonnet 4.5) is one CLI flag. The mock client even produces a valid briefing without any API call — useful for CI.
3. **Every output is auditable.** Briefings cite news IDs that the parser checks against the context. The rule-based confidence scorer **penalizes** hallucinated IDs; the eval layer scores Evidence Accuracy ≤ 0.3 if any cited ID isn't in `relevant_news`.
4. **Tracing degrades gracefully.** Langfuse is wired through every LLM call but the `Tracer` is a no-op when keys are missing — so dev / CI / contributors without an account run identically.
5. **Eval can run without the judge.** Rule-based scorers always run for free; the LLM-as-judge is a second call that can be skipped via `--no-judge`. Good for cost control.

---

## Stack

- **Python 3.11+**, [pydantic 2.x](https://docs.pydantic.dev/) for typed models
- LLM providers: [groq](https://console.groq.com) (default, free), [anthropic](https://console.anthropic.com) (paid, optional)
- Tracing: [langfuse 3.x](https://langfuse.com)
- UI: [streamlit](https://streamlit.io)
- Tests: [pytest 9.x](https://pytest.org)
- Env loading: [python-dotenv](https://github.com/theskumar/python-dotenv)

---

## Live demo

**Try it now:** https://casual-advisor.streamlit.app/

Pick a portfolio in the sidebar, choose `groq` (free tier) or `mock` (no API key), and click **Generate briefing**. Then ask a follow-up question in the chat at the bottom.

> If you hit a Groq rate limit (free tier is ~12k tokens/min), switch the **LLM provider** dropdown to `mock` to see the deterministic pipeline output without any API call.

**Demo video:** _coming soon (2–3 min walkthrough)_

---

## Limitations / things deliberately scoped out

- **No real-time data.** All inputs come from the provided mock JSONs (snapshot of 2026-04-21).
- **No retrieval / RAG.** The briefing's context is built deterministically from structured data, not retrieved from a vector store.
- **No tool-use in chat (yet).** The follow-up chat is briefing-grounded — it answers questions about the briefing the user just saw, but doesn't (yet) invoke `lookup_stock` / `simulate_rebalance` style tools. Easy to add as a follow-up.
- **No production hardening.** No rate limiting, no API key rotation, no proper auth on the deployed UI.

---

## License

MIT
