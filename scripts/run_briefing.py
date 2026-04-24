"""CLI for the Phase 3 reasoning agent.

Usage:
    PYTHONPATH=src python3 scripts/run_briefing.py PORTFOLIO_002
    PYTHONPATH=src python3 scripts/run_briefing.py PORTFOLIO_002 --dry-run
    PYTHONPATH=src python3 scripts/run_briefing.py PORTFOLIO_002 --json

Live mode (default) requires ANTHROPIC_API_KEY in the environment.
Dry-run uses MockLLMClient — no API call, no key needed — to demonstrate
the full pipeline against the deterministic mock briefing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running without `pip install -e .`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Load .env for API keys (no-op if file or python-dotenv missing).
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from financial_agent.data_loader import DataLoader  # noqa: E402
from financial_agent.evaluation import BriefingEvaluator  # noqa: E402
from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer  # noqa: E402
from financial_agent.observability import Tracer  # noqa: E402
from financial_agent.portfolio import PortfolioAnalyzer  # noqa: E402
from financial_agent.reasoning import (  # noqa: E402
    MockLLMClient,
    ReasoningAgent,
    build_context,
)

DATA_DIR = ROOT / "data"


def _resolve_provider(requested: str) -> str:
    """'auto' → groq if its key is set, else anthropic if its key is set, else error."""
    if requested != "auto":
        return requested
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "none"


def _build_client(args):
    """Returns an LLM client or None (with a stderr message) on failure."""
    from financial_agent.reasoning import AnthropicClient, GroqClient

    if args.dry_run:
        # Built later from the per-portfolio context.
        return "DEFER_TO_DRY_RUN"

    provider = _resolve_provider(args.provider)
    if provider == "mock":
        return "DEFER_TO_DRY_RUN"

    if provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            sys.stderr.write("GROQ_API_KEY not set (check .env or your shell).\n")
            return None
        kwargs = {"model": args.model} if args.model else {}
        return GroqClient(**kwargs)

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.stderr.write("ANTHROPIC_API_KEY not set (check .env or your shell).\n")
            return None
        kwargs = {"model": args.model} if args.model else {}
        return AnthropicClient(**kwargs)

    sys.stderr.write(
        "No provider available. Set GROQ_API_KEY or ANTHROPIC_API_KEY in .env, "
        "or re-run with --dry-run.\n"
    )
    return None


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a portfolio briefing.")
    parser.add_argument("portfolio_id", help="e.g., PORTFOLIO_002")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the deterministic MockLLMClient (no API call).",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "groq", "anthropic", "mock"],
        default="auto",
        help="LLM provider. 'auto' picks groq if GROQ_API_KEY set, else anthropic.",
    )
    parser.add_argument(
        "--model",
        help="Override the model name (e.g., 'llama-3.3-70b-versatile' or 'claude-sonnet-4-5').",
    )
    parser.add_argument(
        "--json",
        dest="emit_json",
        action="store_true",
        help="Emit the parsed Briefing as JSON instead of Markdown.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the rendered user prompt and exit.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip the self-evaluation pass.",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge call (rule-based eval only — no extra API call).",
    )
    args = parser.parse_args()

    loader = DataLoader(DATA_DIR)
    if args.portfolio_id not in loader.portfolios:
        sys.stderr.write(
            f"Unknown portfolio_id '{args.portfolio_id}'. "
            f"Available: {', '.join(sorted(loader.portfolios))}\n"
        )
        return 2

    context = make_context(loader, args.portfolio_id)

    if args.show_prompt:
        from financial_agent.reasoning.prompts import (
            SYSTEM_PROMPT,
            estimate_tokens,
            render_user_prompt,
        )
        user = render_user_prompt(context)
        print("=== SYSTEM PROMPT ===")
        print(SYSTEM_PROMPT)
        print()
        print("=== USER PROMPT ===")
        print(user)
        print()
        print(
            f"[~tokens] system={estimate_tokens(SYSTEM_PROMPT)} "
            f"user={estimate_tokens(user)} total={estimate_tokens(SYSTEM_PROMPT + user)}"
        )
        return 0

    # Choose the client. Sentinel string means "use the dry-run mock built from context".
    client = _build_client(args)
    if client is None:
        return 2
    if client == "DEFER_TO_DRY_RUN":
        client = MockLLMClient.from_context(context)

    tracer = Tracer()
    agent = ReasoningAgent(client=client, tracer=tracer)
    run = agent.generate(context)

    # --- Self-evaluation -------------------------------------------------
    eval_result = None
    if not args.no_eval:
        # Reuse the same client for the judge unless it's the deterministic mock.
        judge_client = None if (args.no_judge or args.dry_run) else client
        evaluator = BriefingEvaluator(judge_client=judge_client, tracer=tracer)
        eval_result = evaluator.score(
            run.briefing, context, trace_span=run.trace
        )

    # --- Output ----------------------------------------------------------
    if args.emit_json:
        out = run.briefing.to_dict()
        out["_meta"] = {
            "model": run.response.model,
            "input_tokens": run.response.input_tokens,
            "output_tokens": run.response.output_tokens,
            "rule_based_confidence": round(run.rule_based_confidence, 2),
            "tracing_enabled": tracer.enabled,
        }
        if eval_result:
            out["evaluation"] = eval_result.to_dict()
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        print(run.briefing.to_markdown())
        print()
        print(
            f"_Tokens: in={run.response.input_tokens} out={run.response.output_tokens} "
            f"| Rule-based confidence: {run.rule_based_confidence:.2f}"
            f" | Tracing: {'on' if tracer.enabled else 'off'}_"
        )
        if eval_result:
            print()
            print(eval_result.to_markdown())

    # End the trace explicitly so Langfuse flushes.
    tracer.end_briefing(
        run.trace,
        output={
            "headline": run.briefing.headline,
            "confidence": run.briefing.confidence,
            "eval_overall": (eval_result.overall_score if eval_result else None),
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
