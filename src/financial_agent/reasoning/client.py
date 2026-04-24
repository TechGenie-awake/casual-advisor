"""Thin LLM client abstraction.

A `BaseLLMClient` interface so the agent can be unit-tested with mocks.
The real implementation wraps the Anthropic SDK; the mock implementation
returns a canned XML response. Phase 4 (observability) will subclass these
to add Langfuse traces.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

# Default to Sonnet 4.6 — strong reasoning at low latency / cost.
DEFAULT_MODEL = "claude-sonnet-4-5"
DEFAULT_MAX_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.2


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class BaseLLMClient(Protocol):
    """Minimal interface the ReasoningAgent depends on."""

    def complete(self, system: str, user: str) -> LLMResponse: ...


class AnthropicClient:
    """Wraps the Anthropic SDK with sane defaults for this project."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        # Imported lazily so the rest of the package works without anthropic installed.
        from anthropic import Anthropic

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it in your shell or pass api_key= explicitly."
            )

        self._client = Anthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> LLMResponse:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # `content` is a list of typed blocks; we expect a single text block.
        text_parts = [block.text for block in msg.content if block.type == "text"]
        text = "".join(text_parts)
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
        )


class GroqClient:
    """Wraps the Groq SDK. Free tier; OpenAI-style chat-completions API.

    Default model is Llama 3.3 70B Versatile — the strongest free-tier model
    Groq currently exposes, and capable of reliable XML output for our schema.
    """

    DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        *,
        model: str = DEFAULT_GROQ_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        api_key: str | None = None,
    ) -> None:
        from groq import Groq

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Set it in your shell, in .env, or pass api_key= explicitly."
            )

        self._client = Groq(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system: str, user: str) -> LLMResponse:
        msg = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = msg.choices[0].message.content or ""
        return LLMResponse(
            text=text,
            model=self._model,
            input_tokens=msg.usage.prompt_tokens,
            output_tokens=msg.usage.completion_tokens,
        )


class MockLLMClient:
    """Returns a canned response. Used for unit tests and dry-runs.

    Tests can pass a custom XML string; the dry-run mode in the CLI uses
    `MockLLMClient.from_context(context)` to produce a deterministic
    placeholder briefing built straight from the structured context.
    """

    def __init__(self, response_text: str) -> None:
        self._response = response_text

    def complete(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(
            text=self._response,
            model="mock",
            input_tokens=len(system + user) // 4,
            output_tokens=len(self._response) // 4,
        )

    @classmethod
    def from_context(cls, context) -> "MockLLMClient":
        """Build a deterministic briefing from the context, no LLM required.

        This is a *mechanical* projection of Phase 1+2 outputs into the
        briefing schema — useful for end-to-end pipeline tests and for
        showcasing the output structure without burning API credits.
        """
        snap = context.portfolio_snapshot
        market = context.market

        # Headline: pick the largest signal we have.
        headline = (
            f"{context.portfolio_id} day move {snap.day_pnl_percent:+.2f}% "
            f"on {market.overall_sentiment.value.lower()} market "
            f"({market.benchmark_change_percent:+.2f}%)."
        )

        # Causal chain: top-2 sector moves intersected with the top news.
        links = []
        worst_sectors = sorted(
            context.relevant_sector_trends,
            key=lambda t: t.weighted_change_percent,
        )[:2]
        for trend in worst_sectors:
            related_news = [
                n for n in context.relevant_news
                if trend.sector_code in n.entities.sectors
            ][:2]
            evidence = ",".join(n.id for n in related_news) or "—"
            news_summary = (
                related_news[0].headline if related_news else "no headline news"
            )
            sector_holdings = [
                m for m in (snap.top_losers + snap.top_gainers)
                if m.identifier in (t for t, _ in trend.top_losers + trend.top_gainers)
            ][:3]
            stock_blurb = (
                ", ".join(f"{m.identifier} {m.day_change_percent:+.2f}%" for m in sector_holdings)
                or "various holdings"
            )
            links.append(
                f"<link>"
                f"<macro_event>{news_summary}</macro_event>"
                f"<sector_impact>{trend.sector_code} {trend.weighted_change_percent:+.2f}%</sector_impact>"
                f"<stock_impact>{stock_blurb}</stock_impact>"
                f"<portfolio_impact>Contributed to portfolio {snap.day_pnl_percent:+.2f}% day move</portfolio_impact>"
                f"<evidence>{evidence}</evidence>"
                f"</link>"
            )

        # Conflicts from divergences + conflicting news.
        conflict_xml = []
        for sym, sec, sc, sec_c in context.stock_vs_sector_divergences[:2]:
            conflict_xml.append(
                f"<conflict>"
                f"<description>{sym} moved {sc:+.2f}% while its sector {sec} moved {sec_c:+.2f}%</description>"
                f"<resolution>Stock-specific catalyst overrode the sector trend.</resolution>"
                f"<evidence>—</evidence>"
                f"</conflict>"
            )

        # Recommendations from risks.
        rec_xml = []
        for risk in snap.risks[:3]:
            priority = {"CRITICAL": "HIGH", "WARN": "MEDIUM", "INFO": "LOW"}.get(
                risk.severity.value, "MEDIUM"
            )
            rec_xml.append(
                f'<recommendation priority="{priority}">{risk.message}</recommendation>'
            )

        evidence_all = ",".join(context.evidence_ids()[:6]) or "—"
        confidence = 0.55  # mock is intentionally conservative
        rationale = "Mock client output — no LLM reasoning applied."

        body = f"""<briefing>
<headline>{headline}</headline>
<causal_chain>{''.join(links) or '<link><macro_event>—</macro_event><sector_impact>—</sector_impact><stock_impact>—</stock_impact><portfolio_impact>—</portfolio_impact><evidence>—</evidence></link>'}</causal_chain>
<conflicts>{''.join(conflict_xml)}</conflicts>
<recommendations>{''.join(rec_xml) or '<recommendation priority="LOW">No risks detected.</recommendation>'}</recommendations>
<evidence>{evidence_all}</evidence>
<confidence>{confidence}</confidence>
<confidence_rationale>{rationale}</confidence_rationale>
</briefing>"""
        return cls(body)
