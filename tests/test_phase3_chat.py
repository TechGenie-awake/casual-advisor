"""Phase 3 — Briefing-grounded follow-up chat (Flavor A).

The chat agent is a single-shot conversational layer over a Briefing +
ReasoningContext. These tests do not hit a real LLM — the canned response
is supplied via MockLLMClient.
"""

from __future__ import annotations

import pytest

from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.portfolio import PortfolioAnalyzer
from financial_agent.reasoning import (
    ChatAgent,
    ChatSession,
    MockLLMClient,
    ReasoningAgent,
    build_context,
)


def _ctx(loader, pid):
    portfolio = loader.get_portfolio(pid)
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


def _briefing_for(loader, pid):
    """Build a real briefing using the deterministic mock so tests are stable."""
    ctx = _ctx(loader, pid)
    mock = MockLLMClient.from_context(ctx)
    return ReasoningAgent(client=mock).generate(ctx)


def test_chat_session_starts_empty():
    session = ChatSession(briefing=None, context=None)  # types deliberately ignored
    assert session.messages == []


def test_chat_agent_returns_canned_answer(loader):
    run = _briefing_for(loader, "PORTFOLIO_002")
    session = ChatSession(briefing=run.briefing, context=_ctx(loader, "PORTFOLIO_002"))
    chat = ChatAgent(client=MockLLMClient("Banking dragged ₹46,850 of your day's loss."))

    turn = chat.ask(session, "Why did my portfolio fall so much?")
    assert "Banking" in turn.answer
    assert turn.question.startswith("Why did")


def test_chat_persists_history_across_turns(loader):
    run = _briefing_for(loader, "PORTFOLIO_002")
    session = ChatSession(briefing=run.briefing, context=_ctx(loader, "PORTFOLIO_002"))
    chat = ChatAgent(client=MockLLMClient("First answer."))

    chat.ask(session, "Q1?")
    # Switch the canned response for turn 2.
    chat._client = MockLLMClient("Second answer.")
    chat.ask(session, "Q2?")

    assert len(session.messages) == 4  # 2 user + 2 assistant
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    assert session.messages[2].content == "Q2?"
    assert session.messages[3].content == "Second answer."


def test_chat_payload_includes_briefing_and_context(loader, monkeypatch):
    """Sanity: the user prompt sent to the LLM contains both the briefing and the JSON context."""
    from financial_agent.reasoning import chat as chat_module

    captured: dict = {}

    class CapturingClient:
        _model = "capture"

        def complete(self, system, user):
            from financial_agent.reasoning.client import LLMResponse
            captured["system"] = system
            captured["user"] = user
            return LLMResponse(text="ok", model="capture", input_tokens=10, output_tokens=2)

    run = _briefing_for(loader, "PORTFOLIO_002")
    session = ChatSession(briefing=run.briefing, context=_ctx(loader, "PORTFOLIO_002"))
    ChatAgent(client=CapturingClient()).ask(session, "test question")

    user = captured["user"]
    # Briefing markdown rendered in.
    assert "Causal chain" in user
    # Context JSON rendered in.
    assert "portfolio_snapshot" in user
    # Question rendered in.
    assert "test question" in user
    # System prompt contains the hard-rule about citation.
    assert "Cite only" in captured["system"]


def test_chat_history_appears_in_subsequent_payload(loader):
    """Turn 2's payload must include turn 1's Q+A under 'Conversation so far'."""
    from financial_agent.reasoning import chat as chat_module

    captured: dict = {"calls": []}

    class CapturingClient:
        _model = "capture"

        def __init__(self, reply):
            self._reply = reply

        def complete(self, system, user):
            from financial_agent.reasoning.client import LLMResponse
            captured["calls"].append(user)
            return LLMResponse(text=self._reply, model="capture", input_tokens=1, output_tokens=1)

    run = _briefing_for(loader, "PORTFOLIO_002")
    session = ChatSession(briefing=run.briefing, context=_ctx(loader, "PORTFOLIO_002"))

    a = ChatAgent(client=CapturingClient("first reply"))
    a.ask(session, "first question")

    a._client = CapturingClient("second reply")
    a.ask(session, "second question")

    second_payload = captured["calls"][1]
    assert "first question" in second_payload
    assert "first reply" in second_payload
    assert "second question" in second_payload
