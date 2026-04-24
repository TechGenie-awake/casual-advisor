"""Phase 3.1 — Context filtering: only relevant data reaches the LLM.

Brief requirement (Phase 3 — Prioritization):
    "Highlight only high-impact signals."

This translates to: the context fed to the reasoning layer must already be
filtered to the portfolio's actual exposure. Irrelevant news / sectors must
not leak into the prompt.
"""

import pytest

from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.portfolio import PortfolioAnalyzer
from financial_agent.reasoning import build_context


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
    )


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
def test_context_built_for_each_portfolio(loader, pid):
    ctx = _ctx(loader, pid)
    assert ctx.portfolio_id == pid
    assert ctx.market.overall_sentiment is not None
    assert ctx.portfolio_snapshot.portfolio_id == pid


def test_p2_context_includes_banking_sector_trend(loader):
    """Priya's portfolio is 72% banking — BANKING trend must be in the context."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    sector_codes = {t.sector_code for t in ctx.relevant_sector_trends}
    assert "BANKING" in sector_codes


def test_p2_context_includes_rbi_news(loader):
    """The most relevant news (RBI hawkish) must be surfaced for a banking-heavy portfolio."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    news_ids = {a.id for a in ctx.relevant_news}
    assert "NEWS001" in news_ids


def test_news_caps_respected(loader):
    """Context must not balloon — hard caps protect prompt size."""
    from financial_agent.reasoning.context import (
        MAX_CONFLICTS,
        MAX_DIVERGENCES,
        MAX_NEWS_ITEMS,
        MAX_SECTOR_TRENDS,
    )
    for pid in ("PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"):
        ctx = _ctx(loader, pid)
        assert len(ctx.relevant_news) <= MAX_NEWS_ITEMS
        assert len(ctx.relevant_sector_trends) <= MAX_SECTOR_TRENDS
        assert len(ctx.stock_vs_sector_divergences) <= MAX_DIVERGENCES
        assert len(ctx.conflict_news) <= MAX_CONFLICTS


def test_p2_context_excludes_pure_pharma_news(loader):
    """Priya holds zero pharma — Sun Pharma stock-specific news should not appear."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    news_ids = {a.id for a in ctx.relevant_news}
    # NEWS004 is the Sun Pharma USFDA approval — STOCK_SPECIFIC, no banking exposure.
    assert "NEWS004" not in news_ids


def test_p2_conflict_news_includes_bajaj_finance(loader):
    """Bajaj Finance positive news + bearish FS sector — must surface as a conflict."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    headlines = " ".join(a.headline for a in ctx.conflict_news)
    assert "Bajaj Finance" in headlines


def test_context_to_dict_is_json_serializable(loader):
    """Context must round-trip through JSON for the prompt template."""
    import json
    ctx = _ctx(loader, "PORTFOLIO_001")
    payload = ctx.to_dict()
    assert json.dumps(payload)  # raises if not serializable


def test_evidence_ids_match_relevant_news(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    assert ctx.evidence_ids() == [a.id for a in ctx.relevant_news]
