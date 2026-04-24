"""Build the LLM-facing reasoning context from Phase 1 + Phase 2 outputs.

The single most important guard against weak reasoning is to send the LLM
only what it needs. This module is the filter:

  * Sectors: only those the portfolio is exposed to (look-through).
  * News: only items touching held stocks/sectors, plus market-wide HIGH-impact.
  * Divergences: only those that touch held stocks.
  * Conflicts: news whose sentiment opposes its sector's mood, intersected
    with the portfolio's exposure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from financial_agent.market import (
    MarketAnalyzer,
    MarketIntelligence,
    NewsProcessor,
    SectorAnalyzer,
    SectorTrend,
)
from financial_agent.models import (
    NewsArticle,
    Portfolio,
)
from financial_agent.portfolio import PortfolioAnalyzer, PortfolioSnapshot


# How many of each list to surface to the LLM. Hard caps to control prompt size.
MAX_NEWS_ITEMS = 8
MAX_SECTOR_TRENDS = 8
MAX_DIVERGENCES = 5
MAX_CONFLICTS = 5


@dataclass(frozen=True)
class ReasoningContext:
    """Everything the LLM is allowed to reference. Nothing else.

    The agent's prompt instructs it to refuse to invent facts not present
    here, so this is effectively the "knowledge base" for one briefing.
    """

    portfolio_id: str
    market: MarketIntelligence
    portfolio_snapshot: PortfolioSnapshot
    relevant_sector_trends: list[SectorTrend]
    relevant_news: list[NewsArticle]
    stock_vs_sector_divergences: list[tuple[str, str, float, float]]
    conflict_news: list[NewsArticle]
    extras: dict[str, Any] = field(default_factory=dict)

    def evidence_ids(self) -> list[str]:
        """All news IDs the agent is permitted to cite."""
        return [a.id for a in self.relevant_news]

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "market": self.market.to_dict(),
            "portfolio_snapshot": self.portfolio_snapshot.to_dict(),
            "relevant_sector_trends": [t.to_dict() for t in self.relevant_sector_trends],
            "relevant_news": [
                {
                    "id": a.id,
                    "headline": a.headline,
                    "summary": a.summary,
                    "sentiment": a.sentiment.value,
                    "scope": a.scope.value,
                    "impact": a.impact_level.value,
                    "sectors": a.entities.sectors,
                    "stocks": a.entities.stocks,
                    "causal_factors": a.causal_factors,
                }
                for a in self.relevant_news
            ],
            "stock_vs_sector_divergences": [
                {
                    "symbol": sym,
                    "sector": sec,
                    "stock_change_pct": round(sc, 2),
                    "sector_change_pct": round(sec_c, 2),
                }
                for sym, sec, sc, sec_c in self.stock_vs_sector_divergences
            ],
            "conflict_news": [
                {
                    "id": a.id,
                    "headline": a.headline,
                    "sentiment": a.sentiment.value,
                    "sectors": a.entities.sectors,
                }
                for a in self.conflict_news
            ],
        }


def build_context(
    portfolio: Portfolio,
    *,
    market_analyzer: MarketAnalyzer,
    sector_analyzer: SectorAnalyzer,
    news_processor: NewsProcessor,
    portfolio_analyzer: PortfolioAnalyzer,
) -> ReasoningContext:
    """Assemble the context for one portfolio in a single pass."""
    snapshot = portfolio_analyzer.snapshot()
    market_intel = market_analyzer.analyze()
    all_trends = sector_analyzer.all_trends()

    # --- Holdings exposure (look-through includes MF underlying sectors) ----
    held_sectors: set[str] = set(snapshot.sector_allocation_lookthrough.keys())
    # Also include the raw bucket sectors (covers MF buckets like "DEBT_FUNDS").
    held_sectors.update(snapshot.sector_allocation.keys())
    held_symbols: set[str] = {s.symbol for s in portfolio.stocks}
    for mf in portfolio.mutual_funds:
        held_symbols.update(mf.top_holdings)

    # --- Filter sector trends to those the portfolio touches ----------------
    relevant_trends = sorted(
        (t for code, t in all_trends.items() if code in held_sectors),
        key=lambda t: abs(t.weighted_change_percent),
        reverse=True,
    )[:MAX_SECTOR_TRENDS]

    # --- Filter news to those that could move this portfolio ----------------
    relevant_news = news_processor.filter_for_holdings(
        symbols=held_symbols,
        sectors=held_sectors,
        include_market_wide_high_impact=True,
    )[:MAX_NEWS_ITEMS]

    # --- Stock-vs-sector divergences within held stocks --------------------
    all_divergences = sector_analyzer.divergences()
    portfolio_divergences = [
        d for d in all_divergences if d[0] in {s.symbol for s in portfolio.stocks}
    ][:MAX_DIVERGENCES]

    # --- Conflict news, scoped to portfolio ---------------------------------
    sector_sentiments = {code: t.sentiment for code, t in all_trends.items()}
    all_conflicts = news_processor.conflict_candidates(sector_sentiments)
    portfolio_conflict_news = [
        a for a in all_conflicts
        if any(s in held_sectors for s in a.entities.sectors)
        or any(s in held_symbols for s in a.entities.stocks)
    ][:MAX_CONFLICTS]

    return ReasoningContext(
        portfolio_id=portfolio.portfolio_id,
        market=market_intel,
        portfolio_snapshot=snapshot,
        relevant_sector_trends=relevant_trends,
        relevant_news=relevant_news,
        stock_vs_sector_divergences=portfolio_divergences,
        conflict_news=portfolio_conflict_news,
    )
