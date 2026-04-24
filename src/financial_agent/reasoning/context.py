"""Build the LLM-facing reasoning context from Phase 1 + Phase 2 outputs.

The single most important guard against weak reasoning is to send the LLM
only what it needs. This module is the filter:

  * Sectors: only those the portfolio is exposed to (look-through).
  * News: only items touching held stocks/sectors, plus market-wide HIGH-impact.
  * Divergences: only those that touch held stocks.
  * Conflicts: news whose sentiment opposes its sector's mood, intersected
    with the portfolio's exposure.

It also pre-computes two enrichments that materially boost reasoning quality:

  * `contribution_attribution`: each holding's ₹ share of the day's P&L,
    aggregated by sector. Lets the LLM cite "₹40k of ₹57k" instead of guessing.
  * `relevant_macro_correlations`: a filtered slice of sector_mapping's
    macro-event correlations, scoped to the portfolio's sector exposure.
"""

from __future__ import annotations

from collections import defaultdict
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
MAX_HOLDING_CONTRIBUTORS = 6  # top contributors (positive + negative) the LLM sees by name


@dataclass(frozen=True)
class HoldingContribution:
    identifier: str
    name: str
    sector: str
    day_change_value: float       # ₹ contribution (signed)
    pct_of_day_pnl: float | None  # share of total day P&L; None if day P&L ≈ 0


@dataclass(frozen=True)
class ContributionAttribution:
    """Each holding's ₹ share of today's portfolio P&L."""

    total_day_pnl: float
    by_sector: dict[str, float]            # sector → ₹ contributed
    top_contributors: list[HoldingContribution]  # mix of biggest losers + gainers

    def to_dict(self) -> dict:
        return {
            "total_day_pnl": round(self.total_day_pnl, 2),
            "by_sector": {k: round(v, 2) for k, v in self.by_sector.items()},
            "top_contributors": [
                {
                    "identifier": c.identifier,
                    "name": c.name,
                    "sector": c.sector,
                    "day_change_value": round(c.day_change_value, 2),
                    "pct_of_day_pnl": (
                        None if c.pct_of_day_pnl is None
                        else round(c.pct_of_day_pnl, 2)
                    ),
                }
                for c in self.top_contributors
            ],
        }


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
    contribution_attribution: ContributionAttribution | None = None
    relevant_macro_correlations: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def evidence_ids(self) -> list[str]:
        """All news IDs the agent is permitted to cite."""
        return [a.id for a in self.relevant_news]

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "market": self.market.to_dict(),
            "portfolio_snapshot": self.portfolio_snapshot.to_dict(),
            "contribution_attribution": (
                self.contribution_attribution.to_dict()
                if self.contribution_attribution
                else None
            ),
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
            "relevant_macro_correlations": self.relevant_macro_correlations,
        }


def _compute_attribution(
    portfolio: Portfolio,
    snapshot: PortfolioSnapshot,
) -> ContributionAttribution:
    """Aggregate per-holding ₹ day-change into sector buckets + a top-N list."""
    by_sector: dict[str, float] = defaultdict(float)
    contributions: list[HoldingContribution] = []

    total = snapshot.day_pnl
    pct_safe = abs(total) >= 1.0  # avoid divide-by-near-zero on tiny day P&L

    for s in portfolio.stocks:
        by_sector[s.sector] += s.day_change
        contributions.append(
            HoldingContribution(
                identifier=s.symbol,
                name=s.name,
                sector=s.sector,
                day_change_value=s.day_change,
                pct_of_day_pnl=(s.day_change / total * 100) if pct_safe else None,
            )
        )

    for m in portfolio.mutual_funds:
        # MFs don't sit in a single equity sector — bucket them under their category.
        bucket = (
            "DEBT_FUNDS"
            if "DEBT" in m.category.upper()
            else "HYBRID_FUNDS"
            if "HYBRID" in m.category.upper()
            else "MUTUAL_FUNDS"
        )
        by_sector[bucket] += m.day_change
        contributions.append(
            HoldingContribution(
                identifier=m.scheme_code,
                name=m.scheme_name,
                sector=bucket,
                day_change_value=m.day_change,
                pct_of_day_pnl=(m.day_change / total * 100) if pct_safe else None,
            )
        )

    # Top contributors by absolute ₹ impact — gainers and losers both surface.
    top = sorted(contributions, key=lambda c: abs(c.day_change_value), reverse=True)[
        :MAX_HOLDING_CONTRIBUTORS
    ]
    return ContributionAttribution(
        total_day_pnl=total,
        by_sector=dict(by_sector),
        top_contributors=top,
    )


def _filter_macro_correlations(
    all_correlations: dict[str, dict[str, list[str]]],
    held_sectors: set[str],
) -> dict[str, dict[str, list[str]]]:
    """Keep only macro events that touch a held sector (positively or negatively)."""
    out: dict[str, dict[str, list[str]]] = {}
    for event, impact in all_correlations.items():
        sectors_touched = (
            set(impact.get("negative_impact", []))
            | set(impact.get("positive_impact", []))
            | set(impact.get("neutral", []))
        )
        if sectors_touched & held_sectors:
            out[event] = impact
    return out


def build_context(
    portfolio: Portfolio,
    *,
    market_analyzer: MarketAnalyzer,
    sector_analyzer: SectorAnalyzer,
    news_processor: NewsProcessor,
    portfolio_analyzer: PortfolioAnalyzer,
    macro_correlations: dict[str, dict[str, list[str]]] | None = None,
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

    attribution = _compute_attribution(portfolio, snapshot)

    filtered_macro = (
        _filter_macro_correlations(macro_correlations, held_sectors)
        if macro_correlations
        else {}
    )

    return ReasoningContext(
        portfolio_id=portfolio.portfolio_id,
        market=market_intel,
        portfolio_snapshot=snapshot,
        relevant_sector_trends=relevant_trends,
        relevant_news=relevant_news,
        stock_vs_sector_divergences=portfolio_divergences,
        conflict_news=portfolio_conflict_news,
        contribution_attribution=attribution,
        relevant_macro_correlations=filtered_macro,
    )
