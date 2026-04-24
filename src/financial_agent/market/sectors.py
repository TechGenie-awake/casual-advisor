"""Sector-level trend extraction, derived dynamically from stock data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from financial_agent.market.analyzer import MarketAnalyzer
from financial_agent.models import MarketSnapshot, Sentiment


@dataclass(frozen=True)
class SectorTrend:
    """A single sector's day move, computed from its constituent stocks."""

    sector_code: str
    avg_change_percent: float          # equal-weighted across constituents
    weighted_change_percent: float     # market-cap weighted
    sentiment: Sentiment
    constituent_count: int
    top_gainers: list[tuple[str, float]] = field(default_factory=list)
    top_losers: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sector_code": self.sector_code,
            "avg_change_percent": round(self.avg_change_percent, 2),
            "weighted_change_percent": round(self.weighted_change_percent, 2),
            "sentiment": self.sentiment.value,
            "constituent_count": self.constituent_count,
            "top_gainers": [(s, round(c, 2)) for s, c in self.top_gainers],
            "top_losers": [(s, round(c, 2)) for s, c in self.top_losers],
        }


class SectorAnalyzer:
    """Aggregates per-stock moves into sector trends.

    The dataset already ships `sector_performance`, but Phase 1 requires us to
    derive trends *dynamically from stock data* — so we recompute from the
    `stocks` map and only use the shipped sector data for cross-checks.
    """

    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    def trend_for(self, sector_code: str) -> SectorTrend | None:
        stocks = [s for s in self._snapshot.stocks.values() if s.sector == sector_code]
        if not stocks:
            return None

        avg_change = sum(s.change_percent for s in stocks) / len(stocks)

        # Market-cap weighted change (handles missing market caps gracefully).
        total_mcap = sum(s.market_cap_cr or 0 for s in stocks)
        if total_mcap > 0:
            weighted = sum((s.market_cap_cr or 0) * s.change_percent for s in stocks) / total_mcap
        else:
            weighted = avg_change

        sentiment = MarketAnalyzer.classify(weighted)

        ranked = sorted(stocks, key=lambda s: s.change_percent, reverse=True)
        gainers = [(s.symbol, s.change_percent) for s in ranked if s.change_percent > 0][:3]
        losers = [(s.symbol, s.change_percent) for s in reversed(ranked) if s.change_percent < 0][:3]

        return SectorTrend(
            sector_code=sector_code,
            avg_change_percent=avg_change,
            weighted_change_percent=weighted,
            sentiment=sentiment,
            constituent_count=len(stocks),
            top_gainers=gainers,
            top_losers=losers,
        )

    def all_trends(self) -> dict[str, SectorTrend]:
        sectors: set[str] = {s.sector for s in self._snapshot.stocks.values()}
        sectors.update(self._snapshot.sector_performance.keys())
        result: dict[str, SectorTrend] = {}
        for code in sectors:
            trend = self.trend_for(code)
            if trend is not None:
                result[code] = trend
        return result

    def ranked(self) -> list[SectorTrend]:
        """All sectors sorted from worst to best by weighted change."""
        return sorted(self.all_trends().values(), key=lambda t: t.weighted_change_percent)

    def divergences(self) -> list[tuple[str, str, float, float]]:
        """Stocks moving opposite to their sector. Useful for conflict detection.

        Returns: (symbol, sector, stock_change, sector_change) tuples.
        """
        trends = self.all_trends()
        out = []
        for stock in self._snapshot.stocks.values():
            trend = trends.get(stock.sector)
            if trend is None:
                continue
            sector_change = trend.weighted_change_percent
            # Opposite signs and meaningful magnitude.
            if sector_change * stock.change_percent < 0 and abs(stock.change_percent) >= 0.5:
                out.append((stock.symbol, stock.sector, stock.change_percent, sector_change))
        return sorted(out, key=lambda t: abs(t[2] - t[3]), reverse=True)

    def constituents_by_sector(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for sym, stock in self._snapshot.stocks.items():
            out[stock.sector].append(sym)
        return dict(out)
