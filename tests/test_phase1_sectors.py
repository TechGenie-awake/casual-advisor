"""Phase 1.2 — Sector Extraction: dynamically derive sector trends from stock data.

Brief requirement:
    "Derive sector-level trends dynamically from stock data."
"""

import pytest

from financial_agent.market import SectorAnalyzer
from financial_agent.models import Sentiment


def test_all_unique_stock_sectors_have_trends(loader):
    """Every sector that appears on a stock must yield a SectorTrend."""
    sectors_in_stocks = {s.sector for s in loader.market.stocks.values()}
    trends = SectorAnalyzer(loader.market).all_trends()
    assert sectors_in_stocks.issubset(trends.keys())


def test_banking_is_the_worst_sector(loader):
    """RBI hawkish day → BANKING should be the most-bearish sector."""
    ranked = SectorAnalyzer(loader.market).ranked()
    assert ranked[0].sector_code == "BANKING"
    assert ranked[0].sentiment == Sentiment.BEARISH


def test_it_is_the_top_sector(loader):
    ranked = SectorAnalyzer(loader.market).ranked()
    assert ranked[-1].sector_code == "INFORMATION_TECHNOLOGY"
    assert ranked[-1].sentiment == Sentiment.BULLISH


def test_weighted_change_is_aggregated_from_constituent_stocks(loader):
    """Weighted sector change must be a market-cap weighted mean of its stocks."""
    analyzer = SectorAnalyzer(loader.market)
    trend = analyzer.trend_for("BANKING")
    assert trend is not None
    stocks = [s for s in loader.market.stocks.values() if s.sector == "BANKING"]
    total_mcap = sum(s.market_cap_cr or 0 for s in stocks)
    expected = sum((s.market_cap_cr or 0) * s.change_percent for s in stocks) / total_mcap
    assert trend.weighted_change_percent == pytest.approx(expected, abs=1e-6)


def test_tata_motors_appears_as_divergence(loader):
    """+0.79% inside an Auto sector at -1.44% — must be flagged."""
    divs = SectorAnalyzer(loader.market).divergences()
    symbols = {sym for sym, *_ in divs}
    assert "TATAMOTORS" in symbols


def test_cipla_appears_as_divergence(loader):
    """-0.84% inside a Pharma sector at +0.94% — must be flagged."""
    divs = SectorAnalyzer(loader.market).divergences()
    symbols = {sym for sym, *_ in divs}
    assert "CIPLA" in symbols
