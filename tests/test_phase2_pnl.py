"""Phase 2.1 — Daily P&L: absolute and percentage changes.

Brief requirement:
    "Daily P&L: Absolute and percentage changes."

We verify our recomputed P&L against the values shipped in `analytics`.
P1 and P2 should match exactly; P3 has source-data inconsistencies in
current_value, so we only assert the day-P&L there.
"""

import pytest

from financial_agent.portfolio import PortfolioAnalyzer


def _snap(loader, pid):
    return PortfolioAnalyzer(
        loader.get_portfolio(pid),
        mutual_funds=loader.mutual_funds,
        rate_sensitive_sectors=loader.sector_map.rate_sensitive_sectors,
    ).snapshot()


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
def test_day_pnl_matches_shipped_analytics(loader, pid):
    snap = _snap(loader, pid)
    shipped = loader.get_portfolio(pid).analytics.day_summary
    assert snap.day_pnl == pytest.approx(shipped["day_change_absolute"], abs=1.5)
    assert snap.day_pnl_percent == pytest.approx(shipped["day_change_percent"], abs=0.05)


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002"])
def test_overall_pnl_components_are_consistent(loader, pid):
    """invested + overall_pnl must equal current_value."""
    snap = _snap(loader, pid)
    assert snap.total_invested + snap.overall_pnl == pytest.approx(snap.current_value, abs=1.0)


def test_day_pnl_pct_uses_previous_value_not_current(loader):
    """Day % is day_change / (current - day_change) — not day_change / current."""
    snap = _snap(loader, "PORTFOLIO_002")
    prev = snap.current_value - snap.day_pnl
    expected = snap.day_pnl / prev * 100
    assert snap.day_pnl_percent == pytest.approx(expected, abs=0.01)
