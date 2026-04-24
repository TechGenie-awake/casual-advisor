"""Phase 2.2 — Asset Allocation: breakdown by sector and asset type.

Brief requirement:
    "Asset Allocation: Breakdown by sector and asset type."
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
def test_asset_allocation_sums_to_100(loader, pid):
    snap = _snap(loader, pid)
    assert sum(snap.asset_allocation.values()) == pytest.approx(100.0, abs=0.5)


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
def test_sector_allocation_sums_to_100(loader, pid):
    snap = _snap(loader, pid)
    assert sum(snap.sector_allocation.values()) == pytest.approx(100.0, abs=0.5)


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
def test_lookthrough_allocation_sums_to_100(loader, pid):
    snap = _snap(loader, pid)
    assert sum(snap.sector_allocation_lookthrough.values()) == pytest.approx(100.0, abs=0.5)


def test_p2_is_dominated_by_banking(loader):
    """Portfolio 2 (Priya) must show BANKING as the top sector."""
    snap = _snap(loader, "PORTFOLIO_002")
    top_sector = max(snap.sector_allocation.items(), key=lambda kv: kv[1])
    assert top_sector[0] == "BANKING"
    assert top_sector[1] >= 60.0


def test_p3_is_mutual_fund_heavy(loader):
    """Portfolio 3 (Arun) must show MFs > direct stocks."""
    snap = _snap(loader, "PORTFOLIO_003")
    direct = snap.asset_allocation.get("DIRECT_STOCK", 0.0)
    mf_total = sum(v for k, v in snap.asset_allocation.items() if "MUTUAL_FUND" in k)
    assert mf_total > direct
    assert mf_total >= 70.0


def test_lookthrough_decomposes_mf_into_underlying_sectors(loader):
    """For Portfolio 1, IT exposure under look-through should exceed the raw stock IT weight."""
    snap = _snap(loader, "PORTFOLIO_001")
    raw_it = snap.sector_allocation.get("INFORMATION_TECHNOLOGY", 0.0)
    look_it = snap.sector_allocation_lookthrough.get("INFORMATION_TECHNOLOGY", 0.0)
    assert look_it > raw_it
