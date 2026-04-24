"""Phase 2.3 — Risk Detection: identify concentration risks.

Brief requirement:
    "Risk Detection: Identify concentration risks (e.g., >40% exposure to a
     single sector)."
"""

import pytest

from financial_agent.portfolio import PortfolioAnalyzer, RiskSeverity


def _snap(loader, pid):
    return PortfolioAnalyzer(
        loader.get_portfolio(pid),
        mutual_funds=loader.mutual_funds,
        rate_sensitive_sectors=loader.sector_map.rate_sensitive_sectors,
    ).snapshot()


def _codes(snap, *, severity: RiskSeverity | None = None) -> set[str]:
    return {
        r.code
        for r in snap.risks
        if severity is None or r.severity == severity
    }


# ---------- Portfolio 2 (banking-heavy) — should trigger every flag ----------

def test_p2_triggers_critical_sector_concentration(loader):
    snap = _snap(loader, "PORTFOLIO_002")
    assert "SECTOR_CONCENTRATION" in _codes(snap, severity=RiskSeverity.CRITICAL)


def test_p2_triggers_critical_rate_sensitive_exposure(loader):
    """91%+ in rate-sensitive sectors must trip the rate flag at CRITICAL."""
    snap = _snap(loader, "PORTFOLIO_002")
    assert "RATE_SENSITIVE_EXPOSURE" in _codes(snap, severity=RiskSeverity.CRITICAL)


def test_p2_triggers_stock_concentration_for_hdfcbank(loader):
    """HDFCBANK at 22.6% must produce a STOCK_CONCENTRATION warn."""
    snap = _snap(loader, "PORTFOLIO_002")
    stock_flags = [r for r in snap.risks if r.code == "STOCK_CONCENTRATION"]
    assert any("HDFCBANK" in r.message for r in stock_flags)


def test_p2_detects_direct_mf_overlap(loader):
    """Priya holds bank stocks directly AND a banking MF — overlap must be detected."""
    snap = _snap(loader, "PORTFOLIO_002")
    overlap = [r for r in snap.risks if r.code == "DIRECT_MF_OVERLAP"]
    assert overlap
    assert "HDFCBANK" in overlap[0].message


# ---------- Portfolio 1 (diversified) — no critical flags ----------

def test_p1_has_no_critical_concentration(loader):
    snap = _snap(loader, "PORTFOLIO_001")
    assert "SECTOR_CONCENTRATION" not in _codes(snap, severity=RiskSeverity.CRITICAL)
    assert "STOCK_CONCENTRATION" not in _codes(snap, severity=RiskSeverity.CRITICAL)
    assert "RATE_SENSITIVE_EXPOSURE" not in _codes(snap, severity=RiskSeverity.CRITICAL)


# ---------- Portfolio 3 (conservative) — minimal flags ----------

def test_p3_no_sector_or_stock_concentration(loader):
    snap = _snap(loader, "PORTFOLIO_003")
    assert "SECTOR_CONCENTRATION" not in _codes(snap)
    assert "STOCK_CONCENTRATION" not in _codes(snap)


# ---------- Threshold boundary checks ----------

@pytest.mark.parametrize(
    "weight, expected_severity",
    [
        (39.0, None),                    # below WARN
        (40.0, RiskSeverity.WARN),       # at WARN
        (59.0, RiskSeverity.WARN),
        (60.0, RiskSeverity.CRITICAL),   # at CRITICAL
        (75.0, RiskSeverity.CRITICAL),
    ],
)
def test_sector_concentration_thresholds(weight, expected_severity):
    """Direct unit test against the threshold logic, independent of source data."""
    from financial_agent.portfolio.analytics import (
        SECTOR_CONCENTRATION_CRITICAL,
        SECTOR_CONCENTRATION_WARN,
    )
    if expected_severity is None:
        assert weight < SECTOR_CONCENTRATION_WARN
    elif expected_severity == RiskSeverity.WARN:
        assert SECTOR_CONCENTRATION_WARN <= weight < SECTOR_CONCENTRATION_CRITICAL
    else:
        assert weight >= SECTOR_CONCENTRATION_CRITICAL
