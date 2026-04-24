"""Phase 2 — Portfolio P&L, allocation, and risk detection.

All numbers are recomputed from raw holdings (quantity × price, units × NAV).
The pre-shipped `analytics` block on each portfolio is used only for
verification by the smoke-test, never as input.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from financial_agent.models import (
    AssetType,
    MutualFund,
    MutualFundHolding,
    Portfolio,
    StockHolding,
)


# Risk thresholds — sourced directly from the assignment brief
# ("e.g., >40% exposure to a single sector").
SECTOR_CONCENTRATION_WARN = 40.0       # %
SECTOR_CONCENTRATION_CRITICAL = 60.0   # %
STOCK_CONCENTRATION_WARN = 15.0        # %
STOCK_CONCENTRATION_CRITICAL = 25.0    # %
RATE_SENSITIVE_CONCENTRATION = 60.0    # %


class RiskSeverity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class RiskFlag:
    code: str
    severity: RiskSeverity
    message: str
    metric: float | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "metric": None if self.metric is None else round(self.metric, 2),
        }


@dataclass(frozen=True)
class HoldingMove:
    """A single position's contribution to the day's move."""

    identifier: str       # symbol or scheme_code
    name: str
    asset_type: AssetType
    weight_percent: float
    day_change_percent: float
    day_change_value: float

    def to_dict(self) -> dict:
        return {
            "identifier": self.identifier,
            "name": self.name,
            "asset_type": self.asset_type.value,
            "weight_percent": round(self.weight_percent, 2),
            "day_change_percent": round(self.day_change_percent, 2),
            "day_change_value": round(self.day_change_value, 2),
        }


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Output of Phase 2 — everything downstream layers need.

    All allocations sum to ~100. `sector_allocation_lookthrough` decomposes
    mutual funds to their underlying sector exposures for a true risk view.
    """

    portfolio_id: str
    user_name: str
    risk_profile: str

    total_invested: float
    current_value: float
    overall_pnl: float
    overall_pnl_percent: float

    day_pnl: float
    day_pnl_percent: float

    asset_allocation: dict[str, float]
    sector_allocation: dict[str, float]            # MFs bucketed by category
    sector_allocation_lookthrough: dict[str, float]  # MFs decomposed to underlying sectors

    top_gainers: list[HoldingMove]
    top_losers: list[HoldingMove]

    risks: list[RiskFlag]

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "user_name": self.user_name,
            "risk_profile": self.risk_profile,
            "total_invested": round(self.total_invested, 2),
            "current_value": round(self.current_value, 2),
            "overall_pnl": round(self.overall_pnl, 2),
            "overall_pnl_percent": round(self.overall_pnl_percent, 2),
            "day_pnl": round(self.day_pnl, 2),
            "day_pnl_percent": round(self.day_pnl_percent, 2),
            "asset_allocation": {k: round(v, 2) for k, v in self.asset_allocation.items()},
            "sector_allocation": {k: round(v, 2) for k, v in self.sector_allocation.items()},
            "sector_allocation_lookthrough": {
                k: round(v, 2) for k, v in self.sector_allocation_lookthrough.items()
            },
            "top_gainers": [m.to_dict() for m in self.top_gainers],
            "top_losers": [m.to_dict() for m in self.top_losers],
            "risks": [r.to_dict() for r in self.risks],
        }


def _classify_mf(mf: MutualFund | None, holding: MutualFundHolding) -> AssetType:
    """Coarse equity / debt / hybrid classification.

    If we have the full MutualFund object we use its category; otherwise we
    fall back to the holding's category string.
    """
    category = (mf.category if mf else holding.category).upper()
    if "DEBT" in category or category in {
        "LIQUID",
        "CORPORATE_BOND",
        "GILT",
        "ULTRA_SHORT_DURATION",
    }:
        return AssetType.DEBT_MUTUAL_FUND
    if "HYBRID" in category or category in {
        "BALANCED_ADVANTAGE",
        "AGGRESSIVE_HYBRID",
        "CONSERVATIVE_HYBRID",
    }:
        return AssetType.HYBRID_MUTUAL_FUND
    return AssetType.EQUITY_MUTUAL_FUND


def _mf_bucket_label(asset_type: AssetType, holding: MutualFundHolding) -> str:
    """Human-readable allocation bucket for a mutual fund."""
    if asset_type == AssetType.DEBT_MUTUAL_FUND:
        return "DEBT_FUNDS"
    if asset_type == AssetType.HYBRID_MUTUAL_FUND:
        return "HYBRID_FUNDS"
    cat = holding.category.upper()
    if cat.startswith("SECTORAL_"):
        # Surface the sector if it's a single-sector fund.
        return cat.replace("SECTORAL_", "SECTORAL_")
    return "DIVERSIFIED_MF"


class PortfolioAnalyzer:
    """Compute P&L, allocations, and risks for a single portfolio."""

    def __init__(
        self,
        portfolio: Portfolio,
        mutual_funds: dict[str, MutualFund] | None = None,
        rate_sensitive_sectors: list[str] | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._mfs = mutual_funds or {}
        self._rate_sensitive = set(rate_sensitive_sectors or [])

    # --- P&L --------------------------------------------------------------------

    def _compute_pnl(self) -> tuple[float, float, float, float, float, float]:
        """Recompute total invested / current / P&L from raw holdings."""
        invested = 0.0
        current = 0.0
        day_change = 0.0

        for s in self._portfolio.stocks:
            invested += s.quantity * s.avg_buy_price
            current += s.quantity * s.current_price
            # day_change is already absolute ₹; keep it as the source of truth
            # because it includes intraday flow data we don't otherwise have.
            day_change += s.day_change

        for m in self._portfolio.mutual_funds:
            invested += m.units * m.avg_nav
            current += m.units * m.current_nav
            day_change += m.day_change

        overall_pnl = current - invested
        overall_pnl_pct = (overall_pnl / invested * 100) if invested else 0.0

        # Day's % return is day_change relative to *previous* portfolio value,
        # which is current_value − day_change.
        prev_value = current - day_change
        day_pnl_pct = (day_change / prev_value * 100) if prev_value else 0.0

        return invested, current, overall_pnl, overall_pnl_pct, day_change, day_pnl_pct

    # --- Allocation -------------------------------------------------------------

    def _asset_allocation(self, current_value: float) -> dict[str, float]:
        if current_value <= 0:
            return {}
        bucket: dict[str, float] = defaultdict(float)
        for s in self._portfolio.stocks:
            bucket[AssetType.DIRECT_STOCK.value] += s.quantity * s.current_price
        for m in self._portfolio.mutual_funds:
            asset_type = _classify_mf(self._mfs.get(m.scheme_code), m)
            bucket[asset_type.value] += m.units * m.current_nav
        return {k: v / current_value * 100 for k, v in bucket.items()}

    def _sector_allocation(self, current_value: float) -> dict[str, float]:
        """MFs bucketed by category (matches the source data convention)."""
        if current_value <= 0:
            return {}
        bucket: dict[str, float] = defaultdict(float)
        for s in self._portfolio.stocks:
            bucket[s.sector] += s.quantity * s.current_price
        for m in self._portfolio.mutual_funds:
            asset_type = _classify_mf(self._mfs.get(m.scheme_code), m)
            label = _mf_bucket_label(asset_type, m)
            bucket[label] += m.units * m.current_nav
        return {k: v / current_value * 100 for k, v in bucket.items()}

    def _sector_allocation_lookthrough(self, current_value: float) -> dict[str, float]:
        """MFs decomposed into their underlying sector exposures.

        Equity/hybrid MFs use their published `sector_allocation`. Debt funds
        stay in a `DEBT_INSTRUMENTS` bucket because they have no equity
        sector exposure.
        """
        if current_value <= 0:
            return {}
        bucket: dict[str, float] = defaultdict(float)

        for s in self._portfolio.stocks:
            bucket[s.sector] += s.quantity * s.current_price

        for m in self._portfolio.mutual_funds:
            mf_value = m.units * m.current_nav
            mf_obj = self._mfs.get(m.scheme_code)
            asset_type = _classify_mf(mf_obj, m)

            if asset_type == AssetType.DEBT_MUTUAL_FUND:
                bucket["DEBT_INSTRUMENTS"] += mf_value
                continue

            sector_alloc = mf_obj.sector_allocation if mf_obj else {}
            if not sector_alloc:
                # Unknown sector decomposition — keep as a generic bucket.
                bucket["UNCATEGORIZED_MF"] += mf_value
                continue

            total_weight = sum(sector_alloc.values())
            if total_weight <= 0:
                bucket["UNCATEGORIZED_MF"] += mf_value
                continue

            for sector, weight in sector_alloc.items():
                bucket[sector] += mf_value * (weight / total_weight)

        return {k: v / current_value * 100 for k, v in bucket.items()}

    # --- Movers -----------------------------------------------------------------

    def _movers(self, current_value: float) -> tuple[list[HoldingMove], list[HoldingMove]]:
        moves: list[HoldingMove] = []

        for s in self._portfolio.stocks:
            value = s.quantity * s.current_price
            weight = (value / current_value * 100) if current_value else 0.0
            moves.append(
                HoldingMove(
                    identifier=s.symbol,
                    name=s.name,
                    asset_type=AssetType.DIRECT_STOCK,
                    weight_percent=weight,
                    day_change_percent=s.day_change_percent,
                    day_change_value=s.day_change,
                )
            )

        for m in self._portfolio.mutual_funds:
            value = m.units * m.current_nav
            weight = (value / current_value * 100) if current_value else 0.0
            asset_type = _classify_mf(self._mfs.get(m.scheme_code), m)
            moves.append(
                HoldingMove(
                    identifier=m.scheme_code,
                    name=m.scheme_name,
                    asset_type=asset_type,
                    weight_percent=weight,
                    day_change_percent=m.day_change_percent,
                    day_change_value=m.day_change,
                )
            )

        ranked = sorted(moves, key=lambda x: x.day_change_percent, reverse=True)
        gainers = [m for m in ranked if m.day_change_percent > 0][:3]
        losers = [m for m in reversed(ranked) if m.day_change_percent < 0][:3]
        return gainers, losers

    # --- Risk detection ---------------------------------------------------------

    def _detect_risks(
        self,
        sector_alloc: dict[str, float],
        lookthrough_alloc: dict[str, float],
        stock_weights: dict[str, float],
    ) -> list[RiskFlag]:
        flags: list[RiskFlag] = []

        # --- 1. Single-sector concentration (raw bucket) --------------------
        for sector, weight in sector_alloc.items():
            if weight >= SECTOR_CONCENTRATION_CRITICAL:
                flags.append(
                    RiskFlag(
                        code="SECTOR_CONCENTRATION",
                        severity=RiskSeverity.CRITICAL,
                        message=f"CRITICAL: {weight:.1f}% concentrated in {sector}",
                        metric=weight,
                    )
                )
            elif weight >= SECTOR_CONCENTRATION_WARN:
                flags.append(
                    RiskFlag(
                        code="SECTOR_CONCENTRATION",
                        severity=RiskSeverity.WARN,
                        message=f"Elevated exposure: {weight:.1f}% in {sector}",
                        metric=weight,
                    )
                )

        # --- 2. Single-stock concentration ----------------------------------
        for symbol, weight in stock_weights.items():
            if weight >= STOCK_CONCENTRATION_CRITICAL:
                flags.append(
                    RiskFlag(
                        code="STOCK_CONCENTRATION",
                        severity=RiskSeverity.CRITICAL,
                        message=f"CRITICAL: single stock {symbol} is {weight:.1f}% of portfolio",
                        metric=weight,
                    )
                )
            elif weight >= STOCK_CONCENTRATION_WARN:
                flags.append(
                    RiskFlag(
                        code="STOCK_CONCENTRATION",
                        severity=RiskSeverity.WARN,
                        message=f"Elevated single-stock weight: {symbol} at {weight:.1f}%",
                        metric=weight,
                    )
                )

        # --- 3. Aggregate rate-sensitive exposure (look-through) ------------
        if self._rate_sensitive:
            rate_exposure = sum(
                w for sector, w in lookthrough_alloc.items() if sector in self._rate_sensitive
            )
            if rate_exposure >= RATE_SENSITIVE_CONCENTRATION:
                flags.append(
                    RiskFlag(
                        code="RATE_SENSITIVE_EXPOSURE",
                        severity=RiskSeverity.CRITICAL,
                        message=(
                            f"CRITICAL: {rate_exposure:.1f}% in rate-sensitive sectors "
                            "(banking, realty, financial services, autos, infra)"
                        ),
                        metric=rate_exposure,
                    )
                )
            elif rate_exposure >= 40.0:
                flags.append(
                    RiskFlag(
                        code="RATE_SENSITIVE_EXPOSURE",
                        severity=RiskSeverity.WARN,
                        message=f"Elevated rate-sensitive exposure: {rate_exposure:.1f}%",
                        metric=rate_exposure,
                    )
                )

        # --- 4. MF / direct-stock overlap (look-through duplication) --------
        # Detects "I hold HDFCBANK directly and also via my Banking MF".
        direct_symbols = {s.symbol for s in self._portfolio.stocks}
        overlap_symbols: set[str] = set()
        for m in self._portfolio.mutual_funds:
            for top in m.top_holdings:
                if top in direct_symbols:
                    overlap_symbols.add(top)
        if overlap_symbols:
            flags.append(
                RiskFlag(
                    code="DIRECT_MF_OVERLAP",
                    severity=RiskSeverity.INFO,
                    message=(
                        f"You hold {', '.join(sorted(overlap_symbols))} both directly and via "
                        f"a mutual fund — true exposure is higher than the direct weight suggests."
                    ),
                )
            )

        return flags

    # --- Public entry point -----------------------------------------------------

    def snapshot(self) -> PortfolioSnapshot:
        invested, current, overall_pnl, overall_pnl_pct, day_pnl, day_pnl_pct = self._compute_pnl()

        asset_alloc = self._asset_allocation(current)
        sector_alloc = self._sector_allocation(current)
        lookthrough = self._sector_allocation_lookthrough(current)

        stock_weights = {
            s.symbol: (s.quantity * s.current_price / current * 100) if current else 0.0
            for s in self._portfolio.stocks
        }

        gainers, losers = self._movers(current)
        risks = self._detect_risks(sector_alloc, lookthrough, stock_weights)

        return PortfolioSnapshot(
            portfolio_id=self._portfolio.portfolio_id,
            user_name=self._portfolio.user_name,
            risk_profile=self._portfolio.risk_profile,
            total_invested=invested,
            current_value=current,
            overall_pnl=overall_pnl,
            overall_pnl_percent=overall_pnl_pct,
            day_pnl=day_pnl,
            day_pnl_percent=day_pnl_pct,
            asset_allocation=asset_alloc,
            sector_allocation=sector_alloc,
            sector_allocation_lookthrough=lookthrough,
            top_gainers=gainers,
            top_losers=losers,
            risks=risks,
        )
