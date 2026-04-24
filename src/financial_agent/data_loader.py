"""Loads JSON datasets into typed pydantic models with cached accessors."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

from financial_agent.models import (
    Index,
    MacroCorrelations,
    MarketSnapshot,
    MutualFund,
    MutualFundHolding,
    NewsArticle,
    Portfolio,
    PortfolioAnalytics,
    Sector,
    SectorMap,
    SectorPerformance,
    Stock,
    StockHolding,
)


class DataLoader:
    """Single entry point for loading and querying mock data."""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def _read(self, filename: str) -> dict:
        path = self.data_dir / filename
        with path.open() as f:
            return json.load(f)

    @cached_property
    def market(self) -> MarketSnapshot:
        raw = self._read("market_data.json")
        meta = raw["metadata"]
        indices = {k: Index.model_validate(v | {"name": v["name"]}) for k, v in raw["indices"].items()}
        stocks = {k: Stock.model_validate(v | {"symbol": k}) for k, v in raw["stocks"].items()}
        sectors = {
            k: SectorPerformance.model_validate(v | {"sector_code": k})
            for k, v in raw["sector_performance"].items()
        }
        return MarketSnapshot(
            date=meta["date"],
            market_status=meta.get("market_status", "UNKNOWN"),
            currency=meta.get("currency", "INR"),
            indices=indices,
            stocks=stocks,
            sector_performance=sectors,
        )

    @cached_property
    def news(self) -> list[NewsArticle]:
        raw = self._read("news_data.json")
        return [NewsArticle.model_validate(n) for n in raw["news"]]

    @cached_property
    def mutual_funds(self) -> dict[str, MutualFund]:
        raw = self._read("mutual_funds.json")
        return {k: MutualFund.model_validate(v) for k, v in raw["mutual_funds"].items()}

    @cached_property
    def sector_map(self) -> SectorMap:
        raw = self._read("sector_mapping.json")
        sectors = {k: Sector.model_validate(v | {"code": k}) for k, v in raw["sectors"].items()}
        return SectorMap(
            sectors=sectors,
            macro_correlations=MacroCorrelations(correlations=raw["macro_correlations"]),
            defensive_sectors=raw.get("defensive_sectors", []),
            cyclical_sectors=raw.get("cyclical_sectors", []),
            rate_sensitive_sectors=raw.get("rate_sensitive_sectors", []),
            export_oriented_sectors=raw.get("export_oriented_sectors", []),
        )

    @cached_property
    def historical(self) -> dict:
        return self._read("historical_data.json")

    @cached_property
    def _portfolios_raw(self) -> dict:
        return self._read("portfolios.json")["portfolios"]

    @cached_property
    def portfolios(self) -> dict[str, Portfolio]:
        return {pid: self._build_portfolio(pid, raw) for pid, raw in self._portfolios_raw.items()}

    @staticmethod
    def _build_portfolio(pid: str, raw: dict) -> Portfolio:
        stocks = [StockHolding.model_validate(s) for s in raw["holdings"].get("stocks", [])]
        mfs = [MutualFundHolding.model_validate(m) for m in raw["holdings"].get("mutual_funds", [])]
        analytics = (
            PortfolioAnalytics.model_validate(raw["analytics"]) if "analytics" in raw else None
        )
        return Portfolio(
            portfolio_id=pid,
            user_id=raw["user_id"],
            user_name=raw["user_name"],
            portfolio_type=raw["portfolio_type"],
            risk_profile=raw["risk_profile"],
            investment_horizon=raw["investment_horizon"],
            description=raw["description"],
            total_investment=raw["total_investment"],
            current_value=raw["current_value"],
            overall_gain_loss=raw["overall_gain_loss"],
            overall_gain_loss_percent=raw["overall_gain_loss_percent"],
            stocks=stocks,
            mutual_funds=mfs,
            analytics=analytics,
        )

    # --- Convenience accessors -------------------------------------------------

    def get_portfolio(self, portfolio_id: str) -> Portfolio:
        return self.portfolios[portfolio_id]

    def get_stock(self, symbol: str) -> Stock | None:
        return self.market.stocks.get(symbol)

    def get_sector_perf(self, sector_code: str) -> SectorPerformance | None:
        return self.market.sector_performance.get(sector_code)

    def get_news_for_portfolio(self, portfolio: Portfolio) -> list[NewsArticle]:
        """Return news that touches this portfolio's stocks or sectors, plus market-wide HIGH impact."""
        held_stocks = {h.symbol for h in portfolio.stocks}
        held_sectors = {h.sector for h in portfolio.stocks}
        # Mutual funds expose us to their top holdings as well
        for mf in portfolio.mutual_funds:
            held_stocks.update(mf.top_holdings)
            scheme = self.mutual_funds.get(mf.scheme_code)
            if scheme:
                held_sectors.update(scheme.sector_allocation.keys())

        relevant = []
        for article in self.news:
            if article.scope.value == "MARKET_WIDE" and article.impact_level.value == "HIGH":
                relevant.append(article)
            elif any(s in held_sectors for s in article.entities.sectors):
                relevant.append(article)
            elif any(s in held_stocks for s in article.entities.stocks):
                relevant.append(article)
        return relevant
