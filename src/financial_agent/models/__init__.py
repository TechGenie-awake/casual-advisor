"""Typed data models for all financial entities."""

from financial_agent.models.enums import (
    AssetType,
    ImpactLevel,
    NewsScope,
    Sentiment,
)
from financial_agent.models.market import Index, MarketSnapshot, SectorPerformance, Stock
from financial_agent.models.mutual_fund import MutualFund, MutualFundReturns, MFSectorWeight
from financial_agent.models.news import NewsArticle, NewsEntities
from financial_agent.models.portfolio import (
    MutualFundHolding,
    Portfolio,
    PortfolioAnalytics,
    StockHolding,
)
from financial_agent.models.sector import MacroCorrelations, Sector, SectorMap

__all__ = [
    "AssetType",
    "ImpactLevel",
    "Index",
    "MFSectorWeight",
    "MacroCorrelations",
    "MarketSnapshot",
    "MutualFund",
    "MutualFundHolding",
    "MutualFundReturns",
    "NewsArticle",
    "NewsEntities",
    "NewsScope",
    "Portfolio",
    "PortfolioAnalytics",
    "Sector",
    "SectorMap",
    "SectorPerformance",
    "Sentiment",
    "Stock",
    "StockHolding",
]
