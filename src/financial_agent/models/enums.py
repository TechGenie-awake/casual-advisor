"""Shared enums for sentiment, scope, and asset classification."""

from enum import Enum


class Sentiment(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    MIXED = "MIXED"


class NewsScope(str, Enum):
    MARKET_WIDE = "MARKET_WIDE"
    SECTOR_SPECIFIC = "SECTOR_SPECIFIC"
    STOCK_SPECIFIC = "STOCK_SPECIFIC"


class ImpactLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AssetType(str, Enum):
    DIRECT_STOCK = "DIRECT_STOCK"
    EQUITY_MUTUAL_FUND = "EQUITY_MUTUAL_FUND"
    DEBT_MUTUAL_FUND = "DEBT_MUTUAL_FUND"
    HYBRID_MUTUAL_FUND = "HYBRID_MUTUAL_FUND"
