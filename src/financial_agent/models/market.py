"""Market data: indices, stocks, and sector performance."""

from pydantic import BaseModel, ConfigDict, Field

from financial_agent.models.enums import Sentiment


class Index(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    current_value: float
    previous_close: float
    change_percent: float
    change_absolute: float
    day_high: float
    day_low: float
    week_52_high: float = Field(alias="52_week_high")
    week_52_low: float = Field(alias="52_week_low")
    sentiment: Sentiment


class Stock(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    name: str
    sector: str
    sub_sector: str | None = None
    current_price: float
    previous_close: float
    change_percent: float
    change_absolute: float
    volume: int
    avg_volume_20d: int | None = None
    market_cap_cr: float | None = None
    pe_ratio: float | None = None
    week_52_high: float = Field(alias="52_week_high")
    week_52_low: float = Field(alias="52_week_low")
    beta: float | None = None

    @property
    def volume_ratio(self) -> float | None:
        """Volume vs 20-day average. >1.5 suggests heavy interest."""
        if not self.avg_volume_20d:
            return None
        return self.volume / self.avg_volume_20d


class SectorPerformance(BaseModel):
    sector_code: str
    change_percent: float
    sentiment: Sentiment
    key_drivers: list[str] = Field(default_factory=list)
    top_gainers: list[str] = Field(default_factory=list)
    top_losers: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    """Full market state for a given date."""

    date: str
    market_status: str
    currency: str
    indices: dict[str, Index]
    stocks: dict[str, Stock]
    sector_performance: dict[str, SectorPerformance]
