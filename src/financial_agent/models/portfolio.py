"""Portfolio holdings and analytics models."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StockHolding(BaseModel):
    symbol: str
    name: str
    sector: str
    quantity: float
    avg_buy_price: float
    current_price: float
    investment_value: float
    current_value: float
    gain_loss: float
    gain_loss_percent: float
    day_change: float
    day_change_percent: float
    weight_in_portfolio: float


class MutualFundHolding(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _normalize_nav_key(cls, data: Any) -> Any:
        # Source data has a typo on one fund (current_price → current_nav).
        if isinstance(data, dict) and "current_nav" not in data and "current_price" in data:
            data = {**data, "current_nav": data["current_price"]}
        return data

    scheme_code: str
    scheme_name: str
    category: str
    amc: str
    units: float
    avg_nav: float
    current_nav: float
    investment_value: float
    current_value: float
    gain_loss: float
    gain_loss_percent: float
    day_change: float
    day_change_percent: float
    weight_in_portfolio: float
    top_holdings: list[str] = Field(default_factory=list)


class PortfolioAnalytics(BaseModel):
    """Pre-computed analytics from the source data; useful for verification."""

    sector_allocation: dict[str, float] = Field(default_factory=dict)
    asset_type_allocation: dict[str, float] = Field(default_factory=dict)
    risk_metrics: dict = Field(default_factory=dict)
    day_summary: dict = Field(default_factory=dict)


class Portfolio(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    portfolio_id: str
    user_id: str
    user_name: str
    portfolio_type: str
    risk_profile: str
    investment_horizon: str
    description: str
    total_investment: float
    current_value: float
    overall_gain_loss: float
    overall_gain_loss_percent: float
    stocks: list[StockHolding] = Field(default_factory=list)
    mutual_funds: list[MutualFundHolding] = Field(default_factory=list)
    analytics: PortfolioAnalytics | None = None
