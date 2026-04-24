"""Mutual fund schema."""

from pydantic import BaseModel, Field


class MutualFundReturns(BaseModel):
    one_day: float = Field(alias="1_day")
    one_week: float = Field(alias="1_week")
    one_month: float = Field(alias="1_month")
    three_month: float = Field(alias="3_month")
    six_month: float = Field(alias="6_month")
    one_year: float = Field(alias="1_year")
    three_year_cagr: float = Field(alias="3_year_cagr")
    five_year_cagr: float = Field(alias="5_year_cagr")

    model_config = {"populate_by_name": True}


class MFSectorWeight(BaseModel):
    """Equity MFs expose stock+sector; debt MFs expose issuer+rating. Both share `weight`."""

    stock: str | None = None
    sector: str | None = None
    issuer: str | None = None
    rating: str | None = None
    weight: float


class MutualFund(BaseModel):
    scheme_code: str
    scheme_name: str
    amc: str
    category: str
    sub_category: str
    risk_rating: str
    current_nav: float
    previous_nav: float
    nav_change: float
    nav_change_percent: float
    aum_cr: float
    expense_ratio: float
    benchmark: str
    fund_manager: str
    inception_date: str
    returns: MutualFundReturns
    top_holdings: list[MFSectorWeight] = Field(default_factory=list)
    sector_allocation: dict[str, float] = Field(default_factory=dict)
    portfolio_characteristics: dict[str, float] = Field(default_factory=dict)

    @property
    def is_debt(self) -> bool:
        return "DEBT" in self.category.upper() or self.category in {
            "LIQUID",
            "CORPORATE_BOND",
            "GILT",
            "ULTRA_SHORT_DURATION",
        }

    @property
    def is_hybrid(self) -> bool:
        return "HYBRID" in self.category.upper() or self.category in {
            "BALANCED_ADVANTAGE",
            "AGGRESSIVE_HYBRID",
            "CONSERVATIVE_HYBRID",
        }

    @property
    def is_equity(self) -> bool:
        return not (self.is_debt or self.is_hybrid)
