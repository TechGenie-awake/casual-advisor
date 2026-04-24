"""Sector taxonomy and macro correlations."""

from pydantic import BaseModel, Field


class Sector(BaseModel):
    code: str
    description: str
    index: str | None = None
    sub_sectors: list[str] = Field(default_factory=list)
    rate_sensitive: bool = False
    key_metrics: list[str] = Field(default_factory=list)
    stocks: list[str] = Field(default_factory=list)


class MacroCorrelations(BaseModel):
    """For each macro event, sectors that benefit or suffer."""

    correlations: dict[str, dict[str, list[str]]]


class SectorMap(BaseModel):
    sectors: dict[str, Sector]
    macro_correlations: MacroCorrelations
    defensive_sectors: list[str] = Field(default_factory=list)
    cyclical_sectors: list[str] = Field(default_factory=list)
    rate_sensitive_sectors: list[str] = Field(default_factory=list)
    export_oriented_sectors: list[str] = Field(default_factory=list)
