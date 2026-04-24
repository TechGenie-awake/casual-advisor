"""News article model with sentiment, scope, and entity tags."""

from pydantic import BaseModel, Field

from financial_agent.models.enums import ImpactLevel, NewsScope, Sentiment


class NewsEntities(BaseModel):
    sectors: list[str] = Field(default_factory=list)
    stocks: list[str] = Field(default_factory=list)
    indices: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class NewsArticle(BaseModel):
    id: str
    headline: str
    summary: str
    published_at: str
    source: str
    sentiment: Sentiment
    sentiment_score: float
    scope: NewsScope
    impact_level: ImpactLevel
    entities: NewsEntities
    causal_factors: list[str] = Field(default_factory=list)

    def touches_sector(self, sector: str) -> bool:
        return sector in self.entities.sectors

    def touches_stock(self, symbol: str) -> bool:
        return symbol in self.entities.stocks
