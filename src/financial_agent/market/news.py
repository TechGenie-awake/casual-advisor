"""News classification and prioritization.

The mock data already ships pre-tagged sentiment + scope. This module is the
*structured access layer* — filtering, ranking, and reshaping for downstream
consumers. We do not re-run NLP here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from financial_agent.models import ImpactLevel, NewsArticle, NewsScope, Sentiment


# Numeric weights for impact, used in priority scoring.
_IMPACT_WEIGHT: dict[ImpactLevel, float] = {
    ImpactLevel.HIGH: 1.0,
    ImpactLevel.MEDIUM: 0.6,
    ImpactLevel.LOW: 0.3,
}

# Numeric weights for scope; market-wide news affects more holdings on average.
_SCOPE_WEIGHT: dict[NewsScope, float] = {
    NewsScope.MARKET_WIDE: 1.0,
    NewsScope.SECTOR_SPECIFIC: 0.8,
    NewsScope.STOCK_SPECIFIC: 0.6,
}


@dataclass(frozen=True)
class NewsSummary:
    """Counts and breakdowns across the full news feed."""

    total: int
    by_scope: dict[str, int]
    by_sentiment: dict[str, int]
    by_impact: dict[str, int]
    high_impact_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "by_scope": self.by_scope,
            "by_sentiment": self.by_sentiment,
            "by_impact": self.by_impact,
            "high_impact_ids": self.high_impact_ids,
        }


class NewsProcessor:
    """Filter, rank, and summarize news articles."""

    def __init__(self, articles: list[NewsArticle]) -> None:
        self._articles = articles

    @property
    def articles(self) -> list[NewsArticle]:
        return list(self._articles)

    @staticmethod
    def priority_score(article: NewsArticle) -> float:
        """Combined impact × scope × |sentiment| score in [0, 1]."""
        impact = _IMPACT_WEIGHT[article.impact_level]
        scope = _SCOPE_WEIGHT[article.scope]
        # Normalize sentiment magnitude — strong directional signals score higher.
        sentiment_strength = min(1.0, abs(article.sentiment_score))
        return round(impact * scope * (0.4 + 0.6 * sentiment_strength), 3)

    def by_scope(self, scope: NewsScope) -> list[NewsArticle]:
        return [a for a in self._articles if a.scope == scope]

    def by_impact(self, level: ImpactLevel) -> list[NewsArticle]:
        return [a for a in self._articles if a.impact_level == level]

    def for_sector(self, sector: str) -> list[NewsArticle]:
        return [a for a in self._articles if a.touches_sector(sector)]

    def for_stock(self, symbol: str) -> list[NewsArticle]:
        return [a for a in self._articles if a.touches_stock(symbol)]

    def top(self, n: int = 5) -> list[NewsArticle]:
        return sorted(self._articles, key=self.priority_score, reverse=True)[:n]

    def filter_for_holdings(
        self,
        symbols: set[str],
        sectors: set[str],
        include_market_wide_high_impact: bool = True,
    ) -> list[NewsArticle]:
        """Return only news that could plausibly affect the given holdings."""
        out: list[NewsArticle] = []
        for a in self._articles:
            if include_market_wide_high_impact and a.scope == NewsScope.MARKET_WIDE and a.impact_level == ImpactLevel.HIGH:
                out.append(a)
                continue
            if any(s in sectors for s in a.entities.sectors):
                out.append(a)
                continue
            if any(s in symbols for s in a.entities.stocks):
                out.append(a)
        # Dedupe while preserving order.
        seen: set[str] = set()
        unique: list[NewsArticle] = []
        for a in out:
            if a.id not in seen:
                seen.add(a.id)
                unique.append(a)
        return sorted(unique, key=self.priority_score, reverse=True)

    def summary(self) -> NewsSummary:
        scopes = Counter(a.scope.value for a in self._articles)
        sentiments = Counter(a.sentiment.value for a in self._articles)
        impacts = Counter(a.impact_level.value for a in self._articles)
        high = [a.id for a in self._articles if a.impact_level == ImpactLevel.HIGH]
        return NewsSummary(
            total=len(self._articles),
            by_scope=dict(scopes),
            by_sentiment=dict(sentiments),
            by_impact=dict(impacts),
            high_impact_ids=high,
        )

    def conflict_candidates(self, sector_sentiments: dict[str, Sentiment]) -> list[NewsArticle]:
        """News whose sentiment opposes the prevailing sector mood.

        Example: a POSITIVE article on a stock in a BEARISH sector. These are
        the cases the agent should explicitly call out as conflicting signals.
        """
        out: list[NewsArticle] = []
        bullish_news = {Sentiment.POSITIVE, Sentiment.BULLISH}
        bearish_news = {Sentiment.NEGATIVE, Sentiment.BEARISH}
        for a in self._articles:
            for sector in a.entities.sectors:
                mood = sector_sentiments.get(sector)
                if mood is None:
                    continue
                if a.sentiment in bullish_news and mood == Sentiment.BEARISH:
                    out.append(a)
                    break
                if a.sentiment in bearish_news and mood == Sentiment.BULLISH:
                    out.append(a)
                    break
        return out
