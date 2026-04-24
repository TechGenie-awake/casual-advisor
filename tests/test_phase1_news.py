"""Phase 1.3 — News Processing: classify by Sentiment and Scope.

Brief requirement:
    "Classify news by Sentiment and Scope (Market-wide, Sector-specific, or
     Stock-specific)."
"""

from financial_agent.market import NewsProcessor, SectorAnalyzer
from financial_agent.models import ImpactLevel, NewsScope, Sentiment


def test_summary_counts_match_total(loader):
    np_ = NewsProcessor(loader.news)
    s = np_.summary()
    assert s.total == len(loader.news)
    assert sum(s.by_scope.values()) == s.total
    assert sum(s.by_sentiment.values()) == s.total
    assert sum(s.by_impact.values()) == s.total


def test_every_article_has_valid_scope_and_sentiment(loader):
    """Validate the full feed parses into the typed enums."""
    valid_scopes = set(NewsScope)
    valid_sentiments = set(Sentiment)
    for a in loader.news:
        assert a.scope in valid_scopes
        assert a.sentiment in valid_sentiments


def test_priority_score_is_in_unit_range(loader):
    np_ = NewsProcessor(loader.news)
    for a in loader.news:
        score = np_.priority_score(a)
        assert 0.0 <= score <= 1.0


def test_high_impact_market_wide_outranks_low_impact_stock(loader):
    """A HIGH/MARKET_WIDE article must outrank a LOW/STOCK_SPECIFIC one."""
    np_ = NewsProcessor(loader.news)
    high_market = next(
        a for a in loader.news
        if a.impact_level == ImpactLevel.HIGH and a.scope == NewsScope.MARKET_WIDE
    )
    low_articles = [a for a in loader.news if a.impact_level == ImpactLevel.LOW]
    if not low_articles:
        return  # nothing to compare
    assert np_.priority_score(high_market) > np_.priority_score(low_articles[0])


def test_filter_for_holdings_returns_only_relevant_news(loader):
    """For a banking-only portfolio, IT-stock news must be excluded."""
    np_ = NewsProcessor(loader.news)
    relevant = np_.filter_for_holdings(
        symbols={"HDFCBANK", "ICICIBANK", "SBIN"},
        sectors={"BANKING", "FINANCIAL_SERVICES"},
        include_market_wide_high_impact=True,
    )
    for a in relevant:
        is_market_wide_high = (
            a.scope == NewsScope.MARKET_WIDE and a.impact_level == ImpactLevel.HIGH
        )
        touches_held_sector = any(
            s in {"BANKING", "FINANCIAL_SERVICES"} for s in a.entities.sectors
        )
        touches_held_stock = any(
            s in {"HDFCBANK", "ICICIBANK", "SBIN"} for s in a.entities.stocks
        )
        assert is_market_wide_high or touches_held_sector or touches_held_stock


def test_conflict_candidates_includes_bajaj_finance(loader):
    """Bajaj Finance positive news inside a bearish FS sector — explicit edge case from README."""
    np_ = NewsProcessor(loader.news)
    sector_sentiments = {
        code: t.sentiment
        for code, t in SectorAnalyzer(loader.market).all_trends().items()
    }
    conflicts = np_.conflict_candidates(sector_sentiments)
    headlines = " ".join(a.headline for a in conflicts)
    assert "Bajaj Finance" in headlines
