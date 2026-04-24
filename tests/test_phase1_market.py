"""Phase 1.1 — Trend Analysis: index movements → market sentiment.

Brief requirement:
    "Analyze index movements (NIFTY 50, SENSEX) to determine market sentiment
     (Bullish/Bearish/Neutral)."
"""

import pytest

from financial_agent.market import MarketAnalyzer
from financial_agent.models import Sentiment


@pytest.mark.parametrize(
    "change_pct, expected",
    [
        (1.50, Sentiment.BULLISH),
        (0.50, Sentiment.BULLISH),       # boundary
        (0.49, Sentiment.NEUTRAL),
        (0.00, Sentiment.NEUTRAL),
        (-0.49, Sentiment.NEUTRAL),
        (-0.50, Sentiment.BEARISH),      # boundary
        (-2.33, Sentiment.BEARISH),
    ],
)
def test_classify_thresholds(change_pct, expected):
    assert MarketAnalyzer.classify(change_pct) == expected


def test_overall_market_sentiment_is_bearish_on_2026_04_21(loader):
    """NIFTY50 -1.00% + SENSEX -0.99% should produce BEARISH overall."""
    mi = MarketAnalyzer(loader.market).analyze()
    assert mi.overall_sentiment == Sentiment.BEARISH
    assert mi.benchmark_change_percent == pytest.approx(-0.995, abs=0.01)


def test_each_index_classified_from_movement(loader):
    """Per-index sentiment must be re-derived from change_percent, not blindly trusted."""
    mi = MarketAnalyzer(loader.market).analyze()
    by_code = {v.code: v for v in mi.indices}
    assert by_code["NIFTY50"].sentiment == Sentiment.BEARISH
    assert by_code["BANKNIFTY"].sentiment == Sentiment.BEARISH
    assert by_code["NIFTYIT"].sentiment == Sentiment.BULLISH
    assert by_code["NIFTYPHARMA"].sentiment == Sentiment.BULLISH


def test_divergence_notes_capture_outperforming_indices(loader):
    """When IT/PHARMA rally on a bearish day, the analyzer should flag the divergence."""
    mi = MarketAnalyzer(loader.market).analyze()
    note_blob = " ".join(mi.notes)
    assert "NIFTY IT" in note_blob
    assert "NIFTY PHARMA" in note_blob
