"""Index-level trend analysis and overall market sentiment."""

from __future__ import annotations

from dataclasses import dataclass

from financial_agent.models import Index, MarketSnapshot, Sentiment


# Thresholds for classifying a single index's daily move.
_BULLISH_THRESHOLD = 0.5
_BEARISH_THRESHOLD = -0.5

# The two indices that anchor the overall market view (NIFTY 50 + SENSEX).
_BENCHMARK_INDICES = ("NIFTY50", "SENSEX")


@dataclass(frozen=True)
class IndexView:
    code: str
    name: str
    change_percent: float
    sentiment: Sentiment


@dataclass(frozen=True)
class MarketIntelligence:
    """Output of Phase 1.1 — overall market read."""

    date: str
    overall_sentiment: Sentiment
    benchmark_change_percent: float
    indices: list[IndexView]
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "overall_sentiment": self.overall_sentiment.value,
            "benchmark_change_percent": round(self.benchmark_change_percent, 2),
            "indices": [
                {
                    "code": i.code,
                    "name": i.name,
                    "change_percent": i.change_percent,
                    "sentiment": i.sentiment.value,
                }
                for i in self.indices
            ],
            "notes": self.notes,
        }


class MarketAnalyzer:
    """Computes overall market sentiment from index movements.

    Uses a deterministic rule: average the % change of NIFTY50 + SENSEX, then
    bucket into BULLISH / BEARISH / NEUTRAL. Per-index sentiment is also
    re-derived (we don't trust the source `sentiment` field, even though it
    matches in practice — Phase 1 explicitly says "analyze index movements").
    """

    def __init__(self, snapshot: MarketSnapshot) -> None:
        self._snapshot = snapshot

    @staticmethod
    def classify(change_percent: float) -> Sentiment:
        if change_percent >= _BULLISH_THRESHOLD:
            return Sentiment.BULLISH
        if change_percent <= _BEARISH_THRESHOLD:
            return Sentiment.BEARISH
        return Sentiment.NEUTRAL

    def _index_view(self, code: str, idx: Index) -> IndexView:
        return IndexView(
            code=code,
            name=idx.name,
            change_percent=idx.change_percent,
            sentiment=self.classify(idx.change_percent),
        )

    def analyze(self) -> MarketIntelligence:
        indices = [self._index_view(code, idx) for code, idx in self._snapshot.indices.items()]
        benchmarks = [i for i in indices if i.code in _BENCHMARK_INDICES]
        if not benchmarks:
            benchmarks = indices  # fallback if benchmarks missing
        benchmark_avg = sum(i.change_percent for i in benchmarks) / len(benchmarks)
        overall = self.classify(benchmark_avg)

        notes: list[str] = []
        # Highlight divergences — sectoral indices moving opposite to benchmark.
        for view in indices:
            if view.code in _BENCHMARK_INDICES:
                continue
            if view.sentiment != overall and view.sentiment != Sentiment.NEUTRAL:
                notes.append(
                    f"{view.name} diverged from benchmark "
                    f"({view.change_percent:+.2f}% vs benchmark {benchmark_avg:+.2f}%)"
                )

        return MarketIntelligence(
            date=self._snapshot.date,
            overall_sentiment=overall,
            benchmark_change_percent=benchmark_avg,
            indices=indices,
            notes=notes,
        )
