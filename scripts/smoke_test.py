"""End-to-end smoke test for Phase 1 + Phase 2.

Runs both layers against the three sample portfolios and prints a structured
report. Also cross-checks our recomputed analytics against the values shipped
in the source data — small deltas are expected (we use slightly different
allocation conventions for mutual funds), but P&L should match exactly.

Run:
    PYTHONPATH=src python3 scripts/smoke_test.py
"""

from __future__ import annotations

import json
from pathlib import Path

from financial_agent.data_loader import DataLoader
from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.portfolio import PortfolioAnalyzer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def fmt_money(x: float) -> str:
    return f"₹{x:,.0f}"


def main() -> None:
    loader = DataLoader(DATA_DIR)

    # =========================================================================
    # PHASE 1 — Market Intelligence
    # =========================================================================
    hr("PHASE 1.1 — Index trend & overall market sentiment")
    market = MarketAnalyzer(loader.market).analyze()
    print(f"date           : {market.date}")
    print(f"overall        : {market.overall_sentiment.value}")
    print(f"benchmark avg  : {fmt_pct(market.benchmark_change_percent)}")
    print()
    print(f"{'INDEX':<14}{'CHANGE':>10}   SENTIMENT")
    for v in market.indices:
        print(f"  {v.code:<12}{fmt_pct(v.change_percent):>10}   {v.sentiment.value}")
    if market.notes:
        print()
        print("Divergence notes:")
        for n in market.notes:
            print(f"  - {n}")

    hr("PHASE 1.2 — Sector trends (recomputed from stock data)")
    sectors = SectorAnalyzer(loader.market)
    print(f"{'SECTOR':<25}{'WTD':>9}{'AVG':>9}   SENT  STOCKS")
    for t in sectors.ranked():
        print(
            f"  {t.sector_code:<23}"
            f"{fmt_pct(t.weighted_change_percent):>9}"
            f"{fmt_pct(t.avg_change_percent):>9}"
            f"   {t.sentiment.value:<7}{t.constituent_count}"
        )

    print()
    print("Stock-vs-sector divergences (Phase 3 conflict candidates):")
    for sym, sec, sc, sec_c in sectors.divergences():
        print(f"  {sym:<12} ({sec:<25}) stock {fmt_pct(sc)} vs sector {fmt_pct(sec_c)}")

    hr("PHASE 1.3 — News classification & prioritization")
    np_ = NewsProcessor(loader.news)
    summary = np_.summary()
    print("Counts:")
    print(f"  total: {summary.total}")
    print(f"  by_scope:     {summary.by_scope}")
    print(f"  by_sentiment: {summary.by_sentiment}")
    print(f"  by_impact:    {summary.by_impact}")
    print()
    print("Top 5 by priority score:")
    for a in np_.top(5):
        print(
            f"  {a.id} prio={NewsProcessor.priority_score(a):.2f} "
            f"[{a.scope.value}/{a.impact_level.value}/{a.sentiment.value}]"
        )
        print(f"     {a.headline}")

    # Conflict candidates use sector sentiments from our recomputed trends.
    sector_sentiments = {code: t.sentiment for code, t in sectors.all_trends().items()}
    conflicts = np_.conflict_candidates(sector_sentiments)
    print()
    print(f"News whose sentiment opposes its sector's mood: {len(conflicts)}")
    for a in conflicts:
        print(
            f"  {a.id} [{a.sentiment.value}] in {a.entities.sectors} "
            f"-> {a.headline[:75]}"
        )

    # =========================================================================
    # PHASE 2 — Portfolio Analytics (run on all three portfolios)
    # =========================================================================
    rate_sensitive = loader.sector_map.rate_sensitive_sectors

    for pid in ("PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"):
        portfolio = loader.get_portfolio(pid)
        snap = PortfolioAnalyzer(
            portfolio,
            mutual_funds=loader.mutual_funds,
            rate_sensitive_sectors=rate_sensitive,
        ).snapshot()

        hr(f"PHASE 2 — {pid}: {snap.user_name} ({snap.risk_profile})")
        print(f"Invested      : {fmt_money(snap.total_invested)}")
        print(f"Current value : {fmt_money(snap.current_value)}")
        print(f"Overall P&L   : {fmt_money(snap.overall_pnl)}  ({fmt_pct(snap.overall_pnl_percent)})")
        print(f"Day P&L       : {fmt_money(snap.day_pnl)}  ({fmt_pct(snap.day_pnl_percent)})")

        print()
        print("Asset allocation:")
        for k, v in sorted(snap.asset_allocation.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<25} {v:6.2f}%")

        print()
        print("Sector allocation (MFs as buckets):")
        for k, v in sorted(snap.sector_allocation.items(), key=lambda kv: -kv[1]):
            print(f"  {k:<25} {v:6.2f}%")

        print()
        print("Sector allocation (look-through MF holdings):")
        for k, v in sorted(snap.sector_allocation_lookthrough.items(), key=lambda kv: -kv[1])[:8]:
            print(f"  {k:<25} {v:6.2f}%")

        print()
        print("Top gainers today:")
        for m in snap.top_gainers:
            print(f"  {m.identifier:<10} {fmt_pct(m.day_change_percent):>7}  weight {m.weight_percent:5.2f}%  {m.name[:40]}")
        print("Top losers today:")
        for m in snap.top_losers:
            print(f"  {m.identifier:<10} {fmt_pct(m.day_change_percent):>7}  weight {m.weight_percent:5.2f}%  {m.name[:40]}")

        print()
        print("Risk flags:")
        if not snap.risks:
            print("  (none)")
        for r in snap.risks:
            print(f"  [{r.severity.value}] {r.code}: {r.message}")

        # ---- Verification against shipped analytics --------------------------
        if portfolio.analytics:
            shipped_day = portfolio.analytics.day_summary.get("day_change_absolute")
            shipped_day_pct = portfolio.analytics.day_summary.get("day_change_percent")
            shipped_value = portfolio.current_value
            print()
            print("Verification vs shipped analytics:")
            print(f"  current_value : ours {snap.current_value:,.2f}  vs shipped {shipped_value:,.2f}")
            if shipped_day is not None:
                print(f"  day_pnl       : ours {snap.day_pnl:,.2f}      vs shipped {shipped_day:,.2f}")
            if shipped_day_pct is not None:
                print(f"  day_pnl_pct   : ours {snap.day_pnl_percent:+.2f}%       vs shipped {shipped_day_pct:+.2f}%")
            shipped_sectors = portfolio.analytics.sector_allocation
            if shipped_sectors:
                shared = set(snap.sector_allocation) & set(shipped_sectors)
                if shared:
                    deltas = [
                        (s, snap.sector_allocation[s] - shipped_sectors[s]) for s in shared
                    ]
                    worst = max(deltas, key=lambda d: abs(d[1]))
                    print(f"  largest sector-allocation delta: {worst[0]} {worst[1]:+.2f}pp")

    # =========================================================================
    # JSON dump of full output for downstream layers
    # =========================================================================
    hr("JSON sample — PORTFOLIO_002 snapshot (first 1500 chars)")
    snap2 = PortfolioAnalyzer(
        loader.get_portfolio("PORTFOLIO_002"),
        mutual_funds=loader.mutual_funds,
        rate_sensitive_sectors=rate_sensitive,
    ).snapshot()
    print(json.dumps(snap2.to_dict(), indent=2)[:1500])


if __name__ == "__main__":
    main()
