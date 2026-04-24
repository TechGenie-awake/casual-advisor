"""Phase 4b — Self-evaluation layer.

Verifies the rule-based scorers, the judge XML parser, and the orchestrator
that combines them. No live LLM calls — judge responses are stubbed.
"""

from __future__ import annotations

import pytest

from financial_agent.evaluation import BriefingEvaluator, DimensionScore, EvaluationResult
from financial_agent.evaluation.judge import JudgeParseError, parse_judge_output
from financial_agent.evaluation.rubric import (
    DIMENSIONS,
    WEIGHTS,
    score_causal_depth,
    score_conflict_handling,
    score_evidence_accuracy,
    score_prioritization,
    score_quantification,
)
from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.portfolio import PortfolioAnalyzer
from financial_agent.reasoning import (
    Briefing,
    CausalLink,
    ConflictNote,
    MockLLMClient,
    ReasoningAgent,
    Recommendation,
    build_context,
)


# ---------------------------------------------------------------------------
# Helpers — build a "strong" briefing and a "weak" one for direct rule tests
# ---------------------------------------------------------------------------

def _ctx(loader, pid):
    portfolio = loader.get_portfolio(pid)
    return build_context(
        portfolio,
        market_analyzer=MarketAnalyzer(loader.market),
        sector_analyzer=SectorAnalyzer(loader.market),
        news_processor=NewsProcessor(loader.news),
        portfolio_analyzer=PortfolioAnalyzer(
            portfolio,
            mutual_funds=loader.mutual_funds,
            rate_sensitive_sectors=loader.sector_map.rate_sensitive_sectors,
        ),
        macro_correlations=loader.sector_map.macro_correlations.correlations,
    )


def _strong_briefing(portfolio_id: str) -> Briefing:
    return Briefing(
        portfolio_id=portfolio_id,
        headline="Banking-heavy portfolio fell 2.73% as RBI's hawkish stance hit the sector.",
        causal_chain=[
            CausalLink(
                macro_event="RBI hawkish stance (NEWS001)",
                sector_impact="BANKING -3.11% weighted",
                stock_impact="HDFCBANK -3.51% (22.62% weight, ₹-16,845)",
                portfolio_impact="₹-46,850 of the ₹-57,390 day loss (81.6%)",
                evidence_ids=["NEWS001"],
            ),
            CausalLink(
                macro_event="Global risk-off (NEWS022)",
                sector_impact="FINANCIAL_SERVICES -1.99%",
                stock_impact="BAJFINANCE -2.05% (12% weight, ₹-5,612)",
                portfolio_impact="₹-8,172 of the ₹-57,390 day loss (14.2%)",
                evidence_ids=["NEWS022"],
            ),
        ],
        conflicts=[
            ConflictNote(
                description="INFY positive deal-win news (NEWS008) but low IT allocation (1%)",
                resolution="Banking sector dominance overshadowed IT positives.",
                evidence_ids=["NEWS008"],
            ),
        ],
        recommendations=[Recommendation(text="Reduce banking concentration.", priority="HIGH")],
        confidence=0.85,
        confidence_rationale="Strong macro / sector / stock alignment.",
        evidence_ids=["NEWS001", "NEWS022", "NEWS008"],
    )


def _weak_briefing(portfolio_id: str) -> Briefing:
    return Briefing(
        portfolio_id=portfolio_id,
        headline="Portfolio moved.",
        causal_chain=[
            CausalLink(
                macro_event="Some event",
                sector_impact="A sector moved",
                stock_impact="various stocks",
                portfolio_impact="moved",
                evidence_ids=["FAKE_NEWS_999"],
            ),
        ],
        conflicts=[],
        recommendations=[Recommendation(text="Hold.", priority="LOW")],
        confidence=0.95,
        confidence_rationale="Vibes",
        evidence_ids=["FAKE_NEWS_999"],
    )


# ---------------------------------------------------------------------------
# Rule scorers
# ---------------------------------------------------------------------------

def test_strong_briefing_scores_high_on_causal_depth(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_causal_depth(_strong_briefing("PORTFOLIO_002"), ctx)
    assert score >= 0.95


def test_weak_briefing_loses_points_on_causal_depth(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_causal_depth(_weak_briefing("PORTFOLIO_002"), ctx)
    # Single link with all 5 fields — but single-link penalty drops it.
    assert score < 0.8


def test_evidence_accuracy_full_when_all_ids_in_context(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, critique = score_evidence_accuracy(_strong_briefing("PORTFOLIO_002"), ctx)
    assert score == 1.0
    assert "all in context" in critique


def test_evidence_accuracy_penalizes_hallucinated_ids(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, critique = score_evidence_accuracy(_weak_briefing("PORTFOLIO_002"), ctx)
    assert score <= 0.3
    assert "Hallucinated" in critique


def test_conflict_handling_full_when_all_addressed(loader):
    """Strong briefing has 1 conflict; context has multiple — partial credit only."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_conflict_handling(_strong_briefing("PORTFOLIO_002"), ctx)
    assert 0.0 < score <= 1.0


def test_conflict_handling_zero_when_context_has_conflicts_but_briefing_ignores(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, critique = score_conflict_handling(_weak_briefing("PORTFOLIO_002"), ctx)
    assert score == 0.0
    assert "none addressed" in critique


def test_prioritization_full_when_top_contributor_named(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, critique = score_prioritization(_strong_briefing("PORTFOLIO_002"), ctx)
    assert score == 1.0
    # P2's top contributor is HDFCBANK.
    assert "HDFCBANK" in critique


def test_prioritization_zero_when_top_contributor_missing(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_prioritization(_weak_briefing("PORTFOLIO_002"), ctx)
    assert score == 0.0


def test_quantification_full_when_every_link_has_rupee_and_pct(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_quantification(_strong_briefing("PORTFOLIO_002"), ctx)
    assert score == 1.0


def test_quantification_zero_for_vague_briefing(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    score, _ = score_quantification(_weak_briefing("PORTFOLIO_002"), ctx)
    assert score == 0.0


# ---------------------------------------------------------------------------
# Judge XML parser
# ---------------------------------------------------------------------------

GOLDEN_JUDGE_XML = """<evaluation>
<dimension name="Causal Depth" score="0.85">All links populated.</dimension>
<dimension name="Evidence Accuracy" score="1.00">Citations real.</dimension>
<dimension name="Conflict Handling" score="0.50">One conflict missed.</dimension>
<dimension name="Prioritization" score="0.90">Top contributor leads.</dimension>
<dimension name="Quantification" score="1.00">Every link has ₹ and %.</dimension>
<summary>Solid causal narrative; conflict handling could be sharper.</summary>
</evaluation>"""


def test_judge_parser_extracts_all_dimensions():
    out = parse_judge_output(GOLDEN_JUDGE_XML)
    assert set(out.scores) == set(DIMENSIONS)
    assert out.scores["Causal Depth"].score == 0.85
    assert "Solid" in out.summary


def test_judge_parser_clamps_out_of_range_scores():
    xml = GOLDEN_JUDGE_XML.replace('score="0.85"', 'score="1.7"')
    out = parse_judge_output(xml)
    assert out.scores["Causal Depth"].score == 1.0


def test_judge_parser_strips_markdown_fences():
    wrapped = "```xml\n" + GOLDEN_JUDGE_XML + "\n```"
    out = parse_judge_output(wrapped)
    assert "Causal Depth" in out.scores


def test_judge_parser_raises_on_missing_block():
    with pytest.raises(JudgeParseError):
        parse_judge_output("just prose, no xml")


# ---------------------------------------------------------------------------
# Evaluator end-to-end (rule-only and rule+judge)
# ---------------------------------------------------------------------------

def test_evaluator_rule_only_produces_full_result(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    result = BriefingEvaluator().score(_strong_briefing("PORTFOLIO_002"), ctx)
    assert isinstance(result, EvaluationResult)
    assert result.judge_used is False
    assert len(result.dimensions) == len(DIMENSIONS)
    # Strong briefing should score high overall.
    assert result.overall_score >= 0.7


def test_evaluator_rule_only_weak_briefing_scores_low(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    result = BriefingEvaluator().score(_weak_briefing("PORTFOLIO_002"), ctx)
    assert result.overall_score < 0.5


def test_evaluator_with_judge_blends_scores(loader):
    """Judge scoring everything 0.5; rule says strong; combined should be between."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    flat_judge_xml = """<evaluation>
<dimension name="Causal Depth" score="0.5">x</dimension>
<dimension name="Evidence Accuracy" score="0.5">x</dimension>
<dimension name="Conflict Handling" score="0.5">x</dimension>
<dimension name="Prioritization" score="0.5">x</dimension>
<dimension name="Quantification" score="0.5">x</dimension>
<summary>flat</summary>
</evaluation>"""
    judge = MockLLMClient(flat_judge_xml)
    result = BriefingEvaluator(judge_client=judge).score(
        _strong_briefing("PORTFOLIO_002"), ctx
    )
    assert result.judge_used is True
    # Each dimension's combined = 0.6*rule + 0.4*0.5; rule is high, so > 0.5
    for d in result.dimensions:
        assert d.judge_score == 0.5
        if d.rule_score > 0.5:
            assert d.combined > 0.5


def test_evaluator_degrades_to_rule_only_on_judge_parse_failure(loader):
    """If the judge returns garbage, fall back to rule-only — eval must not crash."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    bad_judge = MockLLMClient("totally not xml at all")
    result = BriefingEvaluator(judge_client=bad_judge).score(
        _strong_briefing("PORTFOLIO_002"), ctx
    )
    assert result.judge_used is False  # gracefully degraded
    assert result.overall_score > 0.0


def test_dimension_weights_sum_to_one():
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_evaluation_result_to_markdown_renders(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    result = BriefingEvaluator().score(_strong_briefing("PORTFOLIO_002"), ctx)
    md = result.to_markdown()
    assert "Self-evaluation" in md
    for dim in DIMENSIONS:
        assert dim in md
