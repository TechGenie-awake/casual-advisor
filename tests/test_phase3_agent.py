"""Phase 3.3 — Reasoning agent end-to-end (with mocked LLM client).

Brief requirements (Phase 3):
    - Causal Linking
    - Conflict Resolution
    - Prioritization

We don't hit the real Anthropic API in tests. Instead we feed canned XML
through the agent and verify the orchestration: parsing, rule-based
confidence scoring, and confidence reconciliation.
"""

import pytest

from financial_agent.market import MarketAnalyzer, NewsProcessor, SectorAnalyzer
from financial_agent.portfolio import PortfolioAnalyzer
from financial_agent.reasoning import (
    MockLLMClient,
    ReasoningAgent,
    build_context,
)


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


STRONG_XML = """<briefing>
<headline>Banking exposure dragged the portfolio down 2.73% on RBI hawkish news.</headline>
<causal_chain>
  <link>
    <macro_event>RBI hawkish stance (NEWS001)</macro_event>
    <sector_impact>BANKING -3.11% weighted</sector_impact>
    <stock_impact>HDFCBANK -3.51%, ICICIBANK -3.13%, SBIN -3.02%</stock_impact>
    <portfolio_impact>Banking exposure of 72% drove ~₹40k of the ₹57k loss.</portfolio_impact>
    <evidence>NEWS001</evidence>
  </link>
  <link>
    <macro_event>FII outflows (NEWS007)</macro_event>
    <sector_impact>FINANCIAL_SERVICES -1.99%</sector_impact>
    <stock_impact>BAJFINANCE -2.05%</stock_impact>
    <portfolio_impact>Added ₹5.6k to losses on the FS allocation.</portfolio_impact>
    <evidence>NEWS007</evidence>
  </link>
</causal_chain>
<conflicts>
  <conflict>
    <description>Bajaj Finance had positive guidance but fell 2%.</description>
    <resolution>Sector-wide rate sensitivity overrode company-specific positives.</resolution>
    <evidence>NEWS011</evidence>
  </conflict>
</conflicts>
<recommendations>
  <recommendation priority="HIGH">Reduce sector concentration below 50%.</recommendation>
</recommendations>
<evidence>NEWS001,NEWS007,NEWS011</evidence>
<confidence>0.85</confidence>
<confidence_rationale>Strong alignment of macro, sector, and stock signals.</confidence_rationale>
</briefing>"""


WEAK_XML = """<briefing>
<headline>Portfolio moved a bit.</headline>
<causal_chain>
  <link>
    <macro_event>Some unrelated event</macro_event>
    <sector_impact>UNKNOWN</sector_impact>
    <stock_impact>various stocks</stock_impact>
    <portfolio_impact>moved</portfolio_impact>
    <evidence>NOT_A_REAL_ID</evidence>
  </link>
</causal_chain>
<conflicts></conflicts>
<recommendations>
  <recommendation priority="LOW">Hold.</recommendation>
</recommendations>
<evidence>NOT_A_REAL_ID</evidence>
<confidence>0.95</confidence>
<confidence_rationale>I am very confident.</confidence_rationale>
</briefing>"""


def test_agent_parses_and_returns_briefing(loader):
    ctx = _ctx(loader, "PORTFOLIO_002")
    agent = ReasoningAgent(client=MockLLMClient(STRONG_XML))
    run = agent.generate(ctx)
    assert run.briefing.portfolio_id == "PORTFOLIO_002"
    assert len(run.briefing.causal_chain) == 2
    assert run.briefing.conflicts
    assert run.briefing.recommendations[0].priority == "HIGH"


def test_strong_briefing_keeps_high_self_reported_confidence(loader):
    """When the LLM cites real high-impact news and references the top sector, trust its score."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    agent = ReasoningAgent(client=MockLLMClient(STRONG_XML))
    run = agent.generate(ctx)
    assert run.rule_based_confidence >= 0.6
    # Within 0.30 → no adjustment, keep the original.
    assert run.briefing.confidence == 0.85


def test_weak_briefing_pulls_overconfident_score_down(loader):
    """Hallucinated evidence + thin chain should trigger reconciliation."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    agent = ReasoningAgent(client=MockLLMClient(WEAK_XML))
    run = agent.generate(ctx)
    assert run.rule_based_confidence < 0.5
    # 0.95 self-reported - 0.X rule = > 0.30 → averaged down
    assert run.briefing.confidence < 0.95
    assert "adjusted" in run.briefing.confidence_rationale.lower()


def test_dry_run_mock_briefing_is_complete(loader):
    """The deterministic mock client must produce a parseable, end-to-end briefing."""
    ctx = _ctx(loader, "PORTFOLIO_002")
    client = MockLLMClient.from_context(ctx)
    agent = ReasoningAgent(client=client)
    run = agent.generate(ctx)
    assert run.briefing.causal_chain
    assert run.briefing.headline
    # Mock client cites real news IDs from the context.
    cited = {eid for link in run.briefing.causal_chain for eid in link.evidence_ids}
    real_ids = set(ctx.evidence_ids())
    assert cited.issubset(real_ids | {"—"})


@pytest.mark.parametrize("pid", ["PORTFOLIO_001", "PORTFOLIO_002", "PORTFOLIO_003"])
def test_dry_run_mock_works_on_every_portfolio(loader, pid):
    ctx = _ctx(loader, pid)
    agent = ReasoningAgent(client=MockLLMClient.from_context(ctx))
    run = agent.generate(ctx)
    assert run.briefing.headline
    assert run.briefing.confidence is not None
