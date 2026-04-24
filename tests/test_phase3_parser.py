"""Phase 3.2 — Output parser: convert LLM XML into a typed Briefing."""

import pytest

from financial_agent.reasoning.parser import BriefingParseError, parse_briefing


GOLDEN_XML = """<briefing>
<headline>Banking-heavy portfolio fell 2.73% as RBI's hawkish stance crushed the sector.</headline>
<causal_chain>
  <link>
    <macro_event>RBI held rates and signaled hawkish stance (NEWS001).</macro_event>
    <sector_impact>BANKING -3.11% weighted, worst sector of the day.</sector_impact>
    <stock_impact>HDFCBANK -3.51% (22.6% weight), ICICIBANK -3.13% (13.8%), SBIN -3.02% (14.7%).</stock_impact>
    <portfolio_impact>Banking exposure of 72% drove ~₹40k of the ₹57k day loss.</portfolio_impact>
    <evidence>NEWS001,NEWS007</evidence>
  </link>
  <link>
    <macro_event>FII outflows of ₹4,500cr added pressure (NEWS007).</macro_event>
    <sector_impact>FINANCIAL_SERVICES -1.99%.</sector_impact>
    <stock_impact>BAJFINANCE -2.05% despite positive guidance.</stock_impact>
    <portfolio_impact>Added ₹5.6k to losses on the 12% FS allocation.</portfolio_impact>
    <evidence>NEWS007,NEWS011</evidence>
  </link>
</causal_chain>
<conflicts>
  <conflict>
    <description>Bajaj Finance reported strong asset quality but fell 2%.</description>
    <resolution>Sector-wide rate-sensitivity overrode company-specific positives.</resolution>
    <evidence>NEWS011</evidence>
  </conflict>
</conflicts>
<recommendations>
  <recommendation priority="HIGH">Reduce single-sector exposure below 50%.</recommendation>
  <recommendation priority="MEDIUM">Trim HDFCBANK from 22.6% to under 15%.</recommendation>
</recommendations>
<evidence>NEWS001,NEWS007,NEWS011</evidence>
<confidence>0.85</confidence>
<confidence_rationale>Strong alignment between macro news, sector move, and portfolio holdings.</confidence_rationale>
</briefing>"""


def test_parse_golden_briefing():
    b = parse_briefing(GOLDEN_XML, portfolio_id="PORTFOLIO_002")
    assert b.portfolio_id == "PORTFOLIO_002"
    assert "RBI" in b.headline or "Banking" in b.headline
    assert len(b.causal_chain) == 2
    assert b.causal_chain[0].evidence_ids == ["NEWS001", "NEWS007"]
    assert len(b.conflicts) == 1
    assert "Bajaj" in b.conflicts[0].description
    assert len(b.recommendations) == 2
    assert b.recommendations[0].priority == "HIGH"
    assert b.confidence == 0.85
    assert "NEWS001" in b.evidence_ids


def test_parser_strips_markdown_fences():
    wrapped = "```xml\n" + GOLDEN_XML + "\n```"
    b = parse_briefing(wrapped, portfolio_id="PORTFOLIO_002")
    assert len(b.causal_chain) == 2


def test_parser_handles_preamble_text():
    """Some models prefix prose. Parser must still find the briefing block."""
    noisy = "Sure, here's the briefing:\n\n" + GOLDEN_XML + "\n\nLet me know if you need more."
    b = parse_briefing(noisy, portfolio_id="PORTFOLIO_002")
    assert b.confidence == 0.85


def test_parser_clamps_confidence_to_unit_range():
    xml = GOLDEN_XML.replace("<confidence>0.85</confidence>", "<confidence>2.5</confidence>")
    b = parse_briefing(xml, portfolio_id="PORTFOLIO_002")
    assert b.confidence == 1.0

    xml = GOLDEN_XML.replace("<confidence>0.85</confidence>", "<confidence>-0.4</confidence>")
    b = parse_briefing(xml, portfolio_id="PORTFOLIO_002")
    assert b.confidence == 0.0


def test_parser_defaults_invalid_priority_to_medium():
    xml = GOLDEN_XML.replace('priority="HIGH"', 'priority="EXTREME"')
    b = parse_briefing(xml, portfolio_id="PORTFOLIO_002")
    assert b.recommendations[0].priority == "MEDIUM"


def test_parser_raises_on_missing_briefing_root():
    with pytest.raises(BriefingParseError):
        parse_briefing("Just a sentence with no XML.", portfolio_id="X")


def test_parser_raises_on_missing_headline():
    no_headline = GOLDEN_XML.replace(
        "<headline>Banking-heavy portfolio fell 2.73% as RBI's hawkish stance crushed the sector.</headline>",
        "",
    )
    with pytest.raises(BriefingParseError):
        parse_briefing(no_headline, portfolio_id="X")


def test_parser_raises_on_malformed_xml():
    with pytest.raises(BriefingParseError):
        parse_briefing("<briefing><headline>oops</headline", portfolio_id="X")


def test_briefing_to_markdown_contains_key_sections():
    b = parse_briefing(GOLDEN_XML, portfolio_id="PORTFOLIO_002")
    md = b.to_markdown()
    assert "Causal chain" in md
    assert "Conflicting signals" in md
    assert "Recommendations" in md
    assert "NEWS001" in md
