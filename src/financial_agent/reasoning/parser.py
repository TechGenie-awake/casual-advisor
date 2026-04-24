"""Parse the LLM's XML response into a typed Briefing.

We use stdlib `xml.etree.ElementTree` rather than regex — it tolerates
attribute ordering and stray whitespace, and gives a clear error if the
LLM produces malformed output.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from financial_agent.reasoning.briefing import (
    Briefing,
    CausalLink,
    ConflictNote,
    Recommendation,
)


class BriefingParseError(ValueError):
    """Raised when the LLM output cannot be coerced into a Briefing."""


def _text(elem: ET.Element | None, default: str = "") -> str:
    if elem is None or elem.text is None:
        return default
    return elem.text.strip()


def _split_ids(raw: str) -> list[str]:
    if not raw:
        return []
    return [token.strip() for token in re.split(r"[,;\s]+", raw) if token.strip()]


def _extract_xml(raw: str) -> str:
    """Pull out the <briefing>...</briefing> block, ignoring surrounding text.

    Some models occasionally wrap output in markdown fences or prepend prose.
    We strip both.
    """
    # Strip markdown code fences if present.
    cleaned = re.sub(r"```(?:xml)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")

    match = re.search(r"<briefing>.*?</briefing>", cleaned, flags=re.DOTALL)
    if not match:
        raise BriefingParseError(
            "No <briefing>...</briefing> block found in LLM output."
        )
    return match.group(0)


def parse_briefing(raw: str, portfolio_id: str) -> Briefing:
    """Convert an LLM response into a Briefing. Raises on malformed input."""
    xml_text = _extract_xml(raw)
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BriefingParseError(f"Malformed XML: {exc}") from exc

    if root.tag != "briefing":
        raise BriefingParseError(f"Expected root <briefing>, got <{root.tag}>")

    headline = _text(root.find("headline"))
    if not headline:
        raise BriefingParseError("Missing <headline>")

    # --- Causal chain ---------------------------------------------------------
    chain: list[CausalLink] = []
    chain_root = root.find("causal_chain")
    if chain_root is not None:
        for link in chain_root.findall("link"):
            chain.append(
                CausalLink(
                    macro_event=_text(link.find("macro_event")),
                    sector_impact=_text(link.find("sector_impact")),
                    stock_impact=_text(link.find("stock_impact")),
                    portfolio_impact=_text(link.find("portfolio_impact")),
                    evidence_ids=_split_ids(_text(link.find("evidence"))),
                )
            )

    # --- Conflicts ------------------------------------------------------------
    conflicts: list[ConflictNote] = []
    conflicts_root = root.find("conflicts")
    if conflicts_root is not None:
        for conflict in conflicts_root.findall("conflict"):
            conflicts.append(
                ConflictNote(
                    description=_text(conflict.find("description")),
                    resolution=_text(conflict.find("resolution")),
                    evidence_ids=_split_ids(_text(conflict.find("evidence"))),
                )
            )

    # --- Recommendations ------------------------------------------------------
    recs: list[Recommendation] = []
    recs_root = root.find("recommendations")
    if recs_root is not None:
        for rec in recs_root.findall("recommendation"):
            priority = (rec.attrib.get("priority") or "MEDIUM").upper()
            if priority not in {"HIGH", "MEDIUM", "LOW"}:
                priority = "MEDIUM"
            recs.append(Recommendation(text=_text(rec), priority=priority))

    # --- Confidence + evidence -----------------------------------------------
    try:
        confidence = float(_text(root.find("confidence"), "0.0"))
    except ValueError:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_ids = _split_ids(_text(root.find("evidence")))
    confidence_rationale = _text(root.find("confidence_rationale"))

    return Briefing(
        portfolio_id=portfolio_id,
        headline=headline,
        causal_chain=chain,
        conflicts=conflicts,
        recommendations=recs,
        confidence=confidence,
        confidence_rationale=confidence_rationale,
        evidence_ids=evidence_ids,
        raw_response=raw,
    )
