"""Typed output models for Phase 3.

Every field below maps directly to a section of the structured XML the LLM
is required to produce. Keeping these as frozen dataclasses (not pydantic)
because they're constructed by the parser, not validated from JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CausalLink:
    """One link in the macro → sector → stock → portfolio chain."""

    macro_event: str       # e.g., "RBI hawkish stance (NEWS001)"
    sector_impact: str     # e.g., "BANKING -3.11% weighted"
    stock_impact: str      # e.g., "HDFCBANK -3.51%, ICICIBANK -3.13%"
    portfolio_impact: str  # e.g., "Portfolio's 72% banking exposure → -2.0pp of -2.73% day move"
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "macro_event": self.macro_event,
            "sector_impact": self.sector_impact,
            "stock_impact": self.stock_impact,
            "portfolio_impact": self.portfolio_impact,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class ConflictNote:
    """An explicit edge case the agent must reconcile, not paper over."""

    description: str       # what the conflict is
    resolution: str        # how the agent reconciles it
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "resolution": self.resolution,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class Recommendation:
    """A single action item, with priority so the UI can rank them."""

    text: str
    priority: str = "MEDIUM"   # HIGH | MEDIUM | LOW

    def to_dict(self) -> dict:
        return {"text": self.text, "priority": self.priority}


@dataclass(frozen=True)
class Briefing:
    """Final output of the reasoning layer."""

    portfolio_id: str
    headline: str
    causal_chain: list[CausalLink]
    conflicts: list[ConflictNote]
    recommendations: list[Recommendation]
    confidence: float                      # 0.0 - 1.0
    confidence_rationale: str
    evidence_ids: list[str] = field(default_factory=list)
    raw_response: str | None = None        # full LLM text for debugging / Langfuse

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "headline": self.headline,
            "causal_chain": [c.to_dict() for c in self.causal_chain],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "recommendations": [r.to_dict() for r in self.recommendations],
            "confidence": round(self.confidence, 2),
            "confidence_rationale": self.confidence_rationale,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_markdown(self) -> str:
        """Human-readable rendering used by the CLI and UI."""
        lines: list[str] = []
        lines.append(f"# Briefing — {self.portfolio_id}")
        lines.append("")
        lines.append(f"**{self.headline}**")
        lines.append("")
        lines.append(f"_Confidence: {self.confidence:.2f} — {self.confidence_rationale}_")
        lines.append("")

        lines.append("## Causal chain")
        if not self.causal_chain:
            lines.append("_(no causal links produced)_")
        for i, link in enumerate(self.causal_chain, 1):
            lines.append(f"{i}. **{link.macro_event}**")
            lines.append(f"   - Sector: {link.sector_impact}")
            lines.append(f"   - Stocks: {link.stock_impact}")
            lines.append(f"   - Portfolio: {link.portfolio_impact}")
            if link.evidence_ids:
                lines.append(f"   - Evidence: {', '.join(link.evidence_ids)}")
        lines.append("")

        if self.conflicts:
            lines.append("## Conflicting signals")
            for c in self.conflicts:
                lines.append(f"- **{c.description}**")
                lines.append(f"  Resolution: {c.resolution}")
                if c.evidence_ids:
                    lines.append(f"  Evidence: {', '.join(c.evidence_ids)}")
            lines.append("")

        if self.recommendations:
            lines.append("## Recommendations")
            for r in self.recommendations:
                lines.append(f"- [{r.priority}] {r.text}")
            lines.append("")

        return "\n".join(lines)
