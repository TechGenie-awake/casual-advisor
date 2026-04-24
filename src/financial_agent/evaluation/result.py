"""Typed output of the evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DimensionScore:
    name: str
    rule_score: float           # 0.0 - 1.0
    judge_score: float | None   # 0.0 - 1.0, or None if judge wasn't run
    combined: float             # weighted blend (or just rule_score if no judge)
    weight: float               # contribution to the overall mean
    rule_critique: str
    judge_critique: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rule_score": round(self.rule_score, 2),
            "judge_score": (None if self.judge_score is None else round(self.judge_score, 2)),
            "combined": round(self.combined, 2),
            "weight": self.weight,
            "rule_critique": self.rule_critique,
            "judge_critique": self.judge_critique,
        }


@dataclass(frozen=True)
class EvaluationResult:
    portfolio_id: str
    overall_score: float
    dimensions: list[DimensionScore]
    summary: str = ""
    judge_used: bool = False
    judge_raw_response: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "portfolio_id": self.portfolio_id,
            "overall_score": round(self.overall_score, 2),
            "dimensions": [d.to_dict() for d in self.dimensions],
            "summary": self.summary,
            "judge_used": self.judge_used,
        }

    def to_markdown(self) -> str:
        lines = ["## Self-evaluation"]
        lines.append(
            f"**Overall: {self.overall_score:.2f}** "
            f"({'rule + judge' if self.judge_used else 'rule-based only'})"
        )
        if self.summary:
            lines.append(f"_{self.summary}_")
        lines.append("")
        lines.append("| Dimension | Rule | Judge | Combined | Critique |")
        lines.append("|---|---:|---:|---:|---|")
        for d in self.dimensions:
            judge = "—" if d.judge_score is None else f"{d.judge_score:.2f}"
            critique = d.judge_critique or d.rule_critique
            lines.append(
                f"| {d.name} | {d.rule_score:.2f} | {judge} | "
                f"**{d.combined:.2f}** | {critique} |"
            )
        return "\n".join(lines)
