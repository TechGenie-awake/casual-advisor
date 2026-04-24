"""Rule-based scorers — one per dimension.

Each scorer takes (briefing, context) and returns (score, critique).
Scores are clamped to [0, 1].
"""

from __future__ import annotations

import re

from financial_agent.reasoning.briefing import Briefing
from financial_agent.reasoning.context import ReasoningContext


# Dimension names + per-dimension weight in the overall score.
# The five names mirror the brief's "Reasoning Quality" rubric line.
DIMENSIONS = [
    "Causal Depth",
    "Evidence Accuracy",
    "Conflict Handling",
    "Prioritization",
    "Quantification",
]

WEIGHTS: dict[str, float] = {
    "Causal Depth": 0.30,
    "Evidence Accuracy": 0.25,
    "Conflict Handling": 0.15,
    "Prioritization": 0.15,
    "Quantification": 0.15,
}

# Regex for "₹12,345" or "Rs 12345" or "12,345 INR" — currency-anchored numbers.
_RUPEE_RE = re.compile(r"(₹|\bRs\.?\b|INR)\s*-?[\d,]+", re.IGNORECASE)
# Regex for percentages (allow signed): "+1.81%", "-3.51 %".
_PERCENT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%")


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


# ---------- Dimension 1: Causal Depth ---------------------------------------

def score_causal_depth(briefing: Briefing, context: ReasoningContext) -> tuple[float, str]:
    """Each link should populate all 5 fields (macro/sector/stock/portfolio/evidence)."""
    if not briefing.causal_chain:
        return 0.0, "No causal links produced."

    per_link_scores: list[float] = []
    for link in briefing.causal_chain:
        fields = [
            link.macro_event.strip(),
            link.sector_impact.strip(),
            link.stock_impact.strip(),
            link.portfolio_impact.strip(),
            ",".join(link.evidence_ids),
        ]
        non_empty = sum(1 for f in fields if f)
        per_link_scores.append(non_empty / 5.0)

    avg = sum(per_link_scores) / len(per_link_scores)
    chain_count = len(briefing.causal_chain)
    # Bonus for having 2+ links; penalty if only 1.
    if chain_count >= 2:
        avg = min(1.0, avg * 1.05)
    else:
        avg *= 0.7

    critique = (
        f"{chain_count} link(s); avg field completeness "
        f"{int(sum(per_link_scores) / len(per_link_scores) * 100)}%."
    )
    return _clamp(avg), critique


# ---------- Dimension 2: Evidence Accuracy ----------------------------------

def score_evidence_accuracy(
    briefing: Briefing, context: ReasoningContext
) -> tuple[float, str]:
    """All cited news IDs must appear in the context's relevant_news.

    Hallucinated IDs are a hard failure. No citations is neutral (0.6).
    """
    real = set(context.evidence_ids())
    cited: set[str] = set(briefing.evidence_ids)
    for link in briefing.causal_chain:
        cited.update(link.evidence_ids)
    for c in briefing.conflicts:
        cited.update(c.evidence_ids)

    # Strip placeholder dashes the mock client uses when no news is cited.
    cited = {c for c in cited if c and c != "—"}

    if not cited:
        return 0.6, "No news IDs cited."

    valid = cited & real
    invalid = cited - real
    if invalid:
        score = max(0.0, len(valid) / len(cited) - 0.3)  # penalty for hallucination
        return _clamp(score), f"Hallucinated IDs: {sorted(invalid)}"
    return 1.0, f"{len(valid)} citation(s), all in context."


# ---------- Dimension 3: Conflict Handling ----------------------------------

def score_conflict_handling(
    briefing: Briefing, context: ReasoningContext
) -> tuple[float, str]:
    """Every conflict in the context should be addressed in the briefing."""
    ctx_conflict_count = (
        len(context.conflict_news) + len(context.stock_vs_sector_divergences)
    )
    if ctx_conflict_count == 0:
        return 1.0, "No conflicts in context to address."

    if not briefing.conflicts:
        return 0.0, f"{ctx_conflict_count} conflict(s) in context, none addressed."

    coverage = min(1.0, len(briefing.conflicts) / max(1, ctx_conflict_count))
    critique = (
        f"Addressed {len(briefing.conflicts)} of {ctx_conflict_count} "
        f"context conflicts."
    )
    return _clamp(coverage), critique


# ---------- Dimension 4: Prioritization -------------------------------------

def score_prioritization(
    briefing: Briefing, context: ReasoningContext
) -> tuple[float, str]:
    """The largest ₹ contributor should appear by name in the chain."""
    attr = context.contribution_attribution
    if attr is None or not attr.top_contributors:
        return 0.5, "No attribution data available."

    top = attr.top_contributors[0]
    chain_text = " ".join(
        link.stock_impact + " " + link.sector_impact
        for link in briefing.causal_chain
    )

    if top.identifier in chain_text:
        return 1.0, f"Top contributor {top.identifier} appears in the chain."
    if top.sector in chain_text:
        return 0.6, (
            f"Top contributor's sector {top.sector} mentioned but "
            f"{top.identifier} not named."
        )
    return 0.0, f"Top contributor {top.identifier} ({top.sector}) is missing."


# ---------- Dimension 5: Quantification -------------------------------------

def score_quantification(
    briefing: Briefing, context: ReasoningContext
) -> tuple[float, str]:
    """Each link should include both a ₹ figure and a % figure."""
    if not briefing.causal_chain:
        return 0.0, "No causal links."

    qualifying = 0
    for link in briefing.causal_chain:
        blob = link.stock_impact + " " + link.portfolio_impact
        has_rupee = bool(_RUPEE_RE.search(blob))
        has_pct = bool(_PERCENT_RE.search(blob))
        if has_rupee and has_pct:
            qualifying += 1

    score = qualifying / len(briefing.causal_chain)
    critique = (
        f"{qualifying} of {len(briefing.causal_chain)} links include "
        f"both ₹ and % values."
    )
    return _clamp(score), critique


# ---------- Registry --------------------------------------------------------

# Single source of truth: dimension name → (scorer, weight).
RULE_SCORERS = {
    "Causal Depth": score_causal_depth,
    "Evidence Accuracy": score_evidence_accuracy,
    "Conflict Handling": score_conflict_handling,
    "Prioritization": score_prioritization,
    "Quantification": score_quantification,
}
