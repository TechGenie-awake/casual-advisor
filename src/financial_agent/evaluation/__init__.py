"""Phase 4 — Self-evaluation of generated briefings.

Scores each briefing across five dimensions on a 0-1 scale via two methods:
  * Rule-based: deterministic regex / set checks (free, always runs)
  * LLM-as-judge: a separate Groq/Anthropic call that critiques the briefing
                  against the same rubric (richer, optional)

The combined score is `0.6 * rule + 0.4 * judge` per dimension; the overall
score is a weighted mean across dimensions (weights in `rubric.WEIGHTS`).
"""

from financial_agent.evaluation.evaluator import BriefingEvaluator
from financial_agent.evaluation.result import (
    DimensionScore,
    EvaluationResult,
)
from financial_agent.evaluation.rubric import DIMENSIONS, WEIGHTS

__all__ = [
    "DIMENSIONS",
    "DimensionScore",
    "BriefingEvaluator",
    "EvaluationResult",
    "WEIGHTS",
]
