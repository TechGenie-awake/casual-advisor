"""BriefingEvaluator — combines rule-based scorers and the optional LLM judge."""

from __future__ import annotations

from financial_agent.evaluation.judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeOutput,
    JudgeParseError,
    parse_judge_output,
    render_judge_prompt,
)
from financial_agent.evaluation.result import DimensionScore, EvaluationResult
from financial_agent.evaluation.rubric import DIMENSIONS, RULE_SCORERS, WEIGHTS
from financial_agent.observability import Tracer
from financial_agent.reasoning.briefing import Briefing
from financial_agent.reasoning.client import BaseLLMClient
from financial_agent.reasoning.context import ReasoningContext


# Per-dimension blend: rule weighted higher because it's auditable.
_RULE_WEIGHT = 0.6
_JUDGE_WEIGHT = 0.4


class BriefingEvaluator:
    """Score a briefing across the 5 reasoning-quality dimensions.

    The evaluator always runs rule-based scorers. If a `judge_client` is
    provided, it also runs the LLM-as-judge pass and blends the two.
    Errors in the judge call (parse failures, API hiccups) degrade
    silently to rule-only — eval should never break the user-facing flow.
    """

    def __init__(
        self,
        *,
        judge_client: BaseLLMClient | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._judge = judge_client
        self._tracer = tracer or Tracer()

    def score(
        self,
        briefing: Briefing,
        context: ReasoningContext,
        *,
        trace_span=None,
    ) -> EvaluationResult:
        rule_results = {
            name: scorer(briefing, context)
            for name, scorer in RULE_SCORERS.items()
        }

        judge_output: JudgeOutput | None = None
        if self._judge is not None:
            try:
                judge_output = self._run_judge(briefing, context)
            except (JudgeParseError, Exception):  # noqa: BLE001 — eval must not break
                judge_output = None

        dimensions: list[DimensionScore] = []
        for name in DIMENSIONS:
            rule_score, rule_critique = rule_results[name]
            judge_score = None
            judge_critique = ""
            if judge_output and name in judge_output.scores:
                js = judge_output.scores[name]
                judge_score = js.score
                judge_critique = js.critique

            if judge_score is None:
                combined = rule_score
            else:
                combined = (_RULE_WEIGHT * rule_score) + (_JUDGE_WEIGHT * judge_score)

            dimensions.append(
                DimensionScore(
                    name=name,
                    rule_score=rule_score,
                    judge_score=judge_score,
                    combined=combined,
                    weight=WEIGHTS[name],
                    rule_critique=rule_critique,
                    judge_critique=judge_critique,
                )
            )

        overall = sum(d.combined * d.weight for d in dimensions)
        result = EvaluationResult(
            portfolio_id=context.portfolio_id,
            overall_score=overall,
            dimensions=dimensions,
            summary=(judge_output.summary if judge_output else ""),
            judge_used=judge_output is not None,
            judge_raw_response=(judge_output.raw_response if judge_output else None),
        )

        # Push every dimension + the overall to Langfuse, if a span is open.
        for d in dimensions:
            self._tracer.log_score(
                trace_span,
                name=f"eval.{_slug(d.name)}",
                value=d.combined,
                comment=d.judge_critique or d.rule_critique,
            )
        self._tracer.log_score(
            trace_span, name="eval.overall", value=overall,
        )

        return result

    # --------------------------------------------------------------------

    def _run_judge(self, briefing: Briefing, context: ReasoningContext) -> JudgeOutput:
        user = render_judge_prompt(briefing, context)
        response = self._judge.complete(system=JUDGE_SYSTEM_PROMPT, user=user)
        return parse_judge_output(response.text)


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")
