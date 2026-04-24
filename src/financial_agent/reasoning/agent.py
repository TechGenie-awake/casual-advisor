"""ReasoningAgent — orchestrates context → prompt → LLM → parsed Briefing.

Also computes a *sanity-check confidence* from signal alignment, which the
LLM's self-reported confidence is cross-checked against. If the two diverge
sharply, we average them — preventing the LLM from over-claiming when its
own evidence is thin.
"""

from __future__ import annotations

from dataclasses import dataclass

from financial_agent.reasoning.briefing import Briefing
from financial_agent.reasoning.client import BaseLLMClient, LLMResponse
from financial_agent.reasoning.context import ReasoningContext
from financial_agent.reasoning.parser import parse_briefing
from financial_agent.reasoning.prompts import SYSTEM_PROMPT, render_user_prompt


@dataclass
class AgentRun:
    """Everything one briefing produced — useful for logging / Langfuse later."""

    briefing: Briefing
    response: LLMResponse
    rule_based_confidence: float


class ReasoningAgent:
    """Single-shot briefing generator.

    A future Phase 4 / chat agent will subclass or compose this to add
    multi-turn behavior and tool use.
    """

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    def generate(self, context: ReasoningContext) -> AgentRun:
        user = render_user_prompt(context)
        response = self._client.complete(system=SYSTEM_PROMPT, user=user)
        briefing = parse_briefing(response.text, portfolio_id=context.portfolio_id)
        rule_conf = self._rule_based_confidence(context, briefing)
        adjusted = self._reconcile_confidence(briefing.confidence, rule_conf)
        if adjusted != briefing.confidence:
            briefing = Briefing(
                portfolio_id=briefing.portfolio_id,
                headline=briefing.headline,
                causal_chain=briefing.causal_chain,
                conflicts=briefing.conflicts,
                recommendations=briefing.recommendations,
                confidence=adjusted,
                confidence_rationale=(
                    f"{briefing.confidence_rationale} "
                    f"(adjusted from self-reported {briefing.confidence:.2f} after "
                    f"rule-based check {rule_conf:.2f})"
                ),
                evidence_ids=briefing.evidence_ids,
                raw_response=briefing.raw_response,
            )
        return AgentRun(briefing=briefing, response=response, rule_based_confidence=rule_conf)

    # ------------------------------------------------------------------------

    @staticmethod
    def _rule_based_confidence(context: ReasoningContext, briefing: Briefing) -> float:
        """Heuristic 0-1 score based on evidence alignment.

        +0.20 if at least one causal link cites a HIGH-impact news ID
        +0.20 if every cited news ID actually appears in the context
        +0.20 if the chain has 2+ links
        +0.20 if every conflict in the context is addressed (or there are none)
        +0.20 if the chain references the portfolio's largest risk sector
        """
        score = 0.0
        evidence_set = {a.id for a in context.relevant_news}
        impact_high = {a.id for a in context.relevant_news if a.impact_level.value == "HIGH"}

        cited_in_chain = {eid for link in briefing.causal_chain for eid in link.evidence_ids}
        cited_total = set(briefing.evidence_ids) | cited_in_chain

        # 1. HIGH-impact citation present
        if cited_in_chain & impact_high:
            score += 0.20

        # 2. All citations are real
        if cited_total and cited_total.issubset(evidence_set):
            score += 0.20
        elif not cited_total:
            score += 0.0  # nothing cited — neutral
        else:
            score -= 0.10  # hallucinated IDs are a penalty

        # 3. Chain depth
        if len(briefing.causal_chain) >= 2:
            score += 0.20

        # 4. Conflicts addressed
        ctx_conflict_count = len(context.conflict_news) + len(
            context.stock_vs_sector_divergences
        )
        if ctx_conflict_count == 0 or briefing.conflicts:
            score += 0.20

        # 5. Largest-weight sector mentioned
        snap = context.portfolio_snapshot
        if snap.sector_allocation:
            top_sector = max(snap.sector_allocation.items(), key=lambda kv: kv[1])[0]
            chain_blob = " ".join(
                link.sector_impact + " " + link.stock_impact + " " + link.portfolio_impact
                for link in briefing.causal_chain
            )
            if top_sector in chain_blob:
                score += 0.20

        return max(0.0, min(1.0, score))

    @staticmethod
    def _reconcile_confidence(self_reported: float, rule_based: float) -> float:
        """If LLM is wildly more confident than the rule-based check, pull it down."""
        if self_reported - rule_based > 0.30:
            return round((self_reported + rule_based) / 2, 2)
        return round(self_reported, 2)
