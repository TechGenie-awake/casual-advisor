"""Phase 3 — Autonomous Reasoning (the agent layer).

Glues Phase 1 (market intelligence) and Phase 2 (portfolio analytics) into
a causal narrative via an LLM. Output is a typed `Briefing` with:

  * a causal chain (macro news → sector → stock → portfolio)
  * explicit conflict notes (positive news + negative price action, etc.)
  * prioritized recommendations
  * a confidence score with rationale
"""

from financial_agent.reasoning.agent import ReasoningAgent
from financial_agent.reasoning.briefing import (
    Briefing,
    CausalLink,
    ConflictNote,
    Recommendation,
)
from financial_agent.reasoning.chat import ChatAgent, ChatMessage, ChatSession, ChatTurn
from financial_agent.reasoning.client import (
    AnthropicClient,
    BaseLLMClient,
    GroqClient,
    MockLLMClient,
)
from financial_agent.reasoning.context import ReasoningContext, build_context

__all__ = [
    "AnthropicClient",
    "BaseLLMClient",
    "Briefing",
    "CausalLink",
    "ChatAgent",
    "ChatMessage",
    "ChatSession",
    "ChatTurn",
    "ConflictNote",
    "GroqClient",
    "MockLLMClient",
    "ReasoningAgent",
    "ReasoningContext",
    "Recommendation",
    "build_context",
]
