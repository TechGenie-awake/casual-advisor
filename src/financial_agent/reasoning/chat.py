"""Briefing-grounded follow-up chat.

The chat agent answers questions *about the briefing the user just saw*,
using the same `ReasoningContext` and citing the same news IDs as the
briefing. This is "Flavor A" in the design — narrow scope, high signal.

The chat does NOT invoke tools and does NOT browse the web. It can only
reason over the typed context object that produced the briefing. This is
intentional: it keeps grounding tight and means every answer is auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

from financial_agent.observability import Tracer
from financial_agent.reasoning.briefing import Briefing
from financial_agent.reasoning.client import BaseLLMClient, LLMResponse
from financial_agent.reasoning.context import ReasoningContext


Role = Literal["user", "assistant"]


CHAT_SYSTEM_PROMPT = """You are an autonomous financial advisor agent answering follow-up questions
about a portfolio briefing the user just received.

You have access to:
- The briefing's headline, causal chain, conflicts, and recommendations
- The full reasoning context (market intelligence, portfolio snapshot, contribution
  attribution, sector trends, news articles, macro correlations)

# Hard rules
1. **Cite only what is in the context.** Every news ID, ₹ figure, % weight, or sector
   move you mention must appear in the briefing or context. Never invent data.
2. **Stay scoped to this portfolio.** If asked about a stock or sector the user
   doesn't hold, briefly note that fact and connect back to their actual exposure.
3. **Be quantitative.** Use the exact figures from `contribution_attribution`,
   `portfolio_snapshot`, and `relevant_sector_trends`. No "significant" or
   "notable" without a number.
4. **One paragraph max** unless the question genuinely requires more. Brevity beats
   thoroughness for chat.
5. **If the answer isn't in the context, say so.** Do not speculate.

Your reply is plain Markdown — no XML, no preamble.
"""


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class ChatSession:
    """In-memory chat history grounded on a single briefing + context."""

    briefing: Briefing
    context: ReasoningContext
    messages: list[ChatMessage] = field(default_factory=list)

    def append(self, role: Role, content: str) -> None:
        self.messages.append(ChatMessage(role=role, content=content))


def _render_user_payload(session: ChatSession, new_question: str) -> str:
    """Build the user-message payload — briefing + context + history + question."""
    briefing_md = session.briefing.to_markdown()
    context_json = json.dumps(session.context.to_dict(), indent=2, ensure_ascii=False)

    history_blocks: list[str] = []
    for m in session.messages:
        prefix = "USER" if m.role == "user" else "ASSISTANT"
        history_blocks.append(f"{prefix}: {m.content}")
    history_text = "\n\n".join(history_blocks) if history_blocks else "(no prior turns)"

    return f"""# Briefing the user just saw
{briefing_md}

# Full reasoning context (JSON — facts you may cite)
```json
{context_json}
```

# Conversation so far
{history_text}

# New user question
{new_question}

Reply now in plain Markdown. Cite news IDs and ₹ figures from the context above."""


@dataclass
class ChatTurn:
    """Result of one user-question → assistant-answer round."""

    question: str
    answer: str
    response: LLMResponse


class ChatAgent:
    """Single-shot follow-up chat agent grounded on a briefing.

    Usage:
        session = ChatSession(briefing=run.briefing, context=ctx)
        agent   = ChatAgent(client=client)
        turn    = agent.ask(session, "Why did Bajaj Finance fall despite positive guidance?")
        print(turn.answer)
    """

    def __init__(self, client: BaseLLMClient, *, tracer: Tracer | None = None) -> None:
        self._client = client
        self._tracer = tracer or Tracer()

    def ask(self, session: ChatSession, question: str, *, trace_span=None) -> ChatTurn:
        user_payload = _render_user_payload(session, question)
        response = self._client.complete(system=CHAT_SYSTEM_PROMPT, user=user_payload)
        answer = response.text.strip()

        # Persist this turn into the session BEFORE returning so callers don't have to.
        session.append("user", question)
        session.append("assistant", answer)

        # Telemetry — every chat turn is its own generation event under the briefing's trace.
        self._tracer.log_generation(
            trace_span,
            name="chat.turn",
            model=response.model,
            system=CHAT_SYSTEM_PROMPT,
            user=user_payload,
            output=answer,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )

        return ChatTurn(question=question, answer=answer, response=response)
