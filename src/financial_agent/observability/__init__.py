"""Phase 4 — Observability via Langfuse.

A thin `Tracer` that wraps Langfuse v3. Gracefully no-ops when LANGFUSE_*
environment variables aren't set, so dev / CI / users without an account
can still run the full pipeline.
"""

from financial_agent.observability.tracer import Tracer

__all__ = ["Tracer"]
