"""Langfuse-backed tracer. Single class, no global state, graceful no-op."""

from __future__ import annotations

import os
from typing import Any


class Tracer:
    """Wraps Langfuse v3 for portfolio briefings + their evaluation scores.

    The class auto-detects credentials from the environment; if either
    LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY is missing, every method
    becomes a no-op. This keeps the rest of the codebase one-pathed:
    callers don't have to wrap calls in `if tracer.enabled:` checks.
    """

    def __init__(self) -> None:
        public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
        host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()

        self._client: Any | None = None
        if not (public and secret):
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=public,
                secret_key=secret,
                host=host,
            )
        except Exception:
            # SDK unavailable or init failed — stay disabled.
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    # --- Trace lifecycle ----------------------------------------------------

    def start_briefing(
        self,
        portfolio_id: str,
        *,
        provider: str,
        model: str,
        extra: dict | None = None,
    ) -> Any | None:
        """Open a span for one full briefing run. Returns a handle or None."""
        if self._client is None:
            return None
        span = self._client.start_span(
            name=f"briefing.{portfolio_id}",
            input={"portfolio_id": portfolio_id, "provider": provider, "model": model},
            metadata=extra or {},
        )
        return span

    def end_briefing(self, span: Any | None, *, output: dict | None = None) -> None:
        if self._client is None or span is None:
            return
        if output is not None:
            span.update(output=output)
        span.end()
        try:
            self._client.flush()
        except Exception:
            pass

    # --- Per-LLM-call generation event --------------------------------------

    def log_generation(
        self,
        span: Any | None,
        *,
        name: str,
        model: str,
        system: str,
        user: str,
        output: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        if self._client is None or span is None:
            return
        try:
            gen = self._client.start_generation(
                name=name,
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            gen.update(
                output=output,
                usage_details={"input": input_tokens, "output": output_tokens},
            )
            gen.end()
        except Exception:
            # Telemetry must never break the user-facing flow.
            pass

    # --- Score attachment ---------------------------------------------------

    def log_score(
        self,
        span: Any | None,
        *,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        if self._client is None or span is None:
            return
        try:
            self._client.create_score(
                trace_id=span.trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception:
            pass
