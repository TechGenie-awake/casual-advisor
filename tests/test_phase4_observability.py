"""Phase 4a — Observability via Langfuse Tracer.

These tests do not hit Langfuse. They verify the wrapper:
  * starts disabled when LANGFUSE_* keys are missing
  * never raises in any method when disabled
  * starts enabled when keys are present (mocked Langfuse client)
"""

from unittest.mock import MagicMock, patch

import pytest

from financial_agent.observability import Tracer


def test_tracer_disabled_when_keys_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = Tracer()
    assert t.enabled is False


def test_tracer_methods_are_no_ops_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = Tracer()
    assert t.start_briefing("X", provider="GroqClient", model="m") is None
    # All methods must accept a None span without raising.
    t.log_generation(None, name="n", model="m", system="s", user="u",
                     output="o", input_tokens=1, output_tokens=1)
    t.log_score(None, name="x", value=0.5)
    t.end_briefing(None, output={"k": "v"})


def test_tracer_enabled_when_both_keys_present(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    fake_client = MagicMock()
    with patch("langfuse.Langfuse", return_value=fake_client):
        t = Tracer()
    assert t.enabled is True


def test_tracer_disabled_if_only_one_key_present(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    t = Tracer()
    assert t.enabled is False
