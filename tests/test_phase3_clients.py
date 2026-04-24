"""Phase 3 — LLM client wrappers.

These tests do NOT hit the real APIs. Instead they verify:
  * GroqClient calls `chat.completions.create` with the right shape
  * AnthropicClient calls `messages.create` with the right shape
  * Both return properly-populated LLMResponse objects
  * Both raise a clear error when the API key is missing

The Anthropic / Groq SDK objects are stubbed via dependency injection
into the wrapper's `_client` attribute after construction.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from financial_agent.reasoning.client import (
    AnthropicClient,
    GroqClient,
    LLMResponse,
    MockLLMClient,
)


# --------------------------------------------------------------------------
# GroqClient
# --------------------------------------------------------------------------

def _fake_groq_completion(text: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    """Build a fake Groq chat.completions.create response object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def test_groq_client_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        GroqClient()


def test_groq_client_uses_explicit_api_key():
    """Passing api_key= bypasses the env var."""
    client = GroqClient(api_key="gsk_test_dummy")
    assert client._model == GroqClient.DEFAULT_GROQ_MODEL


def test_groq_client_complete_returns_llm_response():
    client = GroqClient(api_key="gsk_test_dummy")
    fake = _fake_groq_completion("<briefing>ok</briefing>", 1234, 56)
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = fake

    response = client.complete(system="sys", user="usr")

    assert isinstance(response, LLMResponse)
    assert response.text == "<briefing>ok</briefing>"
    assert response.input_tokens == 1234
    assert response.output_tokens == 56
    assert response.total_tokens == 1290
    assert response.model == GroqClient.DEFAULT_GROQ_MODEL


def test_groq_client_sends_system_and_user_messages():
    client = GroqClient(api_key="gsk_test_dummy")
    client._client = MagicMock()
    client._client.chat.completions.create.return_value = _fake_groq_completion("x")

    client.complete(system="SYSTEM_TEXT", user="USER_TEXT")

    call_kwargs = client._client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == GroqClient.DEFAULT_GROQ_MODEL
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "SYSTEM_TEXT"},
        {"role": "user", "content": "USER_TEXT"},
    ]


# --------------------------------------------------------------------------
# AnthropicClient
# --------------------------------------------------------------------------

def _fake_anthropic_message(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Build a fake Anthropic messages.create response object."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_anthropic_client_raises_on_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_anthropic_client_complete_returns_llm_response():
    client = AnthropicClient(api_key="sk-ant-test")
    fake = _fake_anthropic_message("<briefing>ok</briefing>", 2000, 300)
    client._client = MagicMock()
    client._client.messages.create.return_value = fake

    response = client.complete(system="sys", user="usr")

    assert response.text == "<briefing>ok</briefing>"
    assert response.input_tokens == 2000
    assert response.output_tokens == 300


# --------------------------------------------------------------------------
# MockLLMClient
# --------------------------------------------------------------------------

def test_mock_client_returns_canned_text():
    client = MockLLMClient("CANNED")
    response = client.complete(system="s", user="u")
    assert response.text == "CANNED"
    assert response.model == "mock"
