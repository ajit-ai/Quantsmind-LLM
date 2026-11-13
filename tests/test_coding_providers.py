"""Focused tests for provider-agnostic coding providers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.coding_providers import OpenAICompatibleProvider, OllamaProvider, DeterministicCodingProvider, ProviderConfig, DeterministicCodingProvider
from serving.providers import GenerationConfig, GenerationRequest, ProviderUnavailableError
from serving.provider_diagnostics import diagnostic, diagnostics


def test_deterministic_provider_capabilities_and_generation():
    provider = DeterministicCodingProvider()
    capabilities = provider.capabilities()
    assert capabilities.available is True
    request = GenerationRequest("fix the bug", GenerationConfig(max_new_tokens=10, seed=1))
    result = provider.generate(request)
    assert result.provider == "deterministic-mock"
    assert "def broken" in result.text


def test_openai_provider_requires_credential():
    config = ProviderConfig("openai-compatible", "test-model", base_url="https://example.invalid/v1", api_key=None)
    provider = OpenAICompatibleProvider(config=config, transport=lambda *a, **k: {})
    request = GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1))
    with pytest.raises(ProviderUnavailableError):
        provider.generate(request)


def test_openai_provider_uses_bearer_token_without_logging():
    seen = {}

    def fake_transport(url, body, headers, timeout):
        seen["headers"] = dict(headers)
        seen["url"] = url
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    config = ProviderConfig("openai-compatible", "test-model", base_url="https://model.local/v1", api_key="secret-token-abc123")
    provider = OpenAICompatibleProvider(config=config, transport=fake_transport)
    result = provider.generate(GenerationRequest("say ok", GenerationConfig(max_new_tokens=1, seed=1)))
    assert result.text == "ok"
    assert seen["headers"]["Authorization"].endswith("secret-token-abc123")
    assert "secret-token-abc123" not in result.text


def test_ollama_provider_generates_with_mock_transport():
    def fake_transport(url, body, headers, timeout):
        return {"message": {"content": "local response"}, "done_reason": "stop"}

    config = ProviderConfig("ollama", "local-model", base_url="http://127.0.0.1:11434")
    provider = OllamaProvider(config=config, transport=fake_transport)
    result = provider.generate(GenerationRequest("say local", GenerationConfig(max_new_tokens=4, seed=1)))
    assert result.text == "local response"


def test_provider_diagnostics_report_unconfigured_and_secret_safe():
    report = diagnostic("openai-compatible")
    assert "api_key" not in report
    report = diagnostic("deterministic-mock")
    assert report["available"] is True
    report = diagnostic("not-a-provider")
    assert report["status"] == "unknown_provider"
