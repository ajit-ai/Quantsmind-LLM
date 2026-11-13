"""Focused tests for provider-agnostic coding providers."""

from __future__ import annotations

import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.coding_providers import (
    DeterministicCodingProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
    ProviderConfig,
    _urllib_transport,
)
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    redact_secrets,
)
from serving.provider_diagnostics import diagnostic, diagnostics

_TEST_KEY = "unit-test-only-not-a-secret"


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

    config = ProviderConfig("openai-compatible", "test-model", base_url="https://model.local/v1", api_key=_TEST_KEY)
    provider = OpenAICompatibleProvider(config=config, transport=fake_transport)
    result = provider.generate(GenerationRequest("say ok", GenerationConfig(max_new_tokens=1, seed=1)))
    assert result.text == "ok"
    assert seen["headers"]["Authorization"].endswith(_TEST_KEY)
    assert _TEST_KEY not in result.text


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
    assert report["test_only"] is True
    report = diagnostic("not-a-provider")
    assert report["status"] == "unknown_provider"


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        (401, ProviderAuthenticationError),
        (403, ProviderAuthenticationError),
        (429, ProviderRateLimitError),
        (500, ProviderServerError),
        (503, ProviderServerError),
        (404, ProviderResponseError),
    ],
)
def test_http_status_mapping(monkeypatch, code, error_type):
    def fail(*args, **kwargs):
        raise urllib.error.HTTPError(
            "https://example.invalid", code, "error", {}, io.BytesIO(b'{"error":"nope"}')
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(error_type):
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)


def test_timeout_and_connection_failures_are_structured(monkeypatch):
    def timeout(*args, **kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr("urllib.request.urlopen", timeout)
    with pytest.raises(ProviderTimeoutError):
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)

    def refused(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refused)
    with pytest.raises(ProviderUnavailableError):
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)


def test_malformed_and_empty_responses_are_rejected(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b""))
    with pytest.raises(ProviderResponseError):
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"not-json"))
    with pytest.raises(ProviderResponseError):
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)


def test_http_error_detail_is_redacted(monkeypatch):
    secret = "super-secret-value"

    def fail(*args, **kwargs):
        body = json.dumps({"error": f"bad key api_key={secret}"}).encode("utf-8")
        raise urllib.error.HTTPError(
            "https://example.invalid", 401, "unauthorized", {}, io.BytesIO(body)
        )

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(ProviderAuthenticationError) as caught:
        _urllib_transport("https://example.invalid", b"{}", {}, 1.0)
    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_openai_rejects_empty_choices_and_content():
    config = ProviderConfig(
        "openai-compatible", "test-model", base_url="https://model.local/v1", api_key=_TEST_KEY
    )
    request = GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1))

    empty_choices = OpenAICompatibleProvider(
        config=config, transport=lambda *a, **k: {"choices": []}
    )
    with pytest.raises(ProviderResponseError):
        empty_choices.generate(request)

    empty_content = OpenAICompatibleProvider(
        config=config,
        transport=lambda *a, **k: {"choices": [{"message": {"content": ""}}]},
    )
    with pytest.raises(ProviderResponseError):
        empty_content.generate(request)


def test_ollama_missing_model_is_actionable():
    config = ProviderConfig("ollama", "missing-model", base_url="http://127.0.0.1:11434")
    provider = OllamaProvider(
        config=config,
        transport=lambda *a, **k: {"error": "model 'missing-model' not found"},
    )
    with pytest.raises(ProviderUnavailableError, match="not available locally"):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))


def test_ollama_rejects_malformed_and_empty_content():
    config = ProviderConfig("ollama", "local-model", base_url="http://127.0.0.1:11434")
    provider = OllamaProvider(config=config, transport=lambda *a, **k: {"message": {}})
    with pytest.raises(ProviderResponseError):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))


def test_ollama_probe_is_explicit_and_reports_model_presence():
    config = ProviderConfig("ollama", "local-model", base_url="http://127.0.0.1:11434")
    seen = {}

    def fake_probe(url, timeout):
        seen["url"] = url
        return {"models": [{"name": "local-model"}, {"name": "other-model"}]}

    provider = OllamaProvider(config=config, probe_transport=fake_probe)
    result = provider.probe()
    assert result["model_present"] is True
    assert result["models"] == ["local-model", "other-model"]
    assert seen["url"].endswith("/api/tags")


def test_redact_secrets_removes_bearer_and_named_values():
    text = "Authorization: Bearer abc123 api_key=def456 other=ok"
    redacted = redact_secrets(text, ["abc123", "def456"])
    assert "abc123" not in redacted
    assert "def456" not in redacted
    assert "[REDACTED]" in redacted
