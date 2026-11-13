"""Canonical provider-contract conformance tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.coding_providers import OllamaProvider, OpenAICompatibleProvider, ProviderConfig
from serving.native_provider import NativeTakshProvider
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    GenerationResult,
    ProviderError,
    get_provider,
)

_TEST_KEY = "unit-test-only-not-a-secret"


def _openai():
    config = ProviderConfig(
        "openai-compatible", "contract-model", base_url="https://model.local/v1", api_key=_TEST_KEY
    )
    return OpenAICompatibleProvider(
        config=config,
        transport=lambda *a, **k: {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


def _ollama():
    config = ProviderConfig("ollama", "contract-model", base_url="http://127.0.0.1:11434")
    return OllamaProvider(
        config=config,
        transport=lambda *a, **k: {
            "message": {"content": "ok"},
            "done_reason": "stop",
            "prompt_eval_count": 1,
            "eval_count": 1,
        },
    )


def _providers(tmp_path):
    missing = tmp_path / "missing.pt"
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    return [
        get_provider("deterministic-mock"),
        _openai(),
        _ollama(),
        NativeTakshProvider(checkpoint_path=missing, tokenizer_path=tokenizer),
    ]


def test_all_providers_share_identity_capability_and_result_semantics(tmp_path):
    for provider in _providers(tmp_path):
        identity = provider.identity()
        identity.validated()
        capabilities = provider.capabilities()
        assert capabilities.provider == identity.provider
        assert isinstance(capabilities.configured, bool)
        assert capabilities.status

        request = GenerationRequest("write a function", GenerationConfig(max_new_tokens=2, seed=1))
        if capabilities.available:
            result = provider.generate(request)
            assert isinstance(result, GenerationResult)
            assert result.provider == identity.provider
            assert result.text
            assert result.prompt_tokens >= 0
            assert result.completion_tokens >= 0
            assert result.latency_seconds >= 0
        else:
            with pytest.raises(ProviderError):
                provider.generate(request)


def test_common_request_validation_is_enforced(tmp_path):
    provider = get_provider("deterministic-mock")
    with pytest.raises(ValueError):
        provider.generate(GenerationRequest(""))
    with pytest.raises(ValueError):
        provider.generate(GenerationRequest("x", GenerationConfig(max_new_tokens=0)))
    with pytest.raises(ValueError):
        provider.generate(GenerationRequest("x", GenerationConfig(top_p=0)))
    with pytest.raises(ValueError):
        provider.generate(GenerationRequest("x", GenerationConfig(timeout_seconds=0)))


def test_test_only_providers_are_explicitly_labelled(tmp_path):
    for provider in _providers(tmp_path):
        caps = provider.capabilities()
        expected = provider.name in ("deterministic-mock",)
        assert caps.test_only is expected
        if expected:
            assert caps.status == "test_only"


def test_construction_and_capabilities_do_not_contact_networks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("provider construction/capabilities must not perform network I/O")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    for provider in (_openai(), _ollama()):
        provider.identity()
        provider.capabilities()


def test_native_missing_checkpoint_is_fail_closed(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    provider = NativeTakshProvider(
        checkpoint_path=tmp_path / "missing.pt", tokenizer_path=tokenizer
    )
    assert provider.capabilities().available is False
    assert provider.artifact_status()["status"] == "checkpoint_missing"
    with pytest.raises(ProviderError):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))
