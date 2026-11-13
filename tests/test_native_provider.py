"""Focused native-provider contract tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import serving.native_provider as native_provider
from serving.foundation_inference import CheckpointIncompatibleError
from serving.native_provider import NativeTakshProvider
from serving.providers import GenerationConfig, GenerationRequest, ProviderUnavailableError


def test_native_provider_missing_configuration_is_unavailable(monkeypatch):
    monkeypatch.delenv("TAKSH_NATIVE_CHECKPOINT", raising=False)
    monkeypatch.delenv("TAKSH_NATIVE_TOKENIZER", raising=False)
    provider = NativeTakshProvider()
    capabilities = provider.capabilities()
    assert capabilities.available is False
    assert capabilities.status == "checkpoint_missing"
    request = GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1))
    with pytest.raises(ProviderUnavailableError):
        provider.generate(request)


def test_native_provider_missing_checkpoint_fails_closed(tmp_path, monkeypatch):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("TAKSH_NATIVE_CHECKPOINT", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("TAKSH_NATIVE_TOKENIZER", str(tokenizer))
    provider = NativeTakshProvider()
    assert provider.capabilities().available is False
    assert provider.artifact_status()["status"] == "checkpoint_missing"
    request = GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1))
    with pytest.raises(ProviderUnavailableError):
        provider.generate(request)


def test_native_provider_never_falls_back_to_unknown_checkpoint(tmp_path, monkeypatch):
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    checkpoint = tmp_path / "not-a-checkpoint.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("TAKSH_NATIVE_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("TAKSH_NATIVE_TOKENIZER", str(tokenizer))
    provider = NativeTakshProvider()
    assert provider.capabilities().available is True
    assert provider.artifact_status()["status"] == "configured_unverified"

    probe = provider.probe()
    assert probe["status"] == "checkpoint_incompatible"
    assert provider.capabilities().available is False
    assert provider.capabilities().status == "checkpoint_incompatible"

    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))


def test_native_provider_reports_tokenizer_mismatch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    provider = NativeTakshProvider(checkpoint_path=checkpoint, tokenizer_path=tokenizer)

    def incompatible(*args, **kwargs):
        raise CheckpointIncompatibleError("tokenizer vocab 10 != checkpoint vocab 20")

    monkeypatch.setattr(native_provider, "load_foundation", incompatible)
    result = provider.probe()
    assert result["status"] == "checkpoint_incompatible"
    assert "tokenizer vocab" in result["detail"]
    assert provider.capabilities().available is False


def test_native_provider_enforces_context_before_generation(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    provider = NativeTakshProvider(checkpoint_path=checkpoint, tokenizer_path=tokenizer_path)

    class FakeTokenizer:
        def encode(self, prompt):
            return types.SimpleNamespace(ids=[1, 2, 3, 4, 5])

    class FakeModel:
        def num_params(self):
            return 10

    monkeypatch.setattr(
        native_provider,
        "load_foundation",
        lambda *a, **k: (FakeModel(), FakeTokenizer(), {"max_seq_len": 4}),
    )
    monkeypatch.setattr(
        native_provider,
        "generate_bounded",
        lambda *a, **k: pytest.fail("generation must not start beyond the context limit"),
    )
    with pytest.raises(ProviderUnavailableError, match="context limit"):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))


def test_native_provider_caps_generation_to_remaining_context(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")
    provider = NativeTakshProvider(checkpoint_path=checkpoint, tokenizer_path=tokenizer_path)

    class FakeTokenizer:
        def encode(self, prompt):
            return types.SimpleNamespace(ids=[1, 2, 3])

    class FakeModel:
        def num_params(self):
            return 10

    seen = {}

    def fake_generate(*args, **kwargs):
        seen.update(kwargs)
        return {
            "text": "ok",
            "raw_text": "ok",
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "completion_tokens": 1,
        }

    monkeypatch.setattr(
        native_provider,
        "load_foundation",
        lambda *a, **k: (FakeModel(), FakeTokenizer(), {"max_seq_len": 5}),
    )
    monkeypatch.setattr(native_provider, "generate_bounded", fake_generate)
    result = provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=100, seed=1)))
    assert result.text == "ok"
    assert seen["max_new_tokens"] <= 2
