"""Focused native-provider contract tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.native_provider import NativeTakshProvider
from serving.providers import GenerationConfig, GenerationRequest, ProviderUnavailableError


def test_native_provider_missing_configuration_is_unavailable(monkeypatch):
    monkeypatch.delenv("TAKSH_NATIVE_CHECKPOINT", raising=False)
    monkeypatch.delenv("TAKSH_NATIVE_TOKENIZER", raising=False)
    provider = NativeTakshProvider()
    capabilities = provider.capabilities()
    assert capabilities.available is False
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
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest("hello", GenerationConfig(max_new_tokens=1, seed=1)))
