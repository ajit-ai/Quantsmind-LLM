"""Focused tests for the D2 provider contract."""

from __future__ import annotations

import pytest

from serving import providers
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    ProviderNotFoundError,
    ProviderUnavailableError,
    TakshS2Provider,
    generate_with_timeout,
    get_provider,
    list_providers,
    qualification_metadata,
)


class FakeModel:
    cfg = {"max_seq_len": 512}


class FakeTokenizer:
    pass


def test_provider_registry_lists_s2_and_foundation_placeholders():
    assert "taksh-s2" in list_providers()
    assert "seed-coder-8b-instruct" in list_providers()
    assert "qwen2.5-coder-7b-instruct" in list_providers()
    assert "qwen3-coder-30b-a3b-instruct" in list_providers()


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderNotFoundError):
        get_provider("unknown-provider")


def test_missing_s2_revision_is_rejected():
    with pytest.raises(ValueError, match="revision must be set"):
        TakshS2Provider(FakeModel(), FakeTokenizer(), model_id="s2", revision="")


def test_missing_s2_model_id_is_rejected():
    with pytest.raises(ValueError, match="model_id must be set"):
        TakshS2Provider(FakeModel(), FakeTokenizer(), model_id="", revision="checkpoint-sha")


def test_external_provider_is_unavailable_and_offline():
    provider = get_provider("seed-coder-8b-instruct")
    capabilities = provider.capabilities()
    assert capabilities.available is False
    assert capabilities.offline is True
    assert provider.identity().license_or_rights.startswith("MIT")
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest(prompt="write a function"))


def test_s2_provider_preserves_raw_output(monkeypatch):
    from serving import local_inference

    def fake_generate_ids(model, tokenizer, prompt, **kwargs):
        assert kwargs["binding_repair"] is False
        return {
            "text": "    return first + 1\n",
            "raw_text": "    return value + 1\n",
            "finish_reason": "stop",
            "prompt_tokens": 8,
            "completion_tokens": 6,
            "repair": {
                "enabled": False,
                "applied": False,
                "status": "disabled",
                "reason": "repair disabled",
                "mapping": {},
                "changed": False,
            },
        }

    monkeypatch.setattr(local_inference, "generate_ids", fake_generate_ids)
    provider = TakshS2Provider(
        FakeModel(), FakeTokenizer(),
        model_id="taksh-s2",
        revision="checkpoint-sha",
    )
    result = provider.generate(
        GenerationRequest(
            prompt="def add(first):",
            config=GenerationConfig(binding_repair=False),
        )
    )
    assert result.text == "    return first + 1\n"
    assert result.raw_text == "    return value + 1\n"
    assert result.provider == "taksh-s2"
    assert result.revision == "checkpoint-sha"
    assert result.repair["status"] == "disabled"


def test_provider_timeout_is_reported():
    class SlowProvider:
        name = "slow"

        def identity(self):
            return providers.ProviderModelIdentity(
                provider="slow", model_id="slow", revision="r1",
                source="test", license_or_rights="test",
            )

        def capabilities(self):
            return providers.ProviderCapabilities(provider="slow", available=True, offline=True)

        def generate(self, request):
            return "never reaches this in the timeout test"

    import time

    class SleepingProvider(SlowProvider):
        def generate(self, request):
            time.sleep(0.2)
            return SlowProvider.generate(self, request)

    with pytest.raises(providers.ProviderTimeoutError):
        generate_with_timeout(
            SleepingProvider(),
            GenerationRequest(prompt="x", config=GenerationConfig(timeout_seconds=0.05)),
        )


def test_qualification_metadata_records_identity_and_configuration():
    provider = get_provider("qwen3-coder-30b-a3b-instruct")
    metadata = qualification_metadata(
        provider,
        suite_version="1.0.0",
        suite_sha256="abc123",
        config=GenerationConfig(binding_repair=False),
    )
    assert metadata["provider_identity"]["model_id"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert metadata["provider_identity"]["revision"] == "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
    assert metadata["suite"]["sha256"] == "abc123"
    assert metadata["generation_config"]["binding_repair"] is False
    assert metadata["capabilities"]["available"] is False
