"""Focused tests for the D3 Seed-Coder adapter."""

from __future__ import annotations

import pytest

from serving import providers
from serving.providers import (
    GenerationConfig,
    GenerationRequest,
    ProviderUnavailableError,
    SeedCoderProvider,
    get_provider,
    qualification_metadata,
)


class FakeTensor:
    def __init__(self, shape=(1, 3)):
        self.shape = shape


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 7
    model_max_length = 32768

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered-chat-prompt"

    def __call__(self, text, return_tensors=None):
        assert return_tensors == "pt"
        return FakeInputs({"input_ids": FakeTensor((1, 3))})

    def decode(self, ids, skip_special_tokens=True):
        return "    return value + 1\n"


class FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 8
        return [[1, 2, 3, 4, 5]]


def _model_dir(tmp_path):
    path = tmp_path / "seed-coder"
    path.mkdir()
    (path / "config.json").write_text('{"architectures":["LlamaForCausalLM"]}', encoding="utf-8")
    (path / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    return path


def test_seed_identity_and_pinned_revision(tmp_path):
    provider = SeedCoderProvider(_model_dir(tmp_path))
    identity = provider.identity()
    assert provider.name == "seed-coder-8b-instruct"
    assert identity.model_id == "ByteDance-Seed/Seed-Coder-8B-Instruct"
    assert identity.revision == "1c5af72afa21c22c9272a0b60a3a404195e88fcd"
    assert identity.license_or_rights.startswith("MIT")
    assert provider.capabilities().offline is True
    assert "chat_template" in provider.capabilities().modes


def test_seed_revision_mismatch_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Seed-Coder revision must be"):
        SeedCoderProvider(_model_dir(tmp_path), revision="wrong-revision")


def test_seed_remote_code_is_not_permitted(tmp_path):
    with pytest.raises(ValueError, match="trust_remote_code"):
        SeedCoderProvider(_model_dir(tmp_path), trust_remote_code=True)


def test_seed_missing_checkpoint_is_unavailable(tmp_path):
    provider = SeedCoderProvider(tmp_path / "missing")
    assert provider.capabilities().available is False
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest(prompt="write a function"))


def test_seed_artifact_hashes_are_recorded(tmp_path):
    provider = SeedCoderProvider(_model_dir(tmp_path))
    hashes = provider.artifact_hashes()
    assert hashes["config.json"] is not None
    assert hashes["tokenizer.json"] is not None
    assert hashes["model.safetensors.index.json"] is None


def test_seed_generation_preserves_raw_and_applies_repair(monkeypatch, tmp_path):
    model_dir = _model_dir(tmp_path)
    provider = SeedCoderProvider(model_dir)

    def fake_load(model_path, *, device, dtype):
        assert model_path == model_dir
        assert device == "cpu"
        return FakeModel(), FakeTokenizer()

    monkeypatch.setattr(providers, "_load_seed_resources", fake_load)
    result = provider.generate(
        GenerationRequest(
            prompt="def add(first):",
            config=GenerationConfig(max_new_tokens=8, top_k=1, binding_repair=True),
        )
    )
    assert result.provider == "seed-coder-8b-instruct"
    assert result.raw_text == "    return value + 1\n"
    assert result.text == "    return first + 1\n"
    assert result.repair["status"] == "repaired"
    assert result.metadata["artifact_hashes"]["config.json"] is not None


def test_seed_generation_can_disable_repair(monkeypatch, tmp_path):
    provider = SeedCoderProvider(_model_dir(tmp_path))
    monkeypatch.setattr(
        providers,
        "_load_seed_resources",
        lambda model_path, *, device, dtype: (FakeModel(), FakeTokenizer()),
    )
    result = provider.generate(
        GenerationRequest(
            prompt="def add(first):",
            config=GenerationConfig(max_new_tokens=8, top_k=1, binding_repair=False),
        )
    )
    assert result.text == result.raw_text == "    return value + 1\n"
    assert result.repair["status"] == "disabled"


def test_seed_provider_registry_selection(tmp_path):
    provider = get_provider("seed-coder-8b-instruct", model_path=str(_model_dir(tmp_path)))
    assert isinstance(provider, SeedCoderProvider)
    assert provider.identity().provider == "seed-coder-8b-instruct"


def test_seed_missing_path_does_not_fall_back_to_s2():
    provider = get_provider("seed-coder-8b-instruct")
    assert provider.capabilities().available is False
    with pytest.raises(ProviderUnavailableError):
        provider.generate(GenerationRequest(prompt="write a function"))


def test_seed_qualification_metadata(tmp_path):
    provider = SeedCoderProvider(_model_dir(tmp_path))
    metadata = qualification_metadata(
        provider,
        suite_version="1.0.0",
        suite_sha256="abc123",
        config=GenerationConfig(binding_repair=False),
    )
    assert metadata["provider_identity"]["revision"] == "1c5af72afa21c22c9272a0b60a3a404195e88fcd"
    assert metadata["provider_identity"]["license_or_rights"].startswith("MIT")
    assert metadata["generation_config"]["binding_repair"] is False
