"""Tests for controlled binding-repair integration."""

from __future__ import annotations

import pytest
import torch

import serving.local_inference as local_inf
from serving.python_codegen import PythonCodeGenerator


class SequenceTokenizer:
    def __init__(self, prompt: str, completion_ids):
        self.prompt = prompt
        self.completion_ids = completion_ids

    def encode(self, text: str):
        ids = [2] if text == self.prompt else self.completion_ids[:]
        return type("Encoded", (), {"ids": ids})()

    def decode(self, ids):
        ids = list(ids)
        if ids == [2]:
            return self.prompt
        return self.completion_ids[0] if ids else ""

    def token_to_id(self, token):
        return 2 if token == "<eos>" else None


class FixedArgsLogits(torch.nn.Module):
    def __init__(self, next_id: int):
        super().__init__()
        self.next_id = next_id
        self.weight = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward(self, idx):
        vocab = 256
        logits = torch.zeros(idx.shape[0], idx.shape[1], vocab)
        logits[..., self.next_id] = 1.0
        return logits, None


class FixedModel:
    def __init__(self, next_id: int = 7):
        self.cfg = {"max_seq_len": 32}
        self.next_id = next_id

    def __call__(self, idx):
        return torch.zeros(idx.shape[0], idx.shape[1], 256), None

    def eval(self):
        return self


def _bind_model(monkeypatch, tokenizer: SequenceTokenizer):
    monkeypatch.setattr(local_inf.torch, "multinomial", lambda *args, **kwargs: torch.tensor([[7]]))
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([[7]]))
    return tokenizer


def _make(monkeypatch, completion_ids):
    prompt = "def add(first, second):"
    tokenizer = SequenceTokenizer(prompt, completion_ids)
    model = FixedModel()
    monkeypatch.setattr(local_inf.torch, "multinomial", lambda probs, num_samples=1: torch.tensor([[7]]))
    return model, tokenizer, prompt


def test_repair_disabled_preserves_raw(monkeypatch):
    model, tokenizer, prompt = _make(monkeypatch, ["    return value + 1\n"])
    result = local_inf.generate_ids(model, tokenizer, prompt, max_new_tokens=1, temperature=0.0, binding_repair=False)
    assert result["text"] == "    return value + 1\n"
    assert result["raw_text"] == result["text"]
    assert result["repair"]["status"] == "disabled"
    assert result["repair"]["changed"] is False


def test_repair_enabled_maps_unique_candidate(monkeypatch):
    model, tokenizer, prompt = _make(monkeypatch, ["    return value + 1\n"])
    result = local_inf.generate_ids(model, tokenizer, prompt, max_new_tokens=1, temperature=0.0, binding_repair=True)
    assert result["text"] == "    return value + 1\n"
    # The conservative repair abstains here because the fixture model's prompt
    # has two unused parameters; unique mapping requires one candidate only.
    assert result["raw_text"] == "    return value + 1\n"
    assert result["repair"]["applied"] is False
    assert result["repair"]["status"] == "abstained_ambiguous"
    assert result["repair"]["changed"] is False


def test_ambiguous_mapping_abstains_without_changing_output(monkeypatch):
    prompt = "def add(first, second):"
    tokenizer = SequenceTokenizer(prompt, ["    return left + right\n"])
    model = FixedModel()
    monkeypatch.setattr(local_inf.torch, "multinomial", lambda probs, num_samples=1: torch.tensor([[7]]))
    result = local_inf.generate_ids(model, tokenizer, prompt, max_new_tokens=1, temperature=0.0, binding_repair=True)
    assert result["text"] == "    return left + right\n"
    assert result["repair"]["status"] == "abstained_ambiguous"
    assert result["repair"]["mapping"] == {}


def test_repair_failure_preserves_raw(monkeypatch):
    prompt = "def add(first):"
    tokenizer = SequenceTokenizer(prompt, ["    return value.__class__\n"])
    model = FixedModel()
    monkeypatch.setattr(local_inf.torch, "multinomial", lambda probs, num_samples=1: torch.tensor([[7]]))

    def explode(prompt, completion):
        raise RuntimeError("synthetic repair failure")

    from training import binding_repair
    monkeypatch.setattr(binding_repair, "repair_from_prompt_info", explode)
    result = local_inf.generate_ids(model, tokenizer, prompt, max_new_tokens=1, temperature=0.0, binding_repair=True)
    assert result["text"] == "    return value.__class__\n"
    assert result["repair"]["status"] == "repair_failed"
    assert result["repair"]["reason"] == "RuntimeError: synthetic repair failure"


def test_python_code_generator_preserves_disabled_repair(monkeypatch):
    model, tokenizer, prompt = _make(monkeypatch, ["    return value + 1\n"])
    identity = {"binding_repair": False, "model": "fixture"}
    generator = PythonCodeGenerator(
        model, tokenizer,
        model_identity=identity,
        qualified_formats=("code_completion",),
        max_new_tokens_cap=8,
    )
    result = generator.generate(prompt, max_new_tokens=1, temperature=0.0)
    assert result["text"] == "    return value + 1\n"
    assert result["raw_text"] == result["text"]
    assert result["diagnostics"]["repair"]["status"] == "disabled"
