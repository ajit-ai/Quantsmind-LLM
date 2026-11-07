"""Tests for the M6 runtime contract.

These use a stub runtime rather than real weights: the point is the contract's
bookkeeping (what was requested, what was honoured, what was refused), not
model quality.
"""

from __future__ import annotations

from evaluation.m6_runtime import (GenerationRequest, GenerationResponse,
                                   LocalModel, effective_config,
                                   unsupported_for)


class StubModel(LocalModel):
    runtime_name = "stub"

    def __init__(self, text="ok", *, load_ok=True):
        self.model_id = "stub/model"
        self.text = text
        self.load_ok = load_ok
        self._model = None

    def describe(self) -> dict:
        return {"model_id": self.model_id, "runtime": self.runtime_name,
                "task_head": "causal_lm", "artifact_sha256": "abc",
                "revision": "r1"}

    def preflight(self):
        return self.load_ok, "ok" if self.load_ok else "refused"

    def load(self):
        if not self.load_ok:
            raise RuntimeError("should not load when preflight refuses")
        self._model = object()

    def generate(self, request):
        return GenerationResponse(text=self.text, model_id=self.model_id,
                                  runtime=self.runtime_name,
                                  task_head="causal_lm",
                                  generation_config=effective_config(request))


def test_loaded_is_false_before_load_and_true_after():
    model = StubModel()
    assert model.loaded is False
    model.load()
    assert model.loaded is True


def test_health_is_safe_before_load():
    health = StubModel().health()
    assert health["loaded"] is False
    assert health["model_id"] == "stub/model"


def test_context_manager_loads_and_closes():
    model = StubModel()
    with model as entered:
        assert entered.loaded is True
    model.close()
    model.close()  # idempotent


def test_greedy_is_the_default_configuration():
    config = effective_config(GenerationRequest(prompt="x"))
    assert config["strategy"] == "greedy"
    assert config["do_sample"] is False
    assert "temperature" not in config


def test_sampling_configuration_records_its_parameters():
    config = effective_config(GenerationRequest(
        prompt="x", do_sample=True, temperature=0.7, top_p=0.9, top_k=40,
        seed=7, stop=["\n\n"]))
    assert config["strategy"] == "sampling"
    assert config["temperature"] == 0.7
    assert config["top_p"] == 0.9
    assert config["top_k"] == 40
    assert config["seed"] == 7
    assert config["stop"] == ["\n\n"]


def test_top_k_is_omitted_when_zero():
    config = effective_config(GenerationRequest(prompt="x", do_sample=True,
                                                top_k=0))
    assert "top_k" not in config


def test_embedding_head_reports_what_it_cannot_do():
    reasons = unsupported_for("embedding", GenerationRequest(
        prompt="x", do_sample=True, stop=["z"]))
    joined = " ".join(reasons)
    assert "max_new_tokens" in joined
    assert "do_sample" in joined
    assert "stop" in joined


def test_masked_lm_reports_what_it_cannot_do():
    reasons = unsupported_for("masked_lm", GenerationRequest(prompt="x"))
    assert any("autoregressively" in reason for reason in reasons)


def test_causal_lm_supports_every_setting():
    assert unsupported_for("causal_lm", GenerationRequest(
        prompt="x", do_sample=True, stop=["z"])) == []


def test_response_round_trips_to_dict():
    response = StubModel().generate(GenerationRequest(prompt="x"))
    payload = response.to_dict()
    assert payload["text"] == "ok"
    assert payload["schema"]
    assert payload["model_id"] == "stub/model"
