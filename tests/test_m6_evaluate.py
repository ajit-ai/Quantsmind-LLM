"""Tests for the M6/M5 evaluation bridge.

The bridge must not reimplement M5 scoring, and it must never turn a model that
could not load into a run that "scored zero".
"""

from __future__ import annotations

import json

from evaluation.m6_evaluate import (M5BridgeAdapter, per_task_summary,
                                    read_results, record_blocked)
from evaluation.m6_runtime import GenerationRequest, GenerationResponse, LocalModel


class StubModel(LocalModel):
    runtime_name = "stub"
    model_id = "stub/model"

    def __init__(self, text="def f():\n    pass\n", load_ok=True):
        self.text = text
        self.load_ok = load_ok
        self._model = None
        self.loaded_once = False

    def describe(self):
        return {"model_id": self.model_id, "runtime": self.runtime_name,
                "task_head": "causal_lm", "artifact_sha256": "a" * 64,
                "revision": "r1", "parameters": 123}

    def preflight(self):
        return self.load_ok, "ok" if self.load_ok else "insufficient RAM"

    def load(self):
        self.loaded_once = True
        self._model = object()

    def generate(self, request):
        return GenerationResponse(
            text=self.text, model_id=self.model_id, runtime=self.runtime_name,
            task_head="causal_lm", input_tokens=4, output_tokens=2,
            duration_seconds=0.5)

    def close(self):
        self._model = None


def test_info_carries_provenance_without_loading():
    model = StubModel()
    info = M5BridgeAdapter(model, license_record={
        "license_id": "MIT", "license_verified": True}).info()
    assert model.loaded_once is False
    assert info.license_id == "MIT"
    assert info.license_verified is True
    assert info.artifact_sha256 == "a" * 64
    assert info.revision == "r1"


def test_generate_translates_response_into_the_m5_shape():
    adapter = M5BridgeAdapter(StubModel(text="hello"))
    payload = adapter.generate("prompt", max_new_tokens=8)
    assert payload["text"] == "hello"
    assert payload["completion_tokens"] == 2
    assert payload["tokens_per_second"] == 4.0
    assert payload["token_ids"] is None


def test_generate_maps_greedy_onto_do_sample_false():
    captured = {}

    class Capturing(StubModel):
        def generate(self, request):
            captured["request"] = request
            return super().generate(request)

    M5BridgeAdapter(Capturing()).generate("p", greedy=True)
    assert captured["request"].do_sample is False
    M5BridgeAdapter(Capturing()).generate("p", greedy=False)
    assert captured["request"].do_sample is True


def test_stop_list_is_copied_not_shared():
    captured = {}

    class Capturing(StubModel):
        def generate(self, request):
            captured["request"] = request
            return super().generate(request)

    stops = ["\n\n"]
    M5BridgeAdapter(Capturing()).generate("p", stop=stops)
    assert captured["request"].stop == ["\n\n"]
    assert captured["request"].stop is not stops


def test_release_closes_the_runtime():
    model = StubModel()
    model.load()
    M5BridgeAdapter(model).release()
    assert model.loaded is False


def test_blocked_run_records_the_reason_and_stays_incomplete(tmp_path):
    manifest = record_blocked(tmp_path, "org/model", "no weights on disk")
    on_disk = json.loads((tmp_path / "run_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["complete"] is False
    assert "summary" not in manifest
    assert on_disk["m6"]["blocked_reason"] == "no weights on disk"


def test_per_task_summary_is_taken_from_the_manifest(tmp_path):
    summary = per_task_summary({"by_task": {"code_generation": {
        "syntax": {"numerator": 1, "denominator": 2, "rate": 0.5,
                   "checks": ["dropped"]}}}})
    assert summary["code_generation"]["syntax"] == {
        "numerator": 1, "denominator": 2, "rate": 0.5}


def test_read_results_skips_malformed_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"a": 1}\nnot json\n\n{"a": 2}\n', encoding="utf-8")
    assert read_results(path) == [{"a": 1}, {"a": 2}]
    assert read_results(tmp_path / "missing.jsonl") == []
    assert read_results(None) == []


def test_generation_request_defaults_are_explicit():
    request = GenerationRequest(prompt="x")
    assert request.do_sample is False
    assert request.max_new_tokens == 96
    assert request.stop == []
