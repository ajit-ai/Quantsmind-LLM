"""Tests for the M5 runner: manifests, resume and selective rerun.

A stub adapter stands in for a real model so these tests never load weights.
"""

from __future__ import annotations

import json

from evaluation.m5_runner import evaluate, write_blocked_run


class StubAdapter:
    """Deterministic stand-in. Records what it was asked to generate."""

    def __init__(self, outputs=None):
        self.model_id = "stub-model"
        self.outputs = outputs or {}
        self.calls = []

    def preflight(self):
        return True, "stub"

    def info(self):
        from evaluation.m5_adapters import AdapterInfo

        return AdapterInfo(adapter="stub", model_id=self.model_id,
                           family="stub", parameters=1)

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return {"text": self.outputs.get(prompt, "stub output"),
                "token_ids": [1], "finish_reason": "length",
                "prompt_tokens": 1, "completion_tokens": 1,
                "seconds": 0.001, "tokens_per_second": 1000.0}

    def release(self):
        pass


def write_benchmark(tmp_path, count=3):
    from evaluation.m5_schema import stable_id

    records = []
    for index in range(count):
        prompt = f"task {index}"
        records.append({
            "id": stable_id("code_generation", prompt),
            "task_type": "code_generation", "prompt": prompt,
            "expected_behavior": "x", "test_specification": "reverse_string",
            "language": "python", "source_id": "s", "license_id": "MIT",
            "split": "eval",
            "provenance": {"origin": "t", "license_evidence": "MIT"},
        })
    path = tmp_path / "bench.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records),
                    encoding="utf-8")
    return path


def read_results(out_dir):
    return [json.loads(l) for l in
            (out_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]


def test_runner_records_manifest_and_results(tmp_path):
    bench = write_benchmark(tmp_path)
    manifest = evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    assert manifest["status"] == "complete"
    assert manifest["examples_evaluated"] == 3
    assert manifest["benchmark_manifest_hash"]
    assert manifest["generation_config"]["strategy"] == "greedy"
    assert manifest["environment"]["python"]
    assert manifest["git"]["commit"]
    assert manifest["git"]["dirty_tree"] is not None
    assert (tmp_path / "run" / "summary.json").exists()
    assert (tmp_path / "run" / "results.jsonl").exists()


def test_output_records_link_model_and_example(tmp_path):
    bench = write_benchmark(tmp_path)
    evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    for row in read_results(tmp_path / "run"):
        assert row["example_id"]
        assert row["model_id"] == "stub-model"
        assert row["task_type"] == "code_generation"
        assert "metrics" in row


def test_resume_does_not_duplicate_results(tmp_path):
    bench = write_benchmark(tmp_path)
    evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    second = StubAdapter()
    manifest = evaluate(second, bench, tmp_path / "run", label="t")
    assert manifest["examples_reused_from_previous_run"] == 3
    assert second.calls == [], "resumed run regenerated completed examples"
    assert len(read_results(tmp_path / "run")) == 3


def test_selective_rerun_targets_only_the_named_example(tmp_path):
    bench = write_benchmark(tmp_path, count=3)
    evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    target = read_results(tmp_path / "run")[0]["example_id"]

    adapter = StubAdapter()
    evaluate(adapter, bench, tmp_path / "run", label="t", only=[target])
    assert len(adapter.calls) == 1
    rows = read_results(tmp_path / "run")
    # A selective rerun appends a replacement record; it does not delete the
    # successful results it did not target.
    assert len({r["example_id"] for r in rows}) == 3
    assert len(rows) == 4


def test_generation_error_does_not_kill_the_run(tmp_path):
    bench = write_benchmark(tmp_path, count=2)

    class Failing(StubAdapter):
        def generate(self, prompt, **kwargs):
            if prompt == "task 0":
                raise RuntimeError("synthetic generation failure")
            return super().generate(prompt, **kwargs)

    manifest = evaluate(Failing(), bench, tmp_path / "run", label="t")
    assert manifest["status"] == "complete_with_errors"
    assert len(manifest["errors"]) == 1
    assert manifest["examples_evaluated"] == 1


def test_blocked_run_never_claims_evaluation(tmp_path):
    manifest = write_blocked_run(tmp_path / "run", "some/model",
                                 "no weights and download not permitted")
    assert manifest["status"] == "blocked"
    assert manifest["complete"] is False
    assert manifest["evaluated"] is False
    on_disk = json.loads(
        (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "blocked"
    assert "summary" not in on_disk


def test_no_resume_does_not_inherit_previous_results(tmp_path):
    """A fresh run must rewrite results, not append a second generation of them."""
    bench = write_benchmark(tmp_path, count=2)
    evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    adapter = StubAdapter()
    manifest = evaluate(adapter, bench, tmp_path / "run", label="t",
                        resume=False)
    assert len(adapter.calls) == 2, "resume=False should regenerate everything"
    assert manifest["examples_evaluated"] == 2
    rows = read_results(tmp_path / "run")
    assert len(rows) == 2, "stale rows from the previous run were retained"
    assert len({r["example_id"] for r in rows}) == 2


def test_interrupted_results_line_is_ignored_on_resume(tmp_path):
    """A truncated final line from a killed run must not corrupt the resume."""
    bench = write_benchmark(tmp_path, count=2)
    evaluate(StubAdapter(), bench, tmp_path / "run", label="t")
    results = tmp_path / "run" / "results.jsonl"
    with results.open("a", encoding="utf-8") as handle:
        handle.write('{"example_id": "half-writ')  # killed mid-write

    adapter = StubAdapter()
    manifest = evaluate(adapter, bench, tmp_path / "run", label="t")
    assert manifest["examples_reused_from_previous_run"] == 2
    assert adapter.calls == []