"""Tests for the M6 consolidated report and strategy decision.

The report is the artefact a reader will trust, so the tests target the ways it
could overstate: crediting a blocked run, promoting vacuous compliance, or
recommending a practical track that was never evaluated.
"""

from __future__ import annotations

import json

from evaluation.m6_report import (build_report, capability_table, load_runs,
                                  strategy_decision)


def metric(n, d):
    return {"numerator": n, "denominator": d,
            "rate": (n / d) if d else None, "preliminary": True}


def write_run(root, name, *, complete=True, status="complete", head="causal_lm",
              candidate=None, compliance=(0, 0), functional=(0, 0),
              syntax=(0, 20), hash_="h1", checks=None):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": name, "status": status, "complete": complete,
        "benchmark_manifest_hash": hash_,
        "model": {"model_id": candidate or name},
        "examples_total": 26, "examples_evaluated": 26, "elapsed_seconds": 1.0,
        "m6": {"candidate_id": candidate or name, "task_head": head,
               "runtime": "fake", "license_id": "MIT",
               "blocked_reason": None if complete else "no weights",
               "hosted_api_used": False},
    }
    if complete:
        manifest["summary"] = {
            "syntax": metric(*syntax), "functional": metric(*functional),
            "compliance": metric(*compliance)}
        manifest["per_task"] = {"code_generation": {
            "syntax": metric(*syntax)}}
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    rows = [{"example_id": "e1", "output": "a real answer longer than a stub",
             "metrics": {"compliance": {
                 "denominator": len(checks or {}),
                 "numerator": sum((checks or {}).values()),
                 "checks": [{"rule": k, "ok": v}
                            for k, v in (checks or {}).items()]}}}]
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return directory


def test_blocked_run_is_not_evaluated_and_has_no_summary(tmp_path):
    write_run(tmp_path, "ok")
    write_run(tmp_path, "blocked", complete=False, status="blocked")
    runs = {r["run_id"]: r for r in load_runs(tmp_path)}
    assert runs["blocked"]["evaluated"] is False
    assert runs["blocked"]["summary"] is None
    assert runs["blocked"]["blocked_reason"] == "no weights"


def test_vacuous_compliance_does_not_raise_evidence_level(tmp_path):
    write_run(tmp_path, "weak", compliance=(4, 6),
              checks={"non_empty": True, "no_fences": True})
    report = build_report(tmp_path, inventory=None, sovereign=None)
    evidence = report["capability"]["weak"]["evidence"]
    assert evidence["highest_level"] < 5


def test_no_practical_candidate_yields_insufficient_evidence(tmp_path):
    write_run(tmp_path, "sovereign", candidate="quantsmind/m3-base",
              syntax=(1, 20))
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["strategy"]["recommendation"] == "INSUFFICIENT_EVIDENCE"
    assert report["strategy"]["provisional"] is True
    assert report["strategy"]["options"]["B_practical_open_weight"]["status"] \
        == "BLOCKED"


def test_practical_run_without_functional_pass_is_hybrid_but_provisional(
        tmp_path):
    write_run(tmp_path, "external", candidate="Qwen/Qwen2.5-Coder-0.5B",
              syntax=(2, 20))
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["strategy"]["recommendation"] == "HYBRID_TRACK"
    assert report["strategy"]["provisional"] is True


def test_practical_functional_pass_supports_practical_track(tmp_path):
    write_run(tmp_path, "external", candidate="Qwen/Qwen2.5-Coder-0.5B",
              functional=(2, 20), syntax=(2, 20))
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["strategy"]["recommendation"] == "PRACTICAL_OPEN_WEIGHT_TRACK"
    assert report["strategy"]["provisional"] is False


def test_no_composite_score_is_ever_emitted(tmp_path):
    write_run(tmp_path, "a", syntax=(5, 20), compliance=(5, 6))
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["composite_score"] is None
    assert "formatting" in report["composite_note"]


def test_divergent_benchmark_hashes_are_flagged(tmp_path):
    write_run(tmp_path, "a", hash_="h1")
    write_run(tmp_path, "b", hash_="h2")
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["same_benchmark_across_runs"] is False
    assert report["benchmark_hashes"] == ["h1", "h2"]


def test_capability_table_skips_blocked_runs(tmp_path):
    write_run(tmp_path, "ok")
    write_run(tmp_path, "blocked", complete=False, status="blocked")
    table = capability_table(load_runs(tmp_path))
    assert list(table) == ["ok"]


def test_missing_root_yields_empty_report():
    report = build_report("does-not-exist", inventory=None, sovereign=None)
    assert report["runs"] == []
    assert report["runs_evaluated"] == []
    assert report["strategy"]["recommendation"] == "INSUFFICIENT_EVIDENCE"


def test_limitations_are_present_and_state_the_unknowns(tmp_path):
    report = build_report(tmp_path, inventory=None, sovereign=None)
    joined = " ".join(report["limitations"]).lower()
    assert "26 examples" in joined
    assert "unavailable" in joined


def test_embedding_head_is_not_treated_as_a_practical_candidate(tmp_path):
    write_run(tmp_path, "embed", head="embedding", candidate="sentence/a",
              compliance=(4, 6), checks={"non_empty": True})
    report = build_report(tmp_path, inventory=None, sovereign=None)
    assert report["strategy"]["recommendation"] == "INSUFFICIENT_EVIDENCE"
