"""Tests for the M5 report generator and candidate readiness assessment.

Both modules exist to make claims checkable, so the tests target the failure
modes that would let a false claim through.
"""

from __future__ import annotations

import json

import pytest

from evaluation.m5_candidates import CANDIDATES, Readiness, assess
from evaluation.m5_candidates import Finding
from evaluation.m5_report import comparison_report, format_report

CAND = CANDIDATES[0]


def write_run(root, name, *, complete=True, status="complete", score=None,
              hash_="h1"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": name, "status": status, "complete": complete,
        "blocker": None if complete else "no weights",
        "benchmark_manifest_hash": hash_,
        "model": {"model_id": "runs/x/checkpoint_best.pt",
                  "parameters": 10, "artifact_sha256": "a" * 64},
        "generation_config": {"strategy": "greedy"},
        "examples_total": 2, "examples_evaluated": 2,
        "elapsed_seconds": 1.0,
        "environment": {"python": "3.14.7"}, "git": {"commit": "abc"},
        "errors": [],
    }
    if complete:
        manifest["summary"] = score or {"syntax": {
            "metric": "syntax", "numerator": 1, "denominator": 2,
            "rate": 0.5, "preliminary": True, "note": ""}}
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")
    rows = [{"example_id": f"e{i}", "metrics": {}} for i in range(2)]
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return directory


def test_blocked_run_contributes_no_score(tmp_path):
    write_run(tmp_path, "evaluated")
    write_run(tmp_path, "blocked", complete=False, status="blocked")
    report = comparison_report(tmp_path)
    by_id = {r["run_id"]: r for r in report["runs"]}
    assert by_id["evaluated"]["evaluated"] is True
    assert by_id["blocked"]["evaluated"] is False
    assert by_id["blocked"]["summary"] is None
    assert report["runs_evaluated"] == ["evaluated"]


def test_blocked_run_never_renders_as_zero(tmp_path):
    write_run(tmp_path, "blocked", complete=False, status="blocked")
    text = format_report(comparison_report(tmp_path))
    assert "NOT RUN" in text
    assert "blocked" in text
    assert "0.0%" not in text, "a blocked run was rendered as a zero score"


def test_report_never_produces_a_composite(tmp_path):
    write_run(tmp_path, "evaluated")
    report = comparison_report(tmp_path)
    assert report["composite_score"] is None
    assert report["composite_note"]


def test_divergent_benchmarks_are_flagged(tmp_path):
    write_run(tmp_path, "a", hash_="h1")
    write_run(tmp_path, "b", hash_="h2")
    report = comparison_report(tmp_path)
    assert report["same_benchmark_across_runs"] is False
    assert report["benchmark_hashes_observed"] == ["h1", "h2"]


def test_identical_benchmarks_are_confirmed(tmp_path):
    write_run(tmp_path, "a", hash_="same")
    write_run(tmp_path, "b", hash_="same")
    assert comparison_report(tmp_path)["same_benchmark_across_runs"] is True


def test_count_mismatch_is_reported_not_hidden(tmp_path):
    directory = write_run(tmp_path, "evaluated")
    (directory / "results.jsonl").write_text(
        json.dumps({"example_id": "only-one", "metrics": {}}) + "\n",
        encoding="utf-8")
    report = comparison_report(tmp_path)
    check = report["runs"][0]["count_verification"]
    assert check["consistent"] is False
    assert check["reported_examples_evaluated"] == 2
    assert check["rows_with_metrics"] == 1


def test_missing_run_root_yields_an_empty_report(tmp_path):
    report = comparison_report(tmp_path / "does-not-exist")
    assert report["runs"] == []
    assert report["runs_evaluated"] == []


def test_directory_without_a_manifest_is_ignored(tmp_path):
    (tmp_path / "stray").mkdir()
    assert comparison_report(tmp_path)["runs"] == []


def test_preliminary_flag_survives_into_the_report(tmp_path):
    write_run(tmp_path, "evaluated")
    assert "[PRELIMINARY]" in format_report(comparison_report(tmp_path))


class TestCandidateReadiness:
    def test_missing_weights_block_readiness(self):
        result = assess(CAND, extra_paths=[])
        assert result.ready is False
        assert "weights_local" in [f.check for f in result.blockers]

    def test_licence_is_checked_before_anything_is_fetched(self):
        unverified = dict(CAND, license_id=None, license_verified=False)
        result = assess(unverified, extra_paths=[])
        assert "license" in [f.check for f in result.blockers]

    def test_download_is_never_performed_by_the_assessment(self):
        """Readiness must not touch the network or the HF cache."""
        assess(CAND, extra_paths=[])
        assess(dict(CAND, license_id=None, license_verified=False))
        assert all(f.check != "downloaded" for c in CANDIDATES
                   for f in assess(c).findings)

    def test_blockers_are_listed_by_check_name(self):
        payload = assess(CAND).to_dict()
        assert isinstance(payload["blockers"], list)
        assert all(isinstance(b, str) for b in payload["blockers"])
        assert payload["ready"] is False

    def test_serialisable(self):
        json.dumps(assess(CAND).to_dict())

    def test_a_non_finding_is_not_a_blocker(self):
        finding = Finding("sanity", ok=True, detail="fine")
        readiness = Readiness(model_id="m", family="f", parameters=1,
                              license_id="MIT", license_verified=True,
                              ready=True, findings=[finding])
        assert readiness.blockers == []
        assert readiness.ready is True


def test_ram_figure_is_reported_as_a_requirement_not_a_pass():
    result = assess(CAND)
    ram = next(f for f in result.findings if f.check == "ram")
    assert ram.ok is False
    assert "GiB" in ram.detail