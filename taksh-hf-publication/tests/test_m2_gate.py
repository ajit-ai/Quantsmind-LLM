"""Tests for the M2 readiness gate (M2.12).

The gate must not be rigged in either direction: it has to fail when evidence is
missing, and it has to be capable of returning M2 READY when everything really
is in place. A gate that can only ever report failure is not a gate.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import yaml

from data import m2_gate
from data.m2_gate import BLOCKED, FAIL, PASS, Gate, run_gate


def make_release(tmp_path, corpus_path):
    from data.release import build_release
    out = tmp_path / "rel"
    build_release(corpus_path, out, "d1")
    return out


def make_satisfied_world(tmp_path):
    """Build a complete, self-consistent world in which every check can pass."""
    from data.release import sha256_file

    splits = [("train", f"org/repo{i}") for i in range(m2_gate.TARGET_SOURCE_COUNT)]
    splits += [("test", "org/heldout"), ("val", "org/val")]

    records = []
    for index, (split, repo) in enumerate(splits):
        text = f"# file {index}\n" + "def f():\n    return 1\n" * 20
        records.append({
            "text": text,
            # The schema recomputes this, so a placeholder would not conform.
            "content_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(),
            "language": "python",
            "source_id": repo.replace("/", "-"),
            "repo": repo,
            "commit": "a" * 40,
            "spdx": "MIT",
            "attribution": f"Copyright {repo}; licensed under MIT",
            "file_path": f"{index}.py",
            "retrieved_at": "2026-10-14",
            "split": split,
        })
    corpus = tmp_path / "data" / "corpus.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )

    token_bin = corpus.with_suffix(".tokens.bin")
    token_bin.write_bytes(b"\x00\x01" * 64)
    token_meta = corpus.with_suffix(".tokens.meta.json")
    token_meta.write_text(json.dumps({
        "vocab_size": 512, "num_tokens": m2_gate.TARGET_TOKENS,
        "num_documents": len(records), "dataset_schema_validated": True,
        "corpus_sha256": sha256_file(corpus),
    }), encoding="utf-8")

    release = make_release(tmp_path, corpus)

    licence = tmp_path / "licence.json"
    licence.write_text(json.dumps({
        "counts": {"checked": 1, "verified": 1, "rejected": 0,
                   "needs_human_review": 0},
        "verified_ids": ["a"], "needs_human_review_ids": [],
        "all_verified": True,
    }), encoding="utf-8")

    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text('{"task_id": "test/0"}\n', encoding="utf-8")
    holdout_manifest = tmp_path / "holdout.manifest.json"
    holdout_manifest.write_text(json.dumps({
        "items": 1, "repositories": ["org/heldout"],
        "separation": {"ok": True},
        "evaluation_protocol": {
            "executable_benchmark": False,
            "published_suites_status": "unavailable",
            "limitations": ["not executable"],
        },
    }), encoding="utf-8")

    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(yaml.safe_dump({"model": {"vocab_size": 512}}),
                         encoding="utf-8")
    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({
        "run": {"model_config": str(model_cfg)},
        "data": {"dataset_tag": "corpus"},
    }), encoding="utf-8")

    return {
        "corpus": str(corpus),
        "token_meta": str(token_meta),
        "licence_report": str(licence),
        "release_dir": str(release),
        "train_config": str(config),
        "holdout": str(holdout),
        "holdout_manifest": str(holdout_manifest),
    }


def find(report, check_id):
    return next(c for c in report["checks"] if c["id"] == check_id)


@pytest.fixture
def satisfied(tmp_path, monkeypatch):
    """A world in which every artifact-backed check passes."""
    monkeypatch.setattr(m2_gate, "check_human_eval", lambda gate: gate)
    kwargs = make_satisfied_world(tmp_path)
    monkeypatch.chdir(tmp_path)
    return kwargs


# --------------------------------------------------------------------------- #
# The gate can actually pass
# --------------------------------------------------------------------------- #
def test_gate_reports_ready_when_everything_is_satisfied(satisfied):
    report = run_gate(**satisfied)
    failing = [c for c in report["checks"] if c["status"] != PASS]
    assert report["verdict"] == "M2 READY", json.dumps(failing, indent=2)
    assert report["counts"][FAIL] == 0
    assert report["counts"][BLOCKED] == 0


def test_satisfied_world_evidence_is_real(satisfied):
    report = run_gate(**satisfied)
    schema = find(report, "M2.4-schema")
    assert schema["evidence"].startswith("22/22 valid")
    assert "22 distinct repositories" in find(report, "M2.3-scale")["evidence"]
    assert "100,000,000 tokens" in find(report, "M2.3-tokens")["evidence"]


# --------------------------------------------------------------------------- #
# Evidence, not hardcoded results
# --------------------------------------------------------------------------- #
def test_missing_corpus_fails_rather_than_passing(tmp_path):
    report = run_gate(corpus=str(tmp_path / "absent.jsonl"))
    assert find(report, "M2.4-schema")["status"] == FAIL
    assert report["verdict"] == "M2 BLOCKED"


def test_tampered_release_fails_the_release_check(satisfied, tmp_path):
    (tmp_path / "rel" / "corpus.jsonl").write_text("{}\n", encoding="utf-8")
    report = run_gate(**satisfied)
    assert find(report, "M2.8-release")["status"] == FAIL


def test_vocab_mismatch_fails_training_readiness(satisfied, tmp_path):
    (tmp_path / "model.yaml").write_text(
        yaml.safe_dump({"model": {"vocab_size": 16}}), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    check = find(report, "M2.9-training")
    assert check["status"] == FAIL
    assert "out of range" in check["detail"]


def test_unvalidated_token_artifact_fails_training_readiness(satisfied, tmp_path):
    meta = tmp_path / "data" / "corpus.tokens.meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["dataset_schema_validated"] = False
    meta.write_text(json.dumps(data), encoding="utf-8")
    report = run_gate(**satisfied)
    assert find(report, "M2.9-training")["status"] == FAIL


def test_config_pointing_at_a_missing_artifact_fails(satisfied, tmp_path):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump({
        "run": {"model_config": "configs/model/x.yaml"},
        "data": {"dataset_tag": "not-built"},
    }), encoding="utf-8")
    report = run_gate(**satisfied)
    assert find(report, "M2.9-training")["status"] == FAIL


def test_config_without_a_dataset_tag_fails(satisfied, tmp_path):
    (tmp_path / "train.yaml").write_text(
        yaml.safe_dump({"run": {}, "data": {}}), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    assert find(report, "M2.9-training")["status"] == FAIL


def test_short_token_corpus_fails_the_scale_check(satisfied, tmp_path):
    meta = tmp_path / "data" / "corpus.tokens.meta.json"
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["num_tokens"] = 1000
    meta.write_text(json.dumps(data), encoding="utf-8")
    report = run_gate(**satisfied)
    check = find(report, "M2.3-tokens")
    assert check["status"] == FAIL
    assert "%" in check["evidence"]


def test_missing_split_fails_the_separation_check(satisfied, tmp_path):
    rows = [json.loads(line) for line in
            (tmp_path / "data" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        row["split"] = "train"
    (tmp_path / "data" / "corpus.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    check = find(report, "M2.6-splits")
    assert check["status"] == FAIL
    assert "missing split" in check["detail"]


def test_holdout_that_leaks_into_training_fails(satisfied, tmp_path):
    manifest = json.loads(
        (tmp_path / "holdout.manifest.json").read_text(encoding="utf-8")
    )
    manifest["separation"] = {"ok": False, "problem": "shared repositories"}
    (tmp_path / "holdout.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    assert find(report, "M2.10-holdout")["status"] == FAIL


def test_manifest_claiming_an_executable_benchmark_is_rejected(satisfied, tmp_path):
    manifest = json.loads(
        (tmp_path / "holdout.manifest.json").read_text(encoding="utf-8")
    )
    manifest["evaluation_protocol"]["executable_benchmark"] = True
    (tmp_path / "holdout.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    assert find(report, "M2.10-benchmark")["status"] == FAIL


def test_manifest_without_limitations_fails(satisfied, tmp_path):
    manifest = json.loads(
        (tmp_path / "holdout.manifest.json").read_text(encoding="utf-8")
    )
    manifest["evaluation_protocol"]["limitations"] = []
    (tmp_path / "holdout.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    assert find(report, "M2.10-holdout")["status"] == FAIL


# --------------------------------------------------------------------------- #
# Blocked means "a human must decide", not "the code is broken"
# --------------------------------------------------------------------------- #
def test_unresolved_licence_is_blocked_not_failed(satisfied, tmp_path):
    (tmp_path / "licence.json").write_text(json.dumps({
        "counts": {"checked": 9, "verified": 7, "rejected": 2,
                   "needs_human_review": 2},
        "verified_ids": [], "needs_human_review_ids": ["psf-black"],
        "all_verified": False,
    }), encoding="utf-8")
    report = run_gate(**satisfied)
    check = find(report, "M2.3-licence")
    assert check["status"] == BLOCKED
    assert "human" in check["remedy"].lower()
    assert any("psf-black" in d["detail"]
               for d in report["human_decisions_required"])


def test_too_few_sources_is_blocked_not_failed(satisfied, tmp_path):
    rows = [json.loads(line) for line in
            (tmp_path / "data" / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        row["repo"] = "org/only"
    (tmp_path / "data" / "corpus.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    report = run_gate(**satisfied)
    check = find(report, "M2.3-scale")
    assert check["status"] == BLOCKED
    assert "authoris" in check["remedy"]


def test_human_eval_is_always_blocked(tmp_path):
    """Human evaluation cannot be self-certified, even in a perfect world."""
    report = run_gate()
    check = find(report, "M2.11-human-eval")
    assert check["status"] == BLOCKED
    assert "no annotators" in check["detail"]


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #
def test_repository_state_is_blocked_with_named_reasons():
    """The shipped repository is not ready, and the gate must say so plainly."""
    report = run_gate()
    assert report["verdict"] == "M2 BLOCKED"
    assert report["counts"][BLOCKED] >= 2
    ids = {d["id"] for d in report["human_decisions_required"]}
    assert {"M2.3-licence", "M2.11-human-eval"} <= ids


def test_gate_counts_match_the_check_list():
    report = run_gate()
    counted = {s: sum(1 for c in report["checks"] if c["status"] == s)
               for s in (PASS, FAIL, BLOCKED)}
    assert report["counts"] == counted
    assert sum(counted.values()) == len(report["checks"])


def test_every_check_names_a_requirement():
    report = run_gate()
    for check in report["checks"]:
        assert check["requirement"]
        assert check["status"] in (PASS, FAIL, BLOCKED)


def test_gate_object_records_evidence_for_passes():
    gate = Gate()
    gate.ok("x", "requirement text", "evidence text")
    gate.fail("y", "requirement text", "detail text")
    gate.blocked("z", "requirement text", "detail text")
    assert gate.checks[0]["evidence"] == "evidence text"
    assert gate.counts() == {PASS: 1, FAIL: 1, BLOCKED: 1}