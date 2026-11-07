"""Tests for evaluation run records (M2.10 reproducibility)."""

from __future__ import annotations

import json

import pytest

from evaluation.run_manifest import build_run_record, sha256_file


def make_release(tmp_path, corpus_name="corpus.jsonl"):
    from data.release import build_release
    corpus = tmp_path / corpus_name
    corpus.write_text('{"text": "x"}\n', encoding="utf-8")
    out = tmp_path / "rel"
    build_release(corpus, out, "d1")
    return out, corpus


def full_kwargs(tmp_path):
    release, _corpus = make_release(tmp_path)
    holdout = tmp_path / "holdout.jsonl"
    holdout.write_text('{"task_id": "t/0"}\n', encoding="utf-8")
    tokenizer = tmp_path / "tok.json"
    tokenizer.write_text('{"model": {}}\n', encoding="utf-8")
    config = tmp_path / "cfg.yaml"
    config.write_text("run: {}\n", encoding="utf-8")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"\x00\x01")
    return {
        "run_id": "run-1", "checkpoint": checkpoint, "seed": 42,
        "release": release, "holdout": holdout, "tokenizer": tokenizer,
        "config": config, "task_set": "code_holdout", "task_set_version": "v2",
    }


def test_complete_run_is_certified(tmp_path):
    record = build_run_record(**full_kwargs(tmp_path))
    assert record["certified"] is True
    assert record["gaps"] == []
    assert record["files"]["checkpoint"]["sha256"] == sha256_file(
        tmp_path / "ckpt.pt"
    )
    assert record["seed"] == 42
    assert record["release"]["dataset_id"] == "d1"


def test_missing_seed_is_a_named_gap(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["seed"] = None
    record = build_run_record(**kwargs)
    assert record["certified"] is False
    assert "seed is not recorded" in record["gaps"]


def test_missing_task_set_and_version_are_gaps(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["task_set"] = None
    kwargs["task_set_version"] = None
    record = build_run_record(**kwargs)
    assert "task_set is not recorded" in record["gaps"]
    assert "task_set_version is not recorded" in record["gaps"]


def test_missing_checkpoint_is_a_gap(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["checkpoint"] = None
    record = build_run_record(**kwargs)
    assert "checkpoint is not recorded" in record["gaps"]


def test_nonexistent_file_is_a_gap(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["checkpoint"] = tmp_path / "absent.pt"
    record = build_run_record(**kwargs)
    assert record["certified"] is False
    assert any("checkpoint file does not exist" in g for g in record["gaps"])


def test_missing_release_record_is_a_gap(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["release"] = tmp_path / "no-such-release"
    record = build_run_record(**kwargs)
    assert record["certified"] is False
    assert any("release record does not exist" in g for g in record["gaps"])


def test_holdout_from_an_unreleased_corpus_is_flagged(tmp_path):
    """A holdout built from data outside the release means the two have drifted."""
    kwargs = full_kwargs(tmp_path)
    manifest = tmp_path / "holdout.manifest.json"
    manifest.write_text(
        json.dumps({"corpus_sha256": "0" * 64}), encoding="utf-8"
    )
    record = build_run_record(**kwargs)
    assert record["certified"] is False
    assert any("not in the recorded release" in g for g in record["gaps"])


def test_matching_holdout_corpus_passes_the_drift_check(tmp_path):
    kwargs = full_kwargs(tmp_path)
    _release, corpus = make_release(tmp_path)
    manifest = tmp_path / "holdout.manifest.json"
    manifest.write_text(
        json.dumps({"corpus_sha256": sha256_file(corpus)}), encoding="utf-8"
    )
    record = build_run_record(**kwargs)
    assert record["certified"] is True


def test_corrupt_release_record_is_reported_not_raised(tmp_path):
    kwargs = full_kwargs(tmp_path)
    (kwargs["release"] / "RELEASE.json").write_text("{not json", encoding="utf-8")
    record = build_run_record(**kwargs)
    assert record["release"]["exists"] is True
    assert record["release"]["error"] == "RELEASE.json is not valid JSON"
    assert record["certified"] is False


def test_certification_message_names_the_gaps(tmp_path):
    kwargs = full_kwargs(tmp_path)
    kwargs["seed"] = None
    record = build_run_record(**kwargs)
    assert record["certification"].startswith("Run is NOT reproducible")
    assert "seed" in record["certification"]