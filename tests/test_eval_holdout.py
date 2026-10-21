"""Tests for held-out evaluation data preparation (M2.10)."""

from __future__ import annotations

import json

import pytest

from evaluation.holdout import (
    build_holdout,
    check_separation,
    load_baseline_loss,
    load_records,
)


def record(split, repo, sha, text=None, language="python"):
    return {
        "text": text or ("def f():\n    return 1\n" * 20),
        "content_sha1": sha,
        "repo": repo,
        "commit": "a" * 40,
        "file_path": f"{repo.replace('/', '_')}.py",
        "language": language,
        "spdx": "MIT",
        "attribution": "test",
        "retrieved_at": "2026-10-14",
        "split": split,
    }


def write_corpus(tmp_path, records, name="corpus.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# Separation
# --------------------------------------------------------------------------- #
def test_separation_passes_for_disjoint_repositories():
    records = [
        record("train", "org/a", "sha-a1"),
        record("train", "org/a", "sha-a2"),
        record("test", "org/b", "sha-b1"),
    ]
    report = check_separation(records)
    assert report["ok"] is True
    assert report["shared_repositories"] == []
    assert report["shared_content_sha1"] == []


def test_separation_detects_repository_spanning_train_and_test():
    records = [
        record("train", "org/a", "sha-a1"),
        record("test", "org/a", "sha-b1"),
    ]
    report = check_separation(records)
    assert report["ok"] is False
    assert report["shared_repositories"] == ["org/a"]
    assert "span both" in report["problem"]


def test_separation_detects_content_hash_collision_across_splits():
    records = [
        record("train", "org/a", "same-sha"),
        record("test", "org/b", "same-sha"),
    ]
    report = check_separation(records)
    assert report["ok"] is False
    assert report["shared_content_sha1"] == ["same-sha"]
    assert "content hash" in report["problem"]


def test_build_refuses_a_leaking_holdout(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/a", "sha-b1"),
    ])
    with pytest.raises(ValueError, match="not actually held out"):
        build_holdout(corpus, tmp_path / "out.jsonl")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def test_build_writes_items_with_provenance(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1"),
        record("test", "org/c", "sha-c1", language="go"),
    ])
    out = tmp_path / "holdout.jsonl"
    manifest = build_holdout(corpus, out, now="2026-10-18T00:00:00+00:00")

    assert manifest["items"] == 2
    items = load_records(out)
    assert [i["task_id"] for i in items] == ["test/0000", "test/0001"]
    first = items[0]
    assert first["source"]["repo"] == "org/b"
    assert first["source"]["commit"] == "a" * 40
    assert first["source"]["license"] == "MIT"
    assert first["content_sha1"] == "sha-b1"
    assert first["task_type"] == "held_out_language_modelling"
    assert manifest["languages"] == {"python": 1, "go": 1}
    assert manifest["separation"]["ok"] is True


def test_build_skips_short_items(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1", text="x = 1\n"),
        record("test", "org/c", "sha-c1"),
    ])
    manifest = build_holdout(corpus, tmp_path / "holdout.jsonl", min_chars=200)
    assert manifest["items"] == 1
    assert manifest["skipped_short"] == 1


def test_build_fails_when_no_usable_items(tmp_path):
    corpus = write_corpus(tmp_path, [record("train", "org/a", "sha-a1")])
    with pytest.raises(ValueError, match="no held-out records"):
        build_holdout(corpus, tmp_path / "holdout.jsonl")


def test_manifest_never_claims_an_executable_benchmark(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1"),
    ])
    manifest = build_holdout(corpus, tmp_path / "holdout.jsonl")
    protocol = manifest["evaluation_protocol"]
    assert protocol["executable_benchmark"] is False
    assert protocol["published_suites_status"] == "unavailable"
    assert any("not a pass@1" in i.lower() for i in protocol["limitations"])
    assert any("164" in i and "500" in i for i in protocol["limitations"])


def test_baseline_is_recorded_when_vocab_known(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1"),
    ])
    manifest = build_holdout(corpus, tmp_path / "holdout.jsonl", vocab_size=5822)
    baseline = manifest["evaluation_protocol"]["baseline"]
    assert baseline["vocab_size"] == 5822
    assert baseline["uniform_random_loss"] == pytest.approx(load_baseline_loss(5822))


def test_baseline_is_optional(tmp_path):
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1"),
    ])
    manifest = build_holdout(corpus, tmp_path / "holdout.jsonl")
    assert manifest["evaluation_protocol"]["baseline"]["vocab_size"] is None


def test_manifest_pins_the_corpus_checksum(tmp_path):
    from data.release import sha256_file
    corpus = write_corpus(tmp_path, [
        record("train", "org/a", "sha-a1"),
        record("test", "org/b", "sha-b1"),
    ])
    manifest = build_holdout(corpus, tmp_path / "holdout.jsonl")
    assert manifest["corpus_sha256"] == sha256_file(corpus)


def test_shipped_holdout_manifest_is_clean():
    """The committed holdout artifact must still verify."""
    from pathlib import Path
    manifest = json.loads(
        Path("evaluation/code_holdout_v2.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["separation"]["ok"] is True
    assert manifest["items"] > 0
    assert manifest["evaluation_protocol"]["executable_benchmark"] is False