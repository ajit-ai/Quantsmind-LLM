"""Tests for dataset release artifacts (M2.8)."""

from __future__ import annotations

import json

import pytest

from data.release import (
    CHECKSUM_FILE,
    RECORD_FILE,
    build_release,
    verify_release,
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return path


def make_corpus(tmp_path, name="corpus.jsonl", count=3):
    records = [
        {
            "schema_version": "1",
            "kind": "corpus",
            "id": f"r{i}",
            "content": {"text": f"line {i}\n"},
            "source": {"repo": "https://example.invalid/x", "commit": "0" * 40,
                       "path": f"f{i}.py", "license": "MIT"},
            "tags": [],
        }
        for i in range(count)
    ]
    return write_jsonl(tmp_path / name, records)


def make_manifest(tmp_path, name="corpus.manifest.json", status="complete",
                  failures=None):
    manifest = {
        "corpus": {"version": "v2", "id": "code_corpus-v2"},
        "status": status,
        "complete": status == "complete",
        "counts": {"records": 3},
        "splits": {"train": 1},
        "licensing": {"policy_version": "1", "by_spdx": {"MIT": 3}},
        "integrity_failures": failures or [],
        "content_hashes": [{"content_hash": "abc"}],
        "leakage_check": {"split_overlap": 0, "limitations": ["x"]},
    }
    path = tmp_path / name
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def test_build_release_produces_verifiable_set(tmp_path):
    corpus = make_corpus(tmp_path)
    manifest = make_manifest(tmp_path)
    record = build_release(
        corpus, tmp_path / "rel", "code_corpus-v2", manifest_path=manifest
    )

    out = tmp_path / "rel"
    assert (out / RECORD_FILE).exists()
    assert (out / CHECKSUM_FILE).exists()
    assert (out / "corpus.jsonl").exists()
    assert record["dataset_id"] == "code_corpus-v2"
    assert record["contents"]["records"] == 3
    assert record["version"] == "v2"
    # Provenance is carried forward, not dropped.
    assert record["provenance"]["status"] == "complete"
    assert record["provenance"]["licensing"]["by_spdx"] == {"MIT": 3}
    assert verify_release(out)["ok"] is True


def test_release_is_not_claimed_as_published(tmp_path):
    corpus = make_corpus(tmp_path)
    record = build_release(corpus, tmp_path / "rel", "d1")
    # A local verified set must never be conflated with a publication.
    assert record["released"] is False
    assert "NOT a publication" in record["release_note"]


def test_build_release_requires_existing_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_release(tmp_path / "absent.jsonl", tmp_path / "rel", "d1")


# --------------------------------------------------------------------------- #
# Verify: drift, deletion, tampering
# --------------------------------------------------------------------------- #
def test_verify_detects_tampered_shard(tmp_path):
    corpus = make_corpus(tmp_path)
    build_release(corpus, tmp_path / "rel", "d1")
    (tmp_path / "rel" / "corpus.jsonl").write_text(
        '{"kind": "corpus"}\n', encoding="utf-8"
    )
    report = verify_release(tmp_path / "rel")
    assert report["ok"] is False
    assert report["checks"]["checksums_match"] is False
    assert any("checksum mismatch" in p for p in report["problems"])


def test_verify_detects_deleted_file(tmp_path):
    corpus = make_corpus(tmp_path)
    build_release(corpus, tmp_path / "rel", "d1")
    (tmp_path / "rel" / "corpus.jsonl").unlink()
    report = verify_release(tmp_path / "rel")
    assert report["ok"] is False
    assert report["checks"]["files_present"] is False


def test_verify_detects_record_count_drift(tmp_path):
    corpus = make_corpus(tmp_path, count=3)
    build_release(corpus, tmp_path / "rel", "d1")
    # Rewrite the shard AND its checksum so only the claimed count betrays it.
    shard = tmp_path / "rel" / "corpus.jsonl"
    write_jsonl(shard, [
        {"kind": "corpus", "content": {"text": "x"}} for _ in range(2)
    ])
    record = json.loads((tmp_path / "rel" / RECORD_FILE).read_text(encoding="utf-8"))
    from data.release import sha256_file
    record["files"]["corpus.jsonl"]["sha256"] = sha256_file(shard)
    (tmp_path / "rel" / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")
    from data.release import build_release as _br  # noqa: F401
    checksum = (tmp_path / "rel" / CHECKSUM_FILE)
    checksum.write_text(
        f"{sha256_file(shard)}  corpus.jsonl\n", encoding="utf-8"
    )
    report = verify_release(tmp_path / "rel")
    assert report["checks"]["record_count_matches"] is False
    assert any("record claims" in p for p in report["problems"])


def test_verify_detects_incomplete_manifest(tmp_path):
    corpus = make_corpus(tmp_path)
    manifest = make_manifest(tmp_path, status="partial")
    build_release(corpus, tmp_path / "rel", "d1", manifest_path=manifest)
    report = verify_release(tmp_path / "rel")
    assert report["ok"] is False
    assert report["checks"]["corpus_status_complete"] is False


def test_verify_detects_recorded_integrity_failures(tmp_path):
    corpus = make_corpus(tmp_path)
    manifest = make_manifest(tmp_path, failures=["duplicate content hash"])
    build_release(corpus, tmp_path / "rel", "d1", manifest_path=manifest)
    report = verify_release(tmp_path / "rel")
    assert report["ok"] is False
    assert report["checks"]["no_integrity_failures"] is False


def test_verify_detects_checksum_file_disagreement(tmp_path):
    corpus = make_corpus(tmp_path)
    build_release(corpus, tmp_path / "rel", "d1")
    (tmp_path / "rel" / CHECKSUM_FILE).write_text(
        "0" * 64 + "  corpus.jsonl\n", encoding="utf-8"
    )
    report = verify_release(tmp_path / "rel")
    assert report["checks"]["checksum_file_agrees"] is False
    assert report["ok"] is False


def test_verify_missing_release_is_not_ok(tmp_path):
    report = verify_release(tmp_path / "nothing-here")
    assert report["exists"] is False
    assert report["ok"] is False
    assert report["problems"]


# --------------------------------------------------------------------------- #
# Token artifact chain: JSONL -> tokens must be provably the same data
# --------------------------------------------------------------------------- #
def make_token_pair(tmp_path, corpus_sha, num_tokens=17):
    import hashlib
    bin_path = tmp_path / "corpus.tokens.bin"
    bin_path.write_bytes(b"\x01\x02" * num_tokens)
    meta_path = tmp_path / "corpus.tokens.meta.json"
    meta_path.write_text(
        json.dumps({
            "schema": "quantsmind.token_stream/1",
            "corpus_sha256": corpus_sha,
            "num_tokens": num_tokens,
            "num_documents": 3,
            "dataset_schema_validated": True,
            "vocab_size": 5822,
        }),
        encoding="utf-8",
    )
    return bin_path, meta_path


def test_token_chain_links_artifact_to_released_shard(tmp_path):
    corpus = make_corpus(tmp_path)
    sha = hashlib_sha256(corpus)
    bin_path, meta_path = make_token_pair(tmp_path, sha)
    record = build_release(
        corpus, tmp_path / "rel", "d1",
        extra_paths=[bin_path, meta_path],
    )
    assert record["token_chain"]["checked"] is True
    assert record["token_chain"]["matches"] is True
    assert record["token_chain"]["num_tokens"] == 17
    report = verify_release(tmp_path / "rel")
    assert report["checks"]["token_chain_matches_shard"] is True
    assert report["checks"]["token_artifact_schema_validated"] is True
    assert report["ok"] is True


def test_token_artifact_from_a_different_corpus_is_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    # A token artifact encoded from some *other* corpus is not part of this
    # release, however well-formed it is.
    bin_path, meta_path = make_token_pair(tmp_path, "0" * 64)
    build_release(corpus, tmp_path / "rel", "d1", extra_paths=[bin_path, meta_path])
    report = verify_release(tmp_path / "rel")
    assert report["ok"] is False
    assert report["checks"]["token_chain_matches_shard"] is False
    assert any("different corpus" in p for p in report["problems"])


def test_token_artifact_built_without_schema_validation_is_flagged(tmp_path):
    corpus = make_corpus(tmp_path)
    bin_path, meta_path = make_token_pair(tmp_path, hashlib_sha256(corpus))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["dataset_schema_validated"] = False
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    build_release(corpus, tmp_path / "rel", "d1", extra_paths=[bin_path, meta_path])
    report = verify_release(tmp_path / "rel")
    assert report["checks"]["token_artifact_schema_validated"] is False
    assert report["ok"] is False


def test_build_release_rejects_missing_extra(tmp_path):
    corpus = make_corpus(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_release(corpus, tmp_path / "rel", "d1",
                      extra_paths=[tmp_path / "absent.bin"])


def hashlib_sha256(path):
    from data.release import sha256_file
    return sha256_file(path)