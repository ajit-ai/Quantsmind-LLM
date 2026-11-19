"""Tests for M6 candidate inventory.

Discovery must be honest about what is on disk: partial downloads, missing
shards and an empty snapshot directory are all distinct failures, and none of
them may be reported as a usable model.
"""

from __future__ import annotations

import json

from evaluation.m6_inventory import (ALIASES, build_candidate,
                                     build_inventory, discover_hf_candidates,
                                     inspect_snapshot)

MIB = 1024 ** 2


def make_snapshot(root, model_id, revision="abc123", files=None):
    repo = root / ("models--" + model_id.replace("/", "--"))
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, content in (files or {}).items():
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    return snapshot


def test_complete_snapshot_is_detected(tmp_path):
    snapshot = make_snapshot(tmp_path, "org/model", files={
        "config.json": "{}",
        "model.safetensors": b"x" * 100,
        "tokenizer.json": "{}",
    })
    detail = inspect_snapshot(snapshot)
    assert detail["complete"] is True
    assert detail["has_config"] and detail["has_tokenizer"]
    assert detail["artifact_bytes"] == 100


def test_partial_download_is_not_complete(tmp_path):
    snapshot = make_snapshot(tmp_path, "org/model", files={
        "config.json": "{}",
        "model-00001-of-00002.safetensors.incomplete": b"x" * 50,
    })
    detail = inspect_snapshot(snapshot)
    assert detail["complete"] is False
    assert detail["incomplete_files"] == [
        "model-00001-of-00002.safetensors.incomplete"]


def test_missing_shard_from_index_is_not_complete(tmp_path):
    index = json.dumps({"weight_map": {"a": "model-1.safetensors",
                                       "b": "model-2.safetensors"},
                        "metadata": {"total_size": 200}})
    snapshot = make_snapshot(tmp_path, "org/model", files={
        "config.json": "{}",
        "model.safetensors.index.json": index,
        "model-1.safetensors": b"x" * 100,
    })
    detail = inspect_snapshot(snapshot)
    assert detail["complete"] is False
    assert detail["missing_shards"] == ["model-2.safetensors"]
    assert detail["declared_total_size"] == 200


def test_snapshot_without_weights_is_not_complete(tmp_path):
    snapshot = make_snapshot(tmp_path, "org/model", files={"config.json": "{}"})
    assert inspect_snapshot(snapshot)["complete"] is False


def test_repository_without_snapshot_is_reported(tmp_path):
    repo = tmp_path / "models--org--empty"
    repo.mkdir()
    (repo / "snapshots").mkdir()
    found = discover_hf_candidates(tmp_path)
    assert found[0]["complete"] is False
    assert "no snapshot" in found[0]["reason"]


def test_discovery_uses_latest_snapshot_and_ref(tmp_path):
    make_snapshot(tmp_path, "org/model", revision="old", files={
        "config.json": "{}", "model.safetensors": b"x"})
    make_snapshot(tmp_path, "org/model", revision="new", files={
        "config.json": "{}", "model.safetensors": b"xy"})
    ref = tmp_path / "models--org--model" / "refs"
    ref.mkdir()
    (ref / "main").write_text("new\n", encoding="utf-8")
    found = discover_hf_candidates(tmp_path)
    assert found[0]["revision"] == "new"
    assert found[0]["revision_from_ref"] == "new"


def test_alias_maps_legacy_cache_name():
    assert ALIASES["bert-base-uncased"] == "google-bert/bert-base-uncased"


def test_absent_candidate_is_not_present_not_blocked_zero(tmp_path):
    cand = build_candidate("Qwen/Qwen2.5-Coder-0.5B-Instruct", None,
                           runtime_ok=True, runtime_reason="ok")
    assert cand.acquisition_status == "NOT_PRESENT"
    assert cand.evaluation_status == "NOT_PRESENT"
    assert "explicit authorisation" in cand.blocking_reason


def test_unverified_licence_blocks_even_a_complete_snapshot(tmp_path):
    snapshot = make_snapshot(tmp_path, "org/unknown", files={
        "config.json": "{}", "model.safetensors": b"x"})
    discovery = inspect_snapshot(snapshot)
    discovery.update({"revision": "abc"})
    cand = build_candidate("org/unknown", discovery,
                           runtime_ok=True, runtime_reason="ok")
    assert cand.evaluation_status == "BLOCKED"
    assert "no verified licence" in cand.blocking_reason


def test_incomplete_snapshot_reason_names_the_problem(tmp_path):
    snapshot = make_snapshot(tmp_path, "org/model", files={
        "config.json": "{}",
        "model.safetensors.incomplete": b"x"})
    discovery = inspect_snapshot(snapshot)
    cand = build_candidate("microsoft/Phi-3-mini-4k-instruct", discovery,
                           runtime_ok=True, runtime_reason="ok")
    assert cand.acquisition_status == "PARTIAL"
    assert "partial downloads" in cand.blocking_reason


def test_inventory_never_marks_absent_candidate_evaluable(tmp_path):
    report = build_inventory(tmp_path, allow_download=False)
    assert report["download_authorised"] is False
    assert report["candidates"]
    assert all(not c["local_available"] for c in report["candidates"]
               if c["acquisition_status"] == "NOT_PRESENT")
    assert all(c["evaluation_status"] != "NOT_EVALUATED"
               for c in report["candidates"]
               if c["acquisition_status"] == "NOT_PRESENT")
