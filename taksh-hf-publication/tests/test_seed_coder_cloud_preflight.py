"""Focused tests for D3.1 cloud preflight and workflow guards."""

from __future__ import annotations

import hashlib
import json

from scripts.preflight_seed_coder_cloud import preflight, validate_artifact_hashes
from scripts.run_seed_coder_cloud_workflow import _plan


def _model_dir(tmp_path):
    path = tmp_path / "model"
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"architectures": ["LlamaForCausalLM"], "model_type": "llama"}),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text('{"version":"1.0"}', encoding="utf-8")
    return path


def test_artifact_hash_validation_passes(tmp_path):
    path = _model_dir(tmp_path)
    config_hash = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
    tokenizer_hash = hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()
    checks = validate_artifact_hashes(
        path,
        expected_config_sha256=config_hash,
        expected_tokenizer_sha256=tokenizer_hash,
    )
    assert all(item["ok"] for item in checks)


def test_artifact_hash_validation_detects_mismatch(tmp_path):
    path = _model_dir(tmp_path)
    checks = validate_artifact_hashes(path, expected_tokenizer_sha256="0" * 64)
    assert not all(item["ok"] for item in checks)


def test_preflight_reports_model_revision_and_requirements():
    report = preflight(model_path=None, min_vram_gib=16.0, min_ram_gib=16.0, min_disk_gib=0.0)
    assert report["model"]["revision"] == "1c5af72afa21c22c9272a0b60a3a404195e88fcd"
    assert report["ready"] is False
    assert any(item["name"] == "model_artifact_directory" and not item["ok"] for item in report["checks"])


def test_workflow_plan_does_not_execute():
    import argparse

    args = argparse.Namespace(
        model_path=None,
        suite="suite.json",
        expected_sha="abc123",
        out_dir="out",
        execute=False,
        max_new_tokens=64,
        context_length=8192,
        batch_size=1,
        timeout=600.0,
        device="cpu",
        dtype="auto",
    )
    plan = _plan(args)
    assert plan["mode"] == "plan_only"
    assert plan["resource_controls"]["batch_size"] == 1
    assert plan["execution"]["automatic_downloads"] is False
    assert plan["execution"]["provider_fallback"] is False
