"""Focused tests for TAKSH L0.7.4 — CI execution, runtime compatibility & resource strategy."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import evaluation.l074_readiness as readiness  # noqa: E402
import evaluation.l074_checkpoint_readiness as ckpt  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"

WORKFLOW = REPO / ".github" / "workflows" / "ci-l0704-manual-qualification.yml"
STRATEGY = REPO / "reports" / "l074_resource_strategy.json"


# --------------------------------------------------------------------------- #
# A. CI workflow definition checks (execution blocked; definition itself pinned)
# --------------------------------------------------------------------------- #
def test_manual_ci_workflow_exists_and_parses():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert data["name"].startswith("CI (L0.7.4")
    assert set(data["jobs"]) >= {"qualify"}


def test_manual_ci_workflow_is_read_only_and_test_only():
    text = WORKFLOW.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert data.get("permissions") == {"contents": "read"}
    code_lines = "\n".join(
        line.split("#", 1)[0] for line in text.splitlines()
    ).lower()
    for banned in ("pypi", "twine", "publish", "upload-artifact",
                   "huggingface", "secrets.", "id-token", "release"):
        assert banned not in code_lines, f"workflow must not reference {banned}"


def test_manual_ci_workflow_trigger_is_dispatch_only():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = data["on"] if "on" in data else data[True]
    assert set(triggers) == {"workflow_dispatch"}
    assert "branches" not in triggers.get("workflow_dispatch", {})


def test_manual_ci_workflow_does_not_trigger_on_push_to_main():
    data = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = data["on"] if "on" in data else data[True]
    assert "push" not in triggers and "pull_request" not in triggers


def test_production_workflow_unchanged_and_active():
    ci = REPO / ".github" / "workflows" / "ci.yml"
    text = ci.read_text(encoding="utf-8")
    assert "workflow_dispatch" not in text
    assert "branches: [develop, main]" in text.replace("  ", " ")


# --------------------------------------------------------------------------- #
# B. Python 3.11 static compatibility evidence
# --------------------------------------------------------------------------- #
def test_requires_python_declares_support():
    import re

    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'requires-python\s*=\s*"([^"]+)"', text)
    assert m and m.group(1) == ">=3.10"


def test_real_code_has_no_syntax_beyond_311():
    import ast

    roots = ["taksh", "training", "serving", "data", "evaluation", "tests"]
    allowed_exceptions = {str(REPO / "scripts" / "eval_v210_agent.py").replace("\\", "/")}
    bad = []
    for root in roots:
        for p in Path(REPO / root).rglob("*.py"):
            src = p.read_text(encoding="utf-8-sig")
            try:
                ast.parse(src, filename=str(p), feature_version=(3, 11))
            except SyntaxError as exc:
                bad.append((str(p), exc.lineno, exc.msg))
    assert bad == [], f"syntax newer than 3.11 in real code: {bad}"


# --------------------------------------------------------------------------- #
# C. Resource strategy report
# --------------------------------------------------------------------------- #
def test_resource_strategy_report_shape():
    report = json.loads(STRATEGY.read_text(encoding="utf-8"))
    assert report["schema"] == "taksh.resource_strategy/1"
    for s in ("A_local_cpu_host", "B_memory_efficient_cpu",
              "C_owner_provided_gpu", "D_external_compute_owner_authorized"):
        assert s in report["strategies"]


def test_resource_strategy_measurements_vs_estimates_separated():
    report = json.loads(STRATEGY.read_text(encoding="utf-8"))
    assert "measured" in report["measured_vs_estimated"]
    assert "estimated" in report["measured_vs_estimated"]
    assert report["measurements"]["host"]["labels"] == "measured"


def test_resource_strategy_records_mx150_truthfully():
    report = json.loads(STRATEGY.read_text(encoding="utf-8"))
    gpu = report["measurements"]["gpu"]
    assert gpu["present"] is True
    assert gpu["model"] == "NVIDIA GeForce MX150"
    assert gpu["torch_cuda_available"] is False


def test_resource_strategy_recommends_min_viable_and_blocking():
    report = json.loads(STRATEGY.read_text(encoding="utf-8"))
    rec = report["recommendation"]
    assert "16 GiB" in rec["minimum_viable_environment"]
    assert rec["resource_strategy_availability"].startswith("BLOCKED")


def test_resource_strategy_invents_no_prices():
    import re

    text = STRATEGY.read_text(encoding="utf-8")
    price_patterns = [
        re.compile(r"\$\s*\d", re.IGNORECASE),
        re.compile(r"\b\d+(\.\d+)?\s*(USD|dollars?|eur|gbp)\b", re.IGNORECASE),
        re.compile(r"\d+(\.\d+)?\s*per\s+(hour|month|second|gpu)\b", re.IGNORECASE),
    ]
    for pat in price_patterns:
        assert not pat.search(text), f"strategy must not contain price pattern {pat.pattern}"


# --------------------------------------------------------------------------- #
# D. Canonical checkpoint readiness
# --------------------------------------------------------------------------- #
def test_canonical_checkpoint_spec_validates():
    spec = ckpt.load_spec()
    assert ckpt.validate_spec(spec) == []
    assert spec["canonical_model"]["parameters"] == 110_658_816


def test_canonical_checkpoint_missing_artifact_rejected():
    out = ckpt.evaluate_candidate(None)
    assert out["accepted"] is False
    assert out["classification"] == "MISSING_ARTIFACT"


def test_v180_sft_rejected_as_geometry_incompatible(tmp_path):
    out = ckpt.evaluate_candidate(REPO / "runs" / "v180_sft" / "checkpoint_last.pt")
    assert out["accepted"] is False
    assert out["classification"] == "GEOMETRY_INCOMPATIBLE"
    assert any("d_model: resolved=384" in p for p in out["reason"].split("; "))


def test_spec_requires_authorization_gate():
    spec = ckpt.load_spec()
    assert spec["provenance_requirements"]["owner_authorization_gate"]
    assert "owner authorization record id" in str(spec["artifact_requirements"]["dataset_eligibility_record"]).lower() or \
           "never fabricated" in str(spec["artifact_requirements"]["dataset_eligibility_record"]).lower()


def test_checkpoint_readiness_report_fails_closed(tmp_path):
    out = tmp_path / "readiness.json"
    rc = ckpt.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["spec_valid"] is True
    assert report["candidates"]["canonical_111m_artifact"]["classification"] == "MISSING_ARTIFACT"
    assert report["candidates"]["v180_sft_rejected"]["classification"] == "GEOMETRY_INCOMPATIBLE"
    assert report["candidates"]["synthetic_fixture_rejected"]["classification"] == "SYNTHETIC_REJECTED"


# --------------------------------------------------------------------------- #
# E. Extended readiness reconciliation
# --------------------------------------------------------------------------- #
def test_extended_readiness_dimension_set_and_verdict(tmp_path):
    out = tmp_path / "readiness.json"
    rc = readiness.main(["--out", str(out)])
    assert rc == 0
    full = json.loads(out.read_text(encoding="utf-8"))
    assert full["dimension_count"] == 18
    assert full["verdict"] == "BLOCKED"
    blocking = set(full["blocking_dimensions"])
    assert "actual_training_authorization" in blocking
    assert "checkpoint_availability_and_identity" in blocking
    assert blocking >= {"full_github_actions_ci_execution", "canonical_checkpoint_readiness",
                        "resource_strategy_availability"}


def test_extended_readiness_invariants_present():
    full = json.loads((REPO / "reports" / "l074_training_readiness.json").read_text(encoding="utf-8"))
    inv = " ".join(full["invariants"])
    assert "focused passing suite is never reported as a full CI pass" in inv
    assert "resource strategy document never implies compute availability" in inv
    assert "conditional internal research permission never implies owner authorization" in inv


def test_extended_readiness_training_eligible_tokens_zero():
    full = json.loads((REPO / "reports" / "l074_training_readiness.json").read_text(encoding="utf-8"))
    assert full["training_eligible_tokens"] == 0
    assert full["dimensions"]["actual_training_authorization"]["status"] == "MISSING"
    assert full["summary"]["owner_authorized_corpus"] is False


def test_extended_readiness_ci_blocked_with_reason():
    full = json.loads((REPO / "reports" / "l074_training_readiness.json").read_text(encoding="utf-8"))
    dim = full["dimensions"]["full_github_actions_ci_execution"]
    assert dim["status"] == "BLOCKED"
    assert "workflow_dispatch" in json.dumps(dim)
    assert "HTTP 422" in json.dumps(dim)


def test_extended_readiness_python311_is_conditional_not_qualified():
    full = json.loads((REPO / "reports" / "l074_training_readiness.json").read_text(encoding="utf-8"))
    dim = full["dimensions"]["python_311_compatibility"]
    assert dim["status"] == "CONDITIONAL"
    assert dim["details"]["runtime_executed"] is False


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_l01_checkpoint_preserved():
    digest = hashlib.sha256()
    with (REPO / "runs" / "v180_sft" / "checkpoint_last.pt").open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA
