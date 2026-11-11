"""Focused tests for TAKSH L0.7.5 — final pretraining readiness & decision gate."""

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

import evaluation.l075_final_readiness as gate  # noqa: E402
import evaluation.l075_decision_register as reg  # noqa: E402
import evaluation.l074_checkpoint_readiness as ckpt  # noqa: E402
import evaluation.l071_karkain as kark  # noqa: E402
from evaluation.l071_corpus_record import record_fingerprint  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_CFG_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"
L051_SHA = "075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d"
REGISTER = REPO / "configs" / "foundation" / "l075_owner_decision_register.yaml"


# --------------------------------------------------------------------------- #
# A. Owner-decision register
# --------------------------------------------------------------------------- #
def test_register_validates():
    data = reg.load_register()
    assert reg.validate_register(data) == []


def test_register_has_all_required_entries():
    data = reg.load_register()
    ids = {e["id"] for e in data["register"]}
    assert set(reg.REQUIRED_IDS) <= ids
    assert len(data["register"]) == 9


def test_register_every_entry_has_evidence_and_action():
    data = reg.load_register()
    for e in data["register"]:
        assert e["evidence_reference"]
        assert e["owner_action"]
        assert e["status"] in reg.VALID_STATUSES


def test_register_summary_matches_entries():
    data = reg.load_register()
    s = data["decision_summary"]
    status_by_id = {e["id"]: e["status"] for e in data["register"]}
    assert s["missing"] == ["A1.1-training-authorization", "A2-compute-authorization"] or sorted(s["missing"]) == sorted(["A1.1-training-authorization", "A2-compute-authorization"])
    assert sorted(s["granted"]) == ["A5-language-scope"]
    assert "A3.3-commercial-checkpoint-distribution" in s["not_authorized"]
    assert "A3.4-external-publication-permission" in s["not_authorized"]
    for bucket, ids in s.items():
        if bucket == "conformance":
            continue
        for i in ids:
            assert status_by_id[i] in {"GRANTED", "CONDITIONAL", "MISSING",
                                       "HUMAN_LEGAL_REVIEW_REQUIRED", "REJECTED",
                                       "NOT_AUTHORIZED", "OWNER_ACTION_REQUIRED"}


def test_register_fabricates_no_training_authorization():
    data = reg.load_register()
    statuses = {e["id"]: e["status"] for e in data["register"]}
    assert statuses["A1.1-training-authorization"] == "MISSING"
    assert statuses["A1.2-internal-research-permission"] == "CONDITIONAL"
    assert statuses["A2-compute-authorization"] == "MISSING"
    assert statuses["A3.3-commercial-checkpoint-distribution"] == "NOT_AUTHORIZED"
    assert statuses["A3.4-external-publication-permission"] == "NOT_AUTHORIZED"


def test_register_tool_writes_report(tmp_path):
    out = tmp_path / "register.json"
    rc = reg.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["decision_summary"]["missing"]


# --------------------------------------------------------------------------- #
# B. Corpus eligibility reconciliation
# --------------------------------------------------------------------------- #
def test_snapshot_id_and_counts():
    snap = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json").read_text(encoding="utf-8"))
    assert snap["snapshot_id"] == "l072-l04-fdb1fee2beaae9c2"
    assert snap["frozen_for_training"] is False
    eligible = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "eligibility.json").read_text(encoding="utf-8"))
    assert eligible["aggregate"]["training_eligible"] == 0
    assert eligible["aggregate"]["technically_eligible"] == 1050


def test_l072_records_integrity_matches():
    root = REPO / "artifacts" / "l072_l04_freeze"
    snap = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (root / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fps = sorted(record_fingerprint(r) for r in records)
    assert hashlib.sha256("".join(fps).encode()).hexdigest() == snap["integrity_hashes"]["records_sha256"]


def test_technical_eligible_is_not_training_eligible():
    eligible = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "eligibility.json").read_text(encoding="utf-8"))
    assert eligible["aggregate"]["technically_eligible"] >= eligible["aggregate"]["training_eligible"] + 1000


# --------------------------------------------------------------------------- #
# C. Canonical model and checkpoint readiness
# --------------------------------------------------------------------------- #
def test_canonical_checkpoint_spec_still_valid():
    spec = ckpt.load_spec()
    assert ckpt.validate_spec(spec) == []


def test_canonical_111m_missing():
    out = ckpt.evaluate_candidate(None)
    assert out["classification"] == "MISSING_ARTIFACT"
    assert out["accepted"] is False


def test_v180_sft_rejected():
    out = ckpt.evaluate_candidate(REPO / "runs" / "v180_sft" / "checkpoint_last.pt")
    assert out["classification"] == "GEOMETRY_INCOMPATIBLE"


def test_synthetic_fixture_rejected_as_production():
    out = ckpt.evaluate_candidate(None, is_synthetic=True)
    assert out["classification"] == "SYNTHETIC_REJECTED"


def test_l060_foundation_config_intact():
    assert hashlib.sha256((REPO / "configs" / "foundation" / "l060_foundation.yaml").read_bytes()).hexdigest() == FOUNDATION_CFG_SHA


# --------------------------------------------------------------------------- #
# D. Compute resource qualification (documented, not provisionable)
# --------------------------------------------------------------------------- #
def test_resource_report_separates_measured_and_estimated():
    report = json.loads((REPO / "reports" / "l074_resource_strategy.json").read_text(encoding="utf-8"))
    assert report["measurements"]["host"]["labels"] == "measured"
    assert "estimates_labelled" in report


def test_mx150_recorded_unusable():
    report = json.loads((REPO / "reports" / "l074_resource_strategy.json").read_text(encoding="utf-8"))
    gpu = report["measurements"]["gpu"]
    assert gpu["model"] == "NVIDIA GeForce MX150"
    assert gpu["torch_cuda_available"] is False


# --------------------------------------------------------------------------- #
# E. CI and Python runtime closure
# --------------------------------------------------------------------------- #
def test_python_runtime_remains_not_executed():
    out = gate.evaluate()
    dim = out["dimensions"]["python_runtime_compatibility"]
    assert dim["status"] == "CONDITIONAL"
    assert "NOT EXECUTED" in dim["evidence"] or "interpreter installed" in dim["evidence"].lower()


def test_ci_blocked_with_exact_probe_evidence():
    out = gate.evaluate()
    dim = out["dimensions"]["full_github_actions_ci_execution"]
    assert dim["status"] == "BLOCKED"
    assert "HTTP" in json.dumps(dim) or "404" in json.dumps(dim) or "422" in json.dumps(dim)


# --------------------------------------------------------------------------- #
# F. Final readiness gate
# --------------------------------------------------------------------------- #
def test_final_gate_dimension_set():
    out = gate.evaluate()
    assert out["dimension_count"] == 18
    assert set(out["dimensions"]) == {
        "corpus_snapshot_integrity", "data_provenance", "technical_corpus_eligibility",
        "legal_training_permission", "owner_training_authorization", "multilingual_language_coverage",
        "tokenizer_model_compatibility", "canonical_checkpoint_availability",
        "hardware_resource_feasibility", "benchmark_thresholds",
        "focused_regression_qualification", "full_github_actions_ci_execution",
        "python_runtime_compatibility", "reproducibility_and_recovery",
        "training_artifact_provenance", "training_environment_authorization",
        "external_checkpoint_distribution_permission", "karkain_readiness_and_integration",
    }


def test_final_gate_verdict_blocked_and_zero_training_tokens():
    out = gate.evaluate()
    assert out["verdict"] == "BLOCKED"
    assert out["training_eligible_tokens"] == 0
    assert out["mandatory_fact_check"]["python_samples"] == 1050
    assert out["mandatory_fact_check"]["python_tokens"] == 1_380_611
    assert out["mandatory_fact_check"]["training_eligible_tokens"] == 0


def test_every_dimension_has_responsibility_and_dates():
    out = gate.evaluate()
    for name, d in out["dimensions"].items():
        assert d["last_verified"] == "2026-11-11", name
        assert d["responsible_party"] in ("owner_decision", "engineering"), name
        assert "blocking_dependency" in d and d["blocking_dependency"], name


def test_owner_decision_dimensions_are_visible():
    out = gate.evaluate()
    owner_blocked = [k for k, d in out["dimensions"].items() if d["owner_decision_required"]]
    assert set(owner_blocked) >= {
        "legal_training_permission", "owner_training_authorization",
        "hardware_resource_feasibility", "full_github_actions_ci_execution",
        "python_runtime_compatibility", "training_artifact_provenance",
        "training_environment_authorization", "external_checkpoint_distribution_permission",
        "karkain_readiness_and_integration",
    }


def test_final_gate_blocking_set():
    out = gate.evaluate()
    assert set(out["blocking_dimensions"]) >= {
        "owner_training_authorization", "canonical_checkpoint_availability",
        "hardware_resource_feasibility", "full_github_actions_ci_execution",
        "training_artifact_provenance", "training_environment_authorization",
        "external_checkpoint_distribution_permission", "karkain_readiness_and_integration",
    }


def test_final_gate_writes_report(tmp_path):
    out = tmp_path / "r.json"
    rc = gate.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "BLOCKED"
    assert report["invariants"]


# --------------------------------------------------------------------------- #
# Karkain and L0.8.0 contract preservation
# --------------------------------------------------------------------------- #
def test_karkain_remains_discovered():
    assert kark.validate() == []
    strategy = yaml.safe_load((REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"
    assert strategy["identity"]["spec_version_pinned"] == "1.1.0"


def test_entry_contract_exists_and_has_mandatory_sections():
    text = (REPO / "docs" / "L075_L080_ENTRY_CONTRACT.md").read_text(encoding="utf-8")
    for section in ("## 1. Data", "## 2. Model", "## 3. Environment", "## 4. Training", "## 5. Evaluation", "## 6. Prohibitions", "## 7. Go / no-go"):
        assert section in text
    assert "must not begin" in text
    assert "check any of these boxes" in text or "no single box" in text or "checkable preconditions" in text


def test_roadmap_ambition_preserved():
    foundation = (REPO / "configs" / "foundation" / "l060_foundation.yaml").read_text(encoding="utf-8")
    assert "taksh-foundation-111m" in foundation
    import yaml as _y

    data = _y.safe_load((REPO / "configs" / "feasibility" / "l051_languages.yaml").read_text(encoding="utf-8"))
    assert len(data["languages"]) == 53


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_v180_sft_preserved():
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


def test_l051_registry_preserved():
    assert hashlib.sha256((REPO / "configs" / "feasibility" / "l051_languages.yaml").read_bytes()).hexdigest() == L051_SHA


def test_train_corpus_preserved():
    assert hashlib.sha256((REPO / "data" / "l04_code_corpus" / "train.jsonl").read_bytes()).hexdigest() == \
        "cf89b8bc803b7363d2643e337a561524644e4924318c90332399f1625dfa1d5f"
