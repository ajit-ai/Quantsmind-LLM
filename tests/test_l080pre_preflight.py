"""Focused tests for TAKSH L0.8.0-PRE — Foundation Training Preflight (NO TRAINING)."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import evaluation.l080pre.model_identity as model_id  # noqa: E402
import evaluation.l080pre.tokenizer_integrity as tok  # noqa: E402
import evaluation.l080pre.corpus_verify as corpus  # noqa: E402
import evaluation.l080pre.training_config as trcfg  # noqa: E402
import evaluation.l080pre.resource_estimate as rest  # noqa: E402
import evaluation.l080pre.checkpoint_recovery as crec  # noqa: E402
import evaluation.l080pre.benchmark_safeguard as bench  # noqa: E402
import evaluation.l080pre.gate as gate  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
FOUNDATION_CFG_SHA = "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"


# --------------------------------------------------------------------------- #
# 2. Owner decision record — A1.1-C and A2-E only
# --------------------------------------------------------------------------- #
def test_owner_record_records_explicit_decisions_only():
    text = (REPO / "docs" / "L075_L080_OWNER_DECISION_RECORD.md").read_text(encoding="utf-8")
    assert "- [x] **A1.1-C" in text
    assert "- [x] **A2-E" in text
    assert "- [ ] **A1.1-A" in text
    assert "- [ ] **A1.1-B" in text
    assert "- [ ] **A2-A" in text
    assert "- [ ] **A2-B" in text
    assert "- [ ] **A2-C" in text
    assert "- [ ] **A2-D" in text
    assert "2026-11-11" in text
    assert "TAKSH-OWNER-DECISION-2026-11-11" in text


def test_owner_record_files_explicit_restrictions():
    text = (REPO / "docs" / "L075_L080_OWNER_DECISION_RECORD.md").read_text(encoding="utf-8")
    for restriction in (
        "No model training",
        "No external dataset acquisition",
        "No GPU rental",
        "No paid APIs",
        "No publication",
        "No tags, releases",
    ):
        assert restriction in text
    assert "TAKSH-OWNER-DECISION-2026-11-11" in text


def test_owner_record_no_box_checked_other_than_a1_1_c_and_a2_e():
    text = (REPO / "docs" / "L075_L080_OWNER_DECISION_RECORD.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if "- [x]" in line:
            assert "A1.1-C" in line or "A2-E" in line, f"unexpected checked box: {line}"


# --------------------------------------------------------------------------- #
# A. Model identity
# --------------------------------------------------------------------------- #
def test_canonical_model_identity():
    result = model_id.verify()
    assert result["architecture_verified"] is True
    assert result["derived_parameters"] == 110_658_816
    assert result["config_sha256"] == FOUNDATION_CFG_SHA
    assert result["v180_sft_rejected_as_foundation"] is True
    assert result["canonical_checkpoint_artifact"] == "MISSING"


def test_canonical_config_parameter_derivation_exact():
    cfg = model_id.load_config()["model"]
    assert model_id.expected_param_count(cfg) == 110_658_816


# --------------------------------------------------------------------------- #
# B. Tokenizer integrity
# --------------------------------------------------------------------------- #
def test_tokenizer_committed_blob_matches_and_working_tree_differs():
    result = tok.verify()
    hm = result["hash_method"]
    assert result["committed_blob_matches_l0706_anchor"] is True
    assert hm["working_tree_sha256"] != hm["committed_git_blob_sha256"]
    assert hm["committed_git_blob_sha256"] == TOK_COMMITTED_SHA
    assert hm["working_tree_sha256"].startswith("554c8e75")
    assert result["vocab_size"] == 5822


def test_tokenizer_never_rewritten():
    result = tok.verify()
    assert result["artifact_modified_or_rewritten"] is False


# --------------------------------------------------------------------------- #
# C. Corpus snapshot
# --------------------------------------------------------------------------- #
def test_corpus_snapshot_integrity_and_no_training_authority():
    result = corpus.verify()
    assert result["identity_and_integrity_verified"] is True
    assert result["records"] == 1052
    assert result["eligible_children"] == 1050
    assert result["eligible_tokens"] == 1_380_611
    assert result["language_python_only"] is True
    assert result["frozen_verdict"] == "FROZEN_RESEARCH_ONLY"
    assert result["frozen_for_training"] is False
    assert result["training_authorized"] is False
    assert result["contaminated_children"] == 0


def test_corpus_leakage_zero():
    result = corpus.verify()
    assert result["leakage_groups_spanning_splits"] == 0
    assert result["max_child_tokens"] == 2043


def test_corpus_benchmarks_intact():
    result = corpus.verify()
    assert result["benchmark_python_intact"] is True
    assert result["benchmark_multilingual_intact"] is True


# --------------------------------------------------------------------------- #
# D. Training configuration proposal
# --------------------------------------------------------------------------- #
def test_training_config_complete_but_not_authorized():
    proposal = trcfg.build_proposal()
    assert proposal["authorization"]["status"] == "NOT_AUTHORIZED"
    assert "NOT PROVIDED" in proposal["provenance"]["owner_authorization_id"]
    assert proposal["model"]["config_id"] == "taksh-foundation-111m"
    assert proposal["tokenizer"]["vocab_size"] == 5822
    assert proposal["optimizer_candidates"][0]["optimizer"] == "AdamW"
    assert proposal["lr_schedule_candidates"][0]["schedule"] == "cosine with warmup"
    assert proposal["batching"]["micro_batch_size"] == 4
    assert "limits" in proposal and proposal["limits"]["max_training_steps"] == 3000


def test_training_config_flags_unverified_hypotheses():
    proposal = trcfg.build_proposal()
    assert proposal["unverified_assumptions"]
    assert any("Adafactor" in a for a in proposal["unverified_assumptions"])


def test_training_config_proposal_files_provenance():
    proposal = trcfg.build_proposal()
    assert proposal["provenance"]["corpus_snapshot_binding"] == "l072-l04-fdb1fee2beaae9c2"


# --------------------------------------------------------------------------- #
# E. Resource estimates
# --------------------------------------------------------------------------- #
def test_resource_estimate_fails_closed():
    result = rest.verify()
    est_total = result["estimates"]["fp32_optimizer_state_total_bytes"]
    assert est_total == 110_658_816 * 4 * 4
    measured = result["measured"]
    if measured.get("available_bytes") is not None:
        assert measured["available_bytes"] < est_total   # host cannot hold the state


def test_resource_estimate_labels_and_no_cloud():
    text = json.dumps(rest.verify())
    assert "psutil" in text
    assert "cloud" not in text.lower() or "consulted" in text.lower()


def test_resource_estimate_mx150_recorded_as_unusable():
    result = rest.verify()
    assert result["gpu"]["model"] == "NVIDIA GeForce MX150"
    assert result["gpu"]["torch_cuda_available"] is False
    assert "NOT VIABLE" in result["gpu"]["viability"]


# --------------------------------------------------------------------------- #
# F. Checkpoint and recovery readiness
# --------------------------------------------------------------------------- #
def test_checkpoint_recovery_synthetic_tamper_rejected(tmp_path):
    import training.checkpoint_lifecycle as cl

    ckpt = tmp_path / "fixture.pt"
    cl.write_checkpoint(ckpt, {"w": [0.0]}, {"m": [0.0]},
                              {"step": 0, "model_cfg": {"vocab_size": 5822},
                               "tokenizer_sha": "a" * 64, "tokens_processed": 0})
    clean = cl.verify_checkpoint(ckpt)
    assert clean["ok"] is True

    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    blob["config"]["tokens_processed"] = 999
    torch.save(blob, ckpt)
    tampered = cl.verify_checkpoint(ckpt)
    assert tampered["ok"] is False
    assert any("fingerprint" in p for p in tampered["problems"])


def test_checkpoint_recovery_artifact_missing():
    result = json.loads(Path("reports").joinpath("l080pre_checkpoint_recovery.json").read_text(encoding="utf-8"))
    assert result["canonical_111m_artifact"] == "MISSING"
    assert result["v180_sft_classification"] == "GEOMETRY_INCOMPATIBLE"
    assert result["synthetic_artifacts_are_never_production"] is True


# --------------------------------------------------------------------------- #
# G. Benchmarks
# --------------------------------------------------------------------------- #
def test_benchmarks_intact():
    assert hashlib.sha256(Path("evaluation/l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA
    assert hashlib.sha256(Path("evaluation/l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_benchmark_safeguard_contamination_zero():
    result = json.loads(Path("reports").joinpath("l080pre_benchmark_safeguard.json").read_text(encoding="utf-8"))
    assert result["contamination_detected"] is False
    assert result["contamination_shadow_probe"][0]["exact_overlap"] == []
    assert result["capability_claim"].startswith("none")
    assert result["frozen_python_benchmark"]["intact"] is True


# --------------------------------------------------------------------------- #
# H. Aggregate gate — preparation vs training verdicts are independent
# --------------------------------------------------------------------------- #
def test_gate_preparation_qualified_but_training_no_go():
    result = json.loads(Path("reports").joinpath("l080pre_aggregate.json").read_text(encoding="utf-8"))
    assert result["preparation_verdict"] == "PREPARATION_QUALIFIED"
    assert result["training_verdict"] == "NO_GO"
    assert result["training_authorized"] is False
    assert result["training_eligible_tokens"] == 0
    assert len(result["blocking_dimensions"]) >= 5


def test_gate_dimension_count_and_governance():
    result = json.loads(Path("reports").joinpath("l080pre_aggregate.json").read_text(encoding="utf-8"))
    assert result["dimension_count"] == 18
    assert result["governance"]["A1_1_C_preparation_only"] is True
    assert result["governance"]["A2_E_no_compute"] is True
    assert result["governance"]["v180_sft_not_canonical"] is True
    assert result["governance"]["frozen_benchmarks_immutable"] is True


def test_gate_dimensions_have_owners_and_evidence():
    result = json.loads(Path("reports").joinpath("l080pre_aggregate.json").read_text(encoding="utf-8"))
    for name, d in result["dimensions"].items():
        assert d["evidence"], name
        assert d["responsible_party"], name
        assert isinstance(d["owner_decision"], bool), name


def test_preparation_verdict_does_not_authorize_training():
    text = Path("reports").joinpath("l080pre_aggregate.json").read_text(encoding="utf-8").lower()
    assert '"training_authorized": true' not in text
    assert "training_tokens_eligible" not in text


# --------------------------------------------------------------------------- #
# Preservation anchors (v180_sft, foundation config, tokenizer blob)
# --------------------------------------------------------------------------- #
def test_l01_checkpoint_preserved():
    digest = hashlib.sha256()
    with Path("runs/v180_sft/checkpoint_last.pt").open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_foundation_config_preserved():
    assert hashlib.sha256(Path("configs/foundation/l060_foundation.yaml").read_bytes()).hexdigest() == FOUNDATION_CFG_SHA


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=str(Path(".").resolve())).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA


# --------------------------------------------------------------------------- #
# Cross-cutting invariants
# --------------------------------------------------------------------------- #
def test_v180_sft_never_substituted_viatooling():
    agg = json.loads(Path("reports").joinpath("l080pre_aggregate.json").read_text(encoding="utf-8"))
    assert agg["governance"]["v180_sft_not_canonical"] is True
    assert agg["dimensions"]["checkpoint_availability"]["status"] == "MISSING"


def test_karkain_remains_discovered_not_qualified():
    strategy = yaml.safe_load(Path("configs/data/l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"
    assert strategy["identity"]["spec_version_pinned"] == "1.1.0"
