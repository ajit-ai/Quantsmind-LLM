"""Taksh v1.9.0 qualification tests: synthetic curriculum, contamination
controls, evaluation protocol, classification and preservation.

Offline, CPU-bounded, deterministic. No training is executed by this suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.synthetic_curriculum import (
    FAMILIES,
    HELDOUT_FAMILIES,
    TRAIN_FAMILIES,
    build_curriculum,
    contamination_report,
    curriculum_digest,
)
from training.synthetic_eval import CLASSIFICATIONS, aggregate_records, classify, evaluate_cases

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def curriculum():
    return build_curriculum(20261021)


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
def test_generation_is_deterministic(curriculum):
    again = build_curriculum(20261021)
    assert curriculum_digest(again) == curriculum_digest(curriculum)
    assert [case.content_sha256 for case in curriculum["train"]] == [
        case.content_sha256 for case in again["train"]
    ]


def test_all_six_families_are_covered(curriculum):
    families = {case.family for cases in curriculum.values() for case in cases}
    assert set(FAMILIES) <= families
    assert set(TRAIN_FAMILIES) < families
    assert set(HELDOUT_FAMILIES) <= families


def test_partitions_have_the_expected_shape(curriculum):
    assert len(curriculum["train"]) == 120
    assert len(curriculum["A"]) == 40
    assert len(curriculum["B"]) == 30
    assert len(curriculum["C"]) == 40


def test_withheld_families_never_appear_in_training(curriculum):
    train_families = {case.family for case in curriculum["train"]}
    for family in HELDOUT_FAMILIES:
        assert family not in train_families
        assert any(case.family == family for case in curriculum["C"])


def test_exact_duplicates_do_not_cross_partitions(curriculum):
    seen: dict = {}
    for partition, cases in curriculum.items():
        for case in cases:
            key = case.content_sha256
            assert key not in seen or seen[key] == partition
            seen[key] = partition


def test_normalized_duplicates_only_appear_in_the_inherited_partitions(curriculum):
    from training.synthetic_curriculum import _normalized_hash

    groups: dict = {}
    for partition, cases in curriculum.items():
        for case in cases:
            groups.setdefault(_normalized_hash(case.reference), set()).add(partition)
    for partitions in groups.values():
        if len(partitions) > 1:
            assert partitions <= {"train", "A"}


def test_structural_partitions_are_distinct_from_training(curriculum):
    report = contamination_report(curriculum)
    assert report["verdict"] == "CLEAN"
    assert report["violation_count"] == 0
    assert report["max_train_similarity"]["B"] < 0.5
    assert report["max_train_similarity"]["C"] < 0.5


def test_contamination_detects_an_injected_leak(curriculum):
    broken = {key: list(value) for key, value in curriculum.items()}
    leaked = broken["train"][0]
    from training.synthetic_curriculum import Case

    broken["C"].append(
        Case(
            case_id="C-leak",
            family=leaked.family,
            partition="C",
            template_id=leaked.template_id,
            prompt=leaked.prompt,
            reference=leaked.reference,
            content_sha256=leaked.content_sha256,
        )
    )
    report = contamination_report(broken)
    assert report["contaminated"] is True
    assert report["verdict"] == "SYNTHETIC_EVALUATION_CONTAMINATED"


def test_synthetic_sources_are_not_approved_for_production():
    from data.python_v1.readiness import assess

    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["approved"] == 0
    assert report["production_training_authorized"] is False


def test_the_experiment_cannot_use_an_unapproved_corpus(tmp_path):
    """A synthetic manifest may not authorise a real corpus."""
    from data.python_v1.preflight import TRAINING_NOT_AUTHORIZED, preflight

    manifest = {
        "schema": "quantsmind.source_manifest/2",
        "source_id": "synthetic",
        "revision": "sha256:" + "0" * 64,
        "declared_license": "MIT",
        "license_evidence": {"evidence_present": True, "evidence_hash": "0" * 64},
        "provenance_evidence": {"established": True},
        "authorization_state": "approved",
        "reviewer": "test",
        "decision_timestamp": "2026-10-21T00:00:00+00:00",
        "commercial_use": "permitted",
        "attribution_obligations": "retain",
        "approved_uses": ["python_pretraining"],
        "personal_data_risk": "none",
        "secrets_risk": "none",
        "third_party_rights_risk": "none",
    }
    path = tmp_path / "m.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = preflight(path, contamination_status="unknown")
    assert decision["decision"] == TRAINING_NOT_AUTHORIZED
    assert "contamination_status_unknown" in decision["blockers"]


# --------------------------------------------------------------------------- #
# Evaluation protocol and classification
# --------------------------------------------------------------------------- #
def test_metrics_keep_denominators():
    records = [
        {"syntax_valid": True, "complete_function": True, "exact_target": False,
         "degenerate": False, "eos_terminated": False, "truncated": True,
         "structural_similarity": 0.4, "tokens": 10, "latency_seconds": 0.1,
         "case_id": "a", "family": "arithmetic", "partition": "A"},
        {"syntax_valid": False, "complete_function": False, "exact_target": False,
         "degenerate": True, "eos_terminated": False, "truncated": True,
         "structural_similarity": 0.0, "tokens": 10, "latency_seconds": 0.1,
         "case_id": "b", "family": "arithmetic", "partition": "A"},
    ]
    metrics = aggregate_records(records)
    assert metrics["cases"] == 2
    assert metrics["syntax_valid"] == 1
    assert metrics["syntax_valid_rate"] == 0.5
    assert metrics["degenerate_rate"] == 0.5
    assert metrics["per_family"]["arithmetic"]["total"] == 2


def test_per_family_reporting_is_present(curriculum):
    records = [
        {"syntax_valid": True, "complete_function": True, "exact_target": False,
         "degenerate": False, "eos_terminated": False, "truncated": False,
         "structural_similarity": 0.3, "tokens": 5, "latency_seconds": 0.0,
         "case_id": "x", "family": "arithmetic", "partition": "A"},
    ]
    metrics = aggregate_records(records)
    assert "arithmetic" in metrics["per_family"]


def test_classification_requires_more_than_loss_reduction():
    baseline = {"A": {"syntax_valid_rate": 0.0, "degenerate_rate": 0.0,
                      "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    trained = {"A": {"syntax_valid_rate": 1.0, "degenerate_rate": 1.0,
                     "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    verdict = classify(baseline, trained, loss_delta=-1.0)
    assert verdict["verdict"] == "SYNTHETIC_PATTERN_LEARNING_ONLY"
    assert verdict["verdict"] in CLASSIFICATIONS


def test_classification_detects_real_generalisation():
    baseline = {
        "A": {"syntax_valid_rate": 0.0, "degenerate_rate": 1.0,
              "exact_target_rate": 0.0, "mean_structural_similarity": 0.0},
        "B": {"syntax_valid_rate": 0.0, "degenerate_rate": 1.0,
              "exact_target_rate": 0.0, "mean_structural_similarity": 0.0},
        "C": {"syntax_valid_rate": 0.0, "degenerate_rate": 1.0,
              "exact_target_rate": 0.0, "mean_structural_similarity": 0.0},
    }
    trained = {
        "A": {"syntax_valid_rate": 0.8, "degenerate_rate": 0.2,
              "exact_target_rate": 0.4, "mean_structural_similarity": 0.5},
        "B": {"syntax_valid_rate": 0.6, "degenerate_rate": 0.2,
              "exact_target_rate": 0.3, "mean_structural_similarity": 0.4},
        "C": {"syntax_valid_rate": 0.5, "degenerate_rate": 0.3,
              "exact_target_rate": 0.2, "mean_structural_similarity": 0.3},
    }
    verdict = classify(baseline, trained, loss_delta=-0.5)
    assert verdict["verdict"] == "SYNTHETIC_GENERALIZATION_DEMONSTRATED"


def test_loss_reduction_without_coding_gain_is_classified():
    baseline = {"A": {"syntax_valid_rate": 0.5, "degenerate_rate": 0.5,
                      "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    trained = {"A": {"syntax_valid_rate": 0.5, "degenerate_rate": 0.5,
                     "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    verdict = classify(baseline, trained, loss_delta=-0.8)
    assert verdict["verdict"] == "LOSS_REDUCTION_WITHOUT_CODING_GAIN"


def test_no_measurable_learning_is_classified():
    baseline = {"A": {"syntax_valid_rate": 0.0, "degenerate_rate": 1.0,
                      "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    trained = {"A": {"syntax_valid_rate": 0.0, "degenerate_rate": 1.0,
                     "exact_target_rate": 0.0, "mean_structural_similarity": 0.0}}
    verdict = classify(baseline, trained, loss_delta=0.0)
    assert verdict["verdict"] == "NO_MEASURABLE_LEARNING"


def test_experiment_stays_unauthorized():
    from training.learning_protocol import EXPERIMENT_STATUS, build_protocol

    protocol = build_protocol()
    assert protocol["status"] == EXPERIMENT_STATUS
    assert len(protocol["authorization_gates"]) >= 7
    assert "statistical_limitations" in protocol


def test_historical_artifacts_are_preserved():
    """The v1.9.0 work must not have touched the frozen artifacts."""
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "tokenizer/", "runs/",
         "evaluation/python_eval/data/", "docs/roadmap/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.stdout.strip() == ""


def test_experimental_checkpoint_is_not_promoted():
    """No production path may point at an experimental checkpoint."""
    config = (REPO_ROOT / "configs" / "train" / "pretrain_v13.yaml").read_text(
        encoding="utf-8"
    )
    assert "v19" not in config
    assert "synthetic" not in config.split("mode:")[1].splitlines()[0]
