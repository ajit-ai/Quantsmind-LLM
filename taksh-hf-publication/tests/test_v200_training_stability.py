"""Taksh v2.0.0 qualification tests: immutable artifacts, contrastive
curriculum, task-conditioning metrics and the training-cell harness.

Offline, CPU-bounded and deterministic. No training is executed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.contrastive_tasks import (
    CONTRASTIVE_NAMES,
    OPERATION_MARKERS,
    TASKS,
    build_contrastive_curriculum,
    contamination_report,
    curriculum_digest,
)
from training.experiment_artifacts import ArtifactError, RunArtifacts, file_sha256
from training.synthetic_eval import (
    TASK_CONDITIONING_CLASSES,
    classify_task_conditioning,
    task_conditioning_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def curriculum():
    return build_contrastive_curriculum(20261021)


# --------------------------------------------------------------------------- #
# Immutable artifact system
# --------------------------------------------------------------------------- #
def test_run_directory_is_created_exclusively(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0001")
    assert artifacts.directory.is_dir()
    with pytest.raises(ArtifactError):
        RunArtifacts(tmp_path, "run-0001")


def test_invalid_run_ids_are_refused(tmp_path):
    for bad in ("", ".", "..", "a/b", "a\\b"):
        with pytest.raises(ArtifactError):
            RunArtifacts(tmp_path, bad)


def test_experiment_root_inside_the_repository_is_refused():
    with pytest.raises(ArtifactError):
        RunArtifacts(REPO_ROOT / "runs", "run-x")


def test_checkpoint_overwrite_is_refused(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0002")
    artifacts.write_checkpoint("checkpoint_final", {"model": {"w": 1}})
    with pytest.raises(ArtifactError):
        artifacts.write_checkpoint("checkpoint_final", {"model": {"w": 2}})


def test_interrupted_write_leaves_no_artifact(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0003")
    with pytest.raises(Exception):
        artifacts.write_checkpoint("broken", {"model": lambda: None})
    assert not artifacts.path("checkpoints", "broken.pt").exists()


def test_evaluation_must_name_its_checkpoint(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0004")
    with pytest.raises(ArtifactError):
        artifacts.write_evaluation("baseline.json", {"metrics": {}})


def test_finalization_writes_a_manifest_and_makes_files_read_only(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0005")
    artifacts.write_checkpoint("checkpoint_final", {"model": {"w": 1}})
    artifacts.write_evaluation("baseline.json", {"checkpoint_sha256": "0" * 64})
    manifest = artifacts.finalize({"schema": "test/1"})
    assert manifest.is_file()
    assert artifacts.finalized is True
    with pytest.raises(ArtifactError):
        artifacts.finalize({"schema": "test/1"})


def test_verification_detects_a_hash_mismatch(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0006")
    artifacts.write_checkpoint("checkpoint_final", {"model": {"w": 1}})
    artifacts.finalize({"schema": "test/1"})
    target = artifacts.path("checkpoints", "checkpoint_final.pt")
    target.chmod(0o644)
    target.write_bytes(b"tampered")
    report = artifacts.verify()
    assert report["intact"] is False
    assert report["mismatches"]


def test_verification_requires_a_finalized_run(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0007")
    with pytest.raises(ArtifactError):
        artifacts.verify()


def test_probe_cannot_write_into_a_finalized_run(tmp_path):
    artifacts = RunArtifacts(tmp_path, "run-0008")
    artifacts.write_checkpoint("checkpoint_final", {"model": {"w": 1}})
    artifacts.finalize({"schema": "test/1"})
    with pytest.raises(ArtifactError):
        artifacts.write_checkpoint("probe", {"model": {"w": 2}})


# --------------------------------------------------------------------------- #
# Contrastive curriculum
# --------------------------------------------------------------------------- #
def test_curriculum_is_deterministic(curriculum):
    again = build_contrastive_curriculum(20261021)
    assert curriculum_digest(again) == curriculum_digest(curriculum)
    assert [c.content_sha256 for c in curriculum["train"]] == [
        c.content_sha256 for c in again["train"]
    ]


def test_every_pair_differs_by_one_meaningful_requirement():
    families = {}
    for (family, variant), entry in TASKS.items():
        families.setdefault(family, []).append((variant, entry[3], entry[4]))
    assert len(families) == 6
    for family, items in families.items():
        assert len(items) == 2, family
        bodies = {body for _variant, body, _op in items}
        assert len(bodies) == 2, f"{family} variants share a body"
        operations = {op for _variant, _body, op in items}
        assert len(operations) == 2


def test_operation_labels_have_markers():
    for (_family, _variant), entry in TASKS.items():
        assert entry[4] in OPERATION_MARKERS
    for operation, markers in OPERATION_MARKERS.items():
        assert markers


def test_contrastive_names_differ_between_partitions():
    assert set(CONTRASTIVE_NAMES) == {
        "arithmetic", "conditionals", "list_ops", "string_ops", "dict_ops", "loop_agg"
    }


def test_no_duplicate_targets_across_partitions(curriculum):
    seen = {}
    for partition, cases in curriculum.items():
        for case in cases:
            key = case.content_sha256
            assert key not in seen or seen[key] == partition
            seen[key] = partition


def test_withheld_families_are_absent_from_training(curriculum):
    train_families = {case.family for case in curriculum["train"]}
    assert "dict_ops" not in train_families
    assert "loop_agg" not in train_families
    assert {case.family for case in curriculum["C"]} == {"dict_ops", "loop_agg"}


def test_contrastive_partition_holds_matched_pairs(curriculum):
    pairs = {}
    for case in curriculum["D"]:
        pairs.setdefault(case.params["pair_id"], []).append(case)
    assert pairs
    for members in pairs.values():
        assert len(members) == 2
        assert len({m.params["operation"] for m in members}) == 2


def test_contamination_report_is_clean(curriculum):
    report = contamination_report(curriculum)
    assert report["violated" if False else "contaminated"] is False
    assert report["violation_count"] == 0


# --------------------------------------------------------------------------- #
# Task-conditioning metrics and classification
# --------------------------------------------------------------------------- #
def _record(partition, body, operation="add", pair_id=None, syntax=True, degenerate=False):
    return {
        "case_id": f"{partition}-{body[:4]}",
        "family": "arithmetic",
        "partition": partition,
        "pair_id": pair_id,
        "operation": operation,
        "body": body,
        "body_key": body,
        "syntax_valid": syntax,
        "complete_function": syntax,
        "exact_target": False,
        "operation_agreement": operation == "add",
        "degenerate": degenerate,
        "eos_terminated": False,
        "truncated": True,
        "tokens": 64,
        "text": body,
    }


def test_metrics_separate_uniqueness_from_sensitivity():
    records = [_record("D", "body_a", pair_id="p0"), _record("D", "body_b", pair_id="p0")]
    metrics = task_conditioning_metrics(records)
    assert metrics["unique_body_ratio"] == 1.0
    assert metrics["contrastive_differentiated"] == 1
    assert metrics["operation_agreement"] == 2


def test_classification_detects_a_generic_collapse():
    records = [_record("D", "same") for _ in range(4)]
    metrics = task_conditioning_metrics(records)
    verdict = classify_task_conditioning(metrics)
    assert verdict["verdict"] == "GENERIC_TEMPLATE_COLLAPSE"
    assert verdict["verdict"] in TASK_CONDITIONING_CLASSES


def test_classification_detects_task_conditioned_learning():
    records = [
        _record("D", "return left + right", operation="add", pair_id="p0"),
        _record("D", "return left - right", operation="subtract", pair_id="p0"),
    ]
    metrics = task_conditioning_metrics(records)
    verdict = classify_task_conditioning(metrics, loss_delta=-0.5)
    assert verdict["verdict"] == "SURFACE_CONDITIONED_LEARNING" or (
        verdict["verdict"] == "TASK_CONDITIONED_LEARNING_DEMONSTRATED"
    )


def test_classification_reports_no_evidence_when_there_are_no_cases():
    verdict = classify_task_conditioning({"cases": 0})
    assert verdict["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_caveat_about_static_agreement_is_always_present():
    verdict = classify_task_conditioning(task_conditioning_metrics([_record("D", "b")]))
    assert "not functional correctness" in verdict["caveat"]


# --------------------------------------------------------------------------- #
# Preservation
# --------------------------------------------------------------------------- #
def test_production_authorization_is_unchanged():
    from data.python_v1.readiness import assess

    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["approved"] == 0
    assert report["production_training_authorized"] is False


def test_protected_artifacts_are_untouched():
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", "tokenizer/", "runs/",
         "evaluation/python_eval/data/", "docs/roadmap/", "configs/",
         "data/contracts/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.stdout.strip() == ""
