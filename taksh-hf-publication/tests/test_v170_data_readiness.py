"""Taksh v1.7.0 qualification tests: candidate research, authorization
preflight, evaluation budgets, repetition metrics and learning-experiment design.

Offline, CPU-bounded, deterministic. No source is approved by any test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.python_v1.candidates import (
    CANDIDATE_STATES,
    EVIDENCE_CLASSES,
    build_review_packet,
    load_registry,
    rank_candidates,
    validate_candidate,
)
from data.python_v1.preflight import (
    TRAINING_AUTHORIZED,
    TRAINING_NOT_AUTHORIZED,
    preflight,
)
from data.python_v1.readiness import assess
from evaluation.python_eval.budget import (
    BudgetError,
    classify_completion,
    classify_termination,
    repeated_line_ratio,
    repetition_metrics,
    validate_budget,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY = REPO_ROOT / "data" / "contracts" / "python_candidates_v1.json"
V14_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v1.json"
V15_CASES = REPO_ROOT / "evaluation" / "python_eval" / "data" / "python_capability_v15.json"


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


# --------------------------------------------------------------------------- #
# Candidate research registry
# --------------------------------------------------------------------------- #
def test_candidate_registry_loads(registry):
    assert len(registry["candidates"]) >= 8
    assert registry["methodology"]["network_access"] == "none used in this milestone"


def test_candidate_states_are_explicit(registry):
    for candidate in registry["candidates"]:
        assert candidate["state"] in CANDIDATE_STATES
        assert candidate["evidence_class"] in EVIDENCE_CLASSES


def test_candidate_states_never_mean_production_approval(registry):
    for candidate in registry["candidates"]:
        assert candidate["state"] != "approved"
        packet = build_review_packet(candidate)
        assert "never implies approval" in packet["note"]


def test_candidate_categories_cover_the_required_set(registry):
    categories = {candidate["category"] for candidate in registry["candidates"]}
    assert {
        "public_python_repositories",
        "python_code_datasets",
        "code_documentation_pairs",
        "tests_and_implementations",
        "algorithmic_examples",
        "licensed_educational_material",
        "owned_original_python",
        "synthetic_or_generated",
    } <= categories


def test_candidate_evidence_gaps_are_recorded(registry):
    for candidate in registry["candidates"]:
        assert candidate["evidence_gaps"]
        assert candidate["required_human_decisions"]


def test_candidate_schema_rejects_an_unknown_state():
    problems = validate_candidate(
        {"candidate_id": "x", "category": "python_code_datasets", "name": "x",
         "evidence_class": "UNKNOWN", "state": "approved",
         "evidence_gaps": [], "required_human_decisions": []}
    )
    assert problems


def test_review_packet_lists_unresolved_questions(registry):
    candidate = registry["candidates"][0]
    packet = build_review_packet(candidate)
    assert packet["schema"] == "quantsmind.candidate_review_packet/1"
    assert packet["unresolved_questions"]
    assert packet["required_human_decisions"]
    assert packet["explicit_blockers"]
    assert packet["recommended_next_action"]


def test_ranking_is_gated_by_authorization_feasibility(registry):
    ranked = rank_candidates(registry["candidates"])
    assert len(ranked) == len(registry["candidates"])
    blocked_flags = [row["authorization_blocked"] for row in ranked]
    assert blocked_flags == sorted(blocked_flags)
    assert any(row["authorization_blocked"] for row in ranked)


def test_ranking_is_deterministic(registry):
    first = rank_candidates(registry["candidates"])
    second = rank_candidates(registry["candidates"])
    assert first == second


# --------------------------------------------------------------------------- #
# Training authorization preflight
# --------------------------------------------------------------------------- #
def _approved_manifest(tmp_path) -> Path:
    from data.python_v1.registry import file_sha256, snapshot_digest

    src = tmp_path / "src"
    src.mkdir()
    (src / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (src / "m.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    entries = [
        (path.relative_to(src).as_posix(), file_sha256(path))
        for path in sorted(src.rglob("*.py"))
    ]
    manifest = {
        "schema": "quantsmind.source_manifest/2",
        "source_id": "approved_src",
        "source_name": "approved_src",
        "origin": "local://approved_src",
        "publisher": "test",
        "revision": snapshot_digest(entries),
        "source_type": "local_directory",
        "declared_license": "MIT",
        "license_evidence": {
            "path": "LICENSE",
            "evidence_present": True,
            "evidence_hash": file_sha256(src / "LICENSE"),
        },
        "intended_use": "python_pretraining",
        "approved_uses": ["python_pretraining"],
        "commercial_use": "permitted for internal research use",
        "modification_rights": "permitted",
        "attribution_obligations": "retain the notice",
        "provenance_evidence": {"established": True, "chain": "locally generated"},
        "collection_method": "generated locally by the test",
        "personal_data_risk": "none",
        "secrets_risk": "none",
        "third_party_rights_risk": "none",
        "contamination_assessment": "screened against the frozen suites",
        "authorization_state": "approved",
        "reviewer": "test-reviewer",
        "decision_timestamp": "2026-10-21T00:00:00+00:00",
        "decision_rationale": "approved for a pipeline test only",
        "revocation_status": "not_revoked",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_preflight_refuses_a_missing_manifest(tmp_path):
    decision = preflight(tmp_path / "nope.json")
    assert decision["decision"] == TRAINING_NOT_AUTHORIZED
    assert "source_manifest_missing" in decision["blockers"]


def test_preflight_refuses_a_pending_source(tmp_path):
    path = _approved_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["authorization_state"] = "pending_review"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = preflight(path)
    assert decision["decision"] == TRAINING_NOT_AUTHORIZED
    assert any("authorization_state_is_pending_review" in b for b in decision["blockers"])


def test_preflight_refuses_a_revoked_source(tmp_path):
    path = _approved_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["revocation_status"] = "revoked"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = preflight(path)
    assert "authorization_revoked" in decision["blockers"]


def test_preflight_refuses_an_expired_source(tmp_path):
    path = _approved_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["decision_expires_on"] = "2020-01-01"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = preflight(path, now="2026-10-21")
    assert "authorization_expired" in decision["blockers"]


def test_preflight_refuses_a_revision_mismatch(tmp_path):
    path = _approved_manifest(tmp_path)
    decision = preflight(path, expected_revision="sha256:" + "0" * 64)
    assert "revision_mismatch" in decision["blockers"]


def test_preflight_refuses_an_out_of_scope_use(tmp_path):
    path = _approved_manifest(tmp_path)
    decision = preflight(path, intended_use="commercial_redistribution")
    assert "use_out_of_approved_scope" in decision["blockers"]


def test_preflight_refuses_an_unassessed_commercial_scope(tmp_path):
    path = _approved_manifest(tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["commercial_use"] = "not_assessed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    decision = preflight(path)
    assert "commercial_use_not_assessed" in decision["blockers"]


def test_preflight_refuses_an_unknown_contamination_status(tmp_path):
    path = _approved_manifest(tmp_path)
    decision = preflight(path, contamination_status="unknown")
    assert "contamination_status_unknown" in decision["blockers"]


def test_preflight_refuses_a_source_outside_the_approved_registry(tmp_path):
    path = _approved_manifest(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"approved_sources": []}), encoding="utf-8")
    decision = preflight(path, registry_path=registry)
    assert "source_not_in_approved_registry" in decision["blockers"]


def test_preflight_never_approves_a_manifest_it_did_not_approve(tmp_path):
    """A preflight is a report, not a mutation."""
    path = _approved_manifest(tmp_path)
    before = path.read_bytes()
    preflight(path)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------- #
# Zero-source readiness
# --------------------------------------------------------------------------- #
def test_zero_source_readiness_stays_blocked():
    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["declared"] == 0
    assert report["eligible_training_sources"] == 0
    assert report["production_training_authorized"] is False
    assert report["default_decision"] == "deny"
    assert report["missing_decisions"]


def test_readiness_does_not_mutate_the_contract():
    path = REPO_ROOT / "data" / "contracts" / "python_v1.yaml"
    before = path.read_bytes()
    assess(path)
    assert path.read_bytes() == before


# --------------------------------------------------------------------------- #
# Generation budget, termination and repetition
# --------------------------------------------------------------------------- #
def test_budget_validation_rejects_an_impossible_combination():
    with pytest.raises(BudgetError):
        validate_budget(400, 256, 512)
    with pytest.raises(BudgetError):
        validate_budget(600, 32, 512)
    with pytest.raises(BudgetError):
        validate_budget(10, 0, 512)


def test_budget_validation_accepts_a_fitting_combination():
    result = validate_budget(100, 128, 512)
    assert result["remaining_context"] == 412


def test_termination_classification():
    assert classify_termination(finish_reason="stop", generated_tokens=10, budget=64, empty=False) == "eos"
    assert classify_termination(finish_reason="length", generated_tokens=64, budget=64, empty=False) == "budget"
    assert classify_termination(finish_reason=None, generated_tokens=0, budget=64, empty=True) == "empty"
    assert classify_termination(finish_reason="context", generated_tokens=64, budget=128, empty=False) == "context"


def test_repetition_metrics_detect_degenerate_output():
    metrics = repetition_metrics([5] * 40)
    assert metrics["max_repeated_run"] == 40
    assert metrics["unique_token_ratio"] < 0.1
    varied = repetition_metrics(list(range(40)))
    assert varied["unique_token_ratio"] == 1.0
    assert varied["max_repeated_run"] == 1


def test_repeated_line_ratio():
    assert repeated_line_ratio("x = 1\nx = 1\ny = 2\n") == pytest.approx(1 / 3)
    assert repeated_line_ratio("") == 0.0


def test_quality_classification_separates_repetition_from_parsing():
    assert classify_completion(text="a" * 100, token_ids=[7] * 50, parseable=True, complete_program=True) == "degenerate_repetition"
    assert classify_completion(text="def f():\n    return 1\n", token_ids=[1, 2, 3, 4, 5, 6], parseable=True, complete_program=True) == "complete_program"
    assert classify_completion(text="x = ", token_ids=[1, 2, 3], parseable=False, complete_program=False) == "invalid"


def test_repetition_metrics_are_deterministic():
    assert repetition_metrics([1, 2, 2, 3]) == repetition_metrics([1, 2, 2, 3])


# --------------------------------------------------------------------------- #
# Frozen benchmarks, experiment plan and resources
# --------------------------------------------------------------------------- #
def test_frozen_benchmarks_are_preserved():
    v14 = json.loads(V14_CASES.read_text(encoding="utf-8"))
    v15 = json.loads(V15_CASES.read_text(encoding="utf-8"))
    assert len(v14["cases"]) == 20
    assert len(v15["cases"]) == 16
    assert all("surface" not in case for case in v14["cases"])


def test_experiment_plan_is_not_authorized_for_execution():
    from training.experiment_plan import EXPERIMENT_STATUS, build_plan

    plan = build_plan(registry_path=REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert plan["status"] == EXPERIMENT_STATUS
    assert plan["authorization"]["status"] == EXPERIMENT_STATUS
    assert plan["blocking_prerequisites"]


def test_experiment_plan_states_hypothesis_criteria_and_stops():
    from training.experiment_plan import build_plan

    plan = build_plan()
    assert "improve" in plan["hypothesis"]
    assert plan["evaluation"]["criteria"]
    assert plan["stop_conditions"]
    assert plan["training"]["seed"]
    assert plan["resources"]["memory_model"]["training_total_gib"] > 0


def test_resource_estimate_is_consistent():
    from training.resources import memory_model, runtime_model

    memory = memory_model(parameters=9_359_104, vocab_size=5822, d_model=256,
                          layers=8, context_length=512)
    assert memory["weight_gib"] < memory["optimizer_gib"]
    assert memory["training_total_gib"] > memory["weight_gib"]
    unknown = runtime_model(tokens=1000, measured_tokens_per_second=None)
    assert unknown["seconds"] is None
    known = runtime_model(tokens=1000, measured_tokens_per_second=100.0)
    assert known["seconds"] == pytest.approx(10.0)


def test_hardware_inventory_is_redacted():
    from training.resources import hardware_inventory

    inventory = hardware_inventory()
    assert inventory["processor"].startswith("sha256:")
    assert "users" not in json.dumps(inventory).lower()
    assert inventory["accelerator"] in ("none", "cuda", "mps")
