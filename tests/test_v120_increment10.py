"""Focused tests for Increment 10 preparation: gates and promotion. No training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.experiment_artifacts import file_sha256
from training.v120_experiment import GateError
from training.v120_experiment10 import (
    DEFAULT_CONFIG,
    check_checkpoint_gate,
    check_curriculum_gates,
    check_digest_gates,
    check_fixture_gates,
    check_functional_gates,
    decide_promotion10,
    experiment_identity,
    load_config,
    run_preflight,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_v120_increment10.py"
PROMOTER = REPO / "scripts" / "promote_v120_candidate10.py"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")


def test_config_loads_and_identity_reproducible():
    first = load_config()
    second = load_config(DEFAULT_CONFIG)
    assert experiment_identity(first) == experiment_identity(second)
    assert first["steps"] == 100
    assert first["identifier_supervision"]["copy_weight"] == 2.0
    assert first["combined_train_examples"] == 268


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "steps": 101},
        lambda c: {**c, "lr": 0},
        lambda c: {**c, "device": "cuda"},
        lambda c: {**c, "run_id": "../escape"},
        lambda c: {
            **c,
            "identifier_supervision": {**c["identifier_supervision"], "copy_weight": 0.5},
        },
    ],
)
def test_config_validation_rejects_unsafe_values(mutate):
    with pytest.raises(GateError):
        validate_config(mutate(load_config()))


def test_gates_pass_and_tamper_fails():
    config = load_config()
    assert check_checkpoint_gate(config) == config["baseline_checkpoint_sha256"]
    digests = check_digest_gates(config)
    assert digests["binding"] == "bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b"
    assert digests["copy"] == "7d8037893e384c85a784010f01fb6f7a6840787d76541cfa0d2af825b3fa9333"
    fixtures = check_fixture_gates(config)
    assert fixtures["ad1"]["cases"] == 29
    assert fixtures["supplemental"]["cases"] == 48
    assert fixtures["copy_probes"]["cases"] == 13
    tampered = {
        **config,
        "curricula": {**config["curricula"], "copy": {**config["curricula"]["copy"], "digest": "0" * 64}},
    }
    with pytest.raises(GateError, match="curriculum digest gate"):
        check_digest_gates(tampered)


def test_combined_quality_and_functional_gates():
    config = load_config()
    result = check_curriculum_gates(config)
    assert result["quality"]["train_examples"] == 268
    assert result["quality"]["heldout_overlap"] == 0
    assert check_functional_gates(config)["copy_examples_verified"] == 112


def test_training_requires_explicit_authorization(tmp_path):
    config = {**load_config(), "experiment_root": str(tmp_path)}
    with pytest.raises(GateError, match="authorization gate"):
        run_preflight(config, authorized=False, for_training=True)
    report = run_preflight(config, authorized=False, for_training=False)
    assert report["mode"] == "dry_run"
    assert not (tmp_path / config["run_id"]).exists()


# --- Promotion matrix ---------------------------------------------------------

def test_promotion_gate_matrix():
    baseline = {
        "baseline_ad1_binding": 6,
        "baseline_ad1_parses": 24,
    }
    good = decide_promotion10(
        copy_exact=11, supplemental_passes=5, supplemental_behaviors=4,
        baseline_ad1_binding=6, candidate_ad1_binding=5,
        baseline_ad1_parses=24, candidate_ad1_parses=24,
    )
    assert good["promote"] is True
    weak_copy = decide_promotion10(
        copy_exact=3, supplemental_passes=5, supplemental_behaviors=4,
        baseline_ad1_binding=6, candidate_ad1_binding=5,
        baseline_ad1_parses=24, candidate_ad1_parses=24,
    )
    assert weak_copy["promote"] is False
    single_template = decide_promotion10(
        copy_exact=11, supplemental_passes=1, supplemental_behaviors=1,
        baseline_ad1_binding=6, candidate_ad1_binding=5,
        baseline_ad1_parses=24, candidate_ad1_parses=24,
    )
    assert single_template["promote"] is False
    ad1_loss = decide_promotion10(
        copy_exact=11, supplemental_passes=5, supplemental_behaviors=4,
        baseline_ad1_binding=6, candidate_ad1_binding=2,
        baseline_ad1_parses=24, candidate_ad1_parses=24,
    )
    assert ad1_loss["promote"] is False
    assert decide_promotion10(
        copy_exact=11, supplemental_passes=5, supplemental_behaviors=4,
        baseline_ad1_binding=6, candidate_ad1_binding=5,
        baseline_ad1_parses=24, candidate_ad1_parses=24,
        hashes_ok=False,
    )["promote"] is False


# --- CLI dry runs (never train) ------------------------------------------------

def test_runner_dry_run_writes_no_checkpoint(tmp_path):
    before = file_sha256(BASELINE)
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--root", str(tmp_path), "--run-id", "dry-test-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry_run"
    assert plan["identifier_supervision"]["copy_weight"] == 2.0
    assert not (tmp_path / "dry-test-run").exists()
    assert file_sha256(BASELINE) == before


def test_runner_refuses_train_without_authorize(tmp_path):
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--train", "--root", str(tmp_path), "--run-id", "noauth-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 2
    assert "authorization" in result.stderr.lower()
    assert not (tmp_path / "noauth-run").exists()


def test_promoter_cli_with_synthetic_evaluation(tmp_path):
    ad1_records = [
        {"case_id": f"A{i}", "category": "t", "parses": True, "exact": False,
         "structural": True, "binding": True, "completion": "    return x\n"}
        for i in range(5)
    ] + [
        {"case_id": f"B{i}", "category": "t", "parses": True, "exact": False,
         "structural": True, "binding": False, "completion": "    return x\n"}
        for i in range(24)
    ]
    c3_records = [
        {"case_id": f"C{i}", "category": "t", "parses": True, "exact": False,
         "structural": True, "binding": False, "functional": {"passed": False, "reason": "x"},
         "completion": "    return x\n"}
        for i in range(54)
    ]
    supp_records = [
        {"case_id": f"S{i}", "behavior": f"b{i % 4}", "parses": True, "exact": False,
         "structural": True, "binding": True, "functional": {"passed": i < 5},
         "completion": "    return x\n"}
        for i in range(48)
    ]
    from evaluation.copy_eval import copy_report

    copy_scored = [
        {"kind": "add", "familiar": i < 6, "parses": True, "exact_copy": i < 11,
         "substitution": False, "propagation": True, "multi_binding": False}
        for i in range(13)
    ]
    side = {
        "copy_report": copy_report(copy_scored),
        "supplemental_report": {"functional_rate": 5 / 48},
        "supplemental_by_behavior": {f"b{k}": {"functional_rate": 1.0} for k in range(4)},
        "ad1": ad1_records,
        "c3": c3_records,
        "supplemental": supp_records,
    }
    baseline_side = {
        **json.loads(json.dumps(side)),
    }
    for record in baseline_side["supplemental"]:
        record["functional"] = {"passed": False}
    baseline_side["ad1"] = [
        {**record, "binding": index < 6, "parses": True}
        for index, record in enumerate(ad1_records)
    ]
    payload = {
        "baseline_checkpoint": str(BASELINE),
        "candidate_checkpoint": str(BASELINE),
        "summary": {"baseline": baseline_side, "candidate": side},
    }
    # Give the summarizer what decide needs on candidate side.
    payload["summary"]["candidate"]["copy_report"] = copy_report(copy_scored)
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PROMOTER), "--evaluation", str(path)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)["decision"]
    assert decision["promote"] is True, decision["reasons"]
