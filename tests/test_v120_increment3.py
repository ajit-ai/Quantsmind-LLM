"""Focused tests for Increment 3 preparation: gates and promotion. No training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.experiment_artifacts import file_sha256
from training.v120_experiment import (
    DEFAULT_CONFIG,
    GateError,
    check_checkpoint_gate,
    check_curriculum_gates,
    check_fixture_gates,
    check_memory_gate,
    check_output_gate,
    decide_promotion,
    experiment_identity,
    load_config,
    run_preflight,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_v120_increment3.py"
PROMOTER = REPO / "scripts" / "promote_v120_candidate.py"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
INCREMENT1_EVAL = Path(
    r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment1\increment1-binding-240\evaluations\heldout_comparison.json"
)


def _summary(ad1_parses, c3_parses, functional_passed, case_counts=(29, 54), collapsed=False):
    ad1_cases, c3_cases = case_counts
    return {
        "ad1": {"cases": ad1_cases, "parses": ad1_parses, "exact": 0, "structural": 0, "binding": 0},
        "c3_heldout": {
            "cases": c3_cases,
            "parses": c3_parses,
            "exact": 0,
            "structural": 0,
            "structural_cases": 48,
            "binding": 0,
            "functional_passed": functional_passed,
            "functional_cases": 45,
        },
        "mode_collapse": {
            "ad1": {"collapsed": collapsed, "reasons": []},
            "c3": {"collapsed": collapsed, "reasons": []},
        },
    }


# --- Configuration ---------------------------------------------------------

def test_config_loads_and_identity_is_reproducible():
    first = load_config()
    second = load_config(DEFAULT_CONFIG)
    assert experiment_identity(first) == experiment_identity(second)
    assert first["steps"] == 100
    assert first["lr"] == 2e-05
    assert first["seed"] == 20261106
    assert first["device"] == "cpu"
    assert first["run_id"] == "increment3-diverse-100"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "steps": 101},
        lambda c: {**c, "steps": 0},
        lambda c: {**c, "lr": 0},
        lambda c: {**c, "lr": 0.001},
        lambda c: {**c, "device": "cuda"},
        lambda c: {**c, "run_id": "../escape"},
        lambda c: {**c, "experiment_root": str(REPO)},
    ],
)
def test_config_validation_rejects_unsafe_values(mutate):
    with pytest.raises(GateError):
        validate_config(mutate(load_config()))


# --- Gates -----------------------------------------------------------------

def test_checkpoint_gate_passes_and_tamper_fails():
    config = load_config()
    assert check_checkpoint_gate(config) == config["baseline_checkpoint_sha256"]
    tampered = {**config, "baseline_checkpoint_sha256": "0" * 64}
    with pytest.raises(GateError, match="checkpoint gate"):
        check_checkpoint_gate(tampered)


def test_curriculum_gates_pass_and_tamper_fails():
    config = load_config()
    report = check_curriculum_gates(config)
    assert report["train_examples"] == 156
    assert report["validation_examples"] == 52
    tampered = {**config, "curriculum_digest": "0" * 64}
    with pytest.raises(GateError, match="curriculum digest gate"):
        check_curriculum_gates(tampered)


def test_fixture_gates_pass_and_tamper_fails():
    config = load_config()
    fixtures = check_fixture_gates(config)
    assert fixtures["ad1"]["cases"] == 29
    assert fixtures["c3"]["cases"] == 54
    tampered = {
        **config,
        "frozen_fixtures": {
            **config["frozen_fixtures"],
            "c3": {**config["frozen_fixtures"]["c3"], "sha256": "0" * 64},
        },
    }
    with pytest.raises(GateError, match="fixture gate"):
        check_fixture_gates(tampered)


def test_memory_gate_thresholds():
    assert check_memory_gate(0.5, available=0.6) == 0.6
    with pytest.raises(GateError, match="memory gate"):
        check_memory_gate(0.5, available=0.4)
    with pytest.raises(GateError, match="memory gate"):
        check_memory_gate(0.5, available=None)


def test_output_gate_requires_new_run_directory(tmp_path):
    config = {**load_config(), "experiment_root": str(tmp_path)}
    assert check_output_gate(config) == tmp_path / config["run_id"]
    (tmp_path / config["run_id"]).mkdir()
    with pytest.raises(GateError, match="output gate"):
        check_output_gate(config)


def test_training_requires_explicit_authorization(tmp_path):
    config = {**load_config(), "experiment_root": str(tmp_path)}
    with pytest.raises(GateError, match="authorization gate"):
        run_preflight(config, authorized=False, for_training=True)
    # Dry runs do not require authorization and write nothing.
    report = run_preflight(config, authorized=False, for_training=False)
    assert report["mode"] == "dry_run"
    assert not (tmp_path / config["run_id"]).exists()


# --- Promotion -------------------------------------------------------------

def test_promotion_accepts_equal_or_better_candidate():
    baseline = _summary(24, 24, 0)
    candidate = _summary(24, 24, 1)
    assert decide_promotion(baseline, candidate)["promote"] is True


def test_promotion_rejects_parse_regression():
    baseline = _summary(24, 24, 0)
    candidate = _summary(15, 24, 0)
    decision = decide_promotion(baseline, candidate)
    assert decision["promote"] is False
    assert any("parse rate fell" in reason for reason in decision["reasons"])


def test_promotion_rejects_functional_regression():
    baseline = _summary(24, 24, 1)
    candidate = _summary(24, 24, 0)
    decision = decide_promotion(baseline, candidate)
    assert decision["promote"] is False
    assert any("functional accuracy fell" in reason for reason in decision["reasons"])


def test_promotion_rejects_mode_collapse():
    baseline = _summary(24, 24, 0)
    candidate = _summary(24, 24, 1, collapsed=True)
    decision = decide_promotion(baseline, candidate)
    assert decision["promote"] is False
    assert any("mode collapse" in reason for reason in decision["reasons"])


def test_promotion_rejects_missing_cases_and_hash_failures():
    baseline = _summary(24, 24, 0)
    candidate = _summary(24, 24, 0, case_counts=(29, 40))
    decision = decide_promotion(baseline, candidate, hashes_ok=False)
    assert decision["promote"] is False
    assert any("required evaluation cases missing" in reason for reason in decision["reasons"])
    assert any("hash verification failed" in reason for reason in decision["reasons"])


def test_training_loss_is_never_consulted():
    # Summaries carry no loss field; promotion must not need one.
    baseline = _summary(24, 24, 0)
    candidate = _summary(25, 25, 1)
    assert decide_promotion(baseline, candidate)["promote"] is True


# --- CLI dry runs (never train) --------------------------------------------

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
    assert plan["steps"] == 100
    assert plan["launch_command_after_approval"].startswith("python scripts/run_v120_increment3.py --train --authorize")
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


def test_promoter_rejects_increment1_candidate():
    if not INCREMENT1_EVAL.is_file():
        pytest.skip("Increment 1 evaluation JSON not present")
    result = subprocess.run(
        [sys.executable, str(PROMOTER), "--evaluation", str(INCREMENT1_EVAL)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)["decision"]
    assert decision["promote"] is False
    assert any("parse rate fell" in reason for reason in decision["reasons"])
