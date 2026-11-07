"""Focused tests for Increment 5 preparation: gates and functional promotion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.experiment_artifacts import file_sha256
from training.v120_experiment import GateError
from training.v120_experiment5 import (
    DEFAULT_CONFIG,
    check_checkpoint_gate,
    check_curriculum_gates,
    check_digest_gates,
    check_fixture_gates,
    check_functional_gates,
    combined_training_examples,
    combined_validation_examples,
    decide_promotion5,
    experiment_identity,
    load_config,
    run_preflight,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_v120_increment5.py"
PROMOTER = REPO / "scripts" / "promote_v120_candidate5.py"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
INCREMENT3_EVAL = Path(
    r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment3\increment3-diverse-100\evaluations\heldout_comparison.json"
)


def _summary(ad1_parses, functional_rate, functional_given_binding, collapsed=False, cases=(29, 54)):
    ad1_cases, c3_cases = cases
    functional = {
        "functional_rate": functional_rate,
        "functional_given_binding": functional_given_binding,
    }
    return {
        "ad1": {"cases": ad1_cases, "parses": ad1_parses},
        "c3_heldout": {"cases": c3_cases, "parses": 24, "functional_passed": 0, "functional_cases": 45},
        "functional_analysis": {"c3": functional},
        "mode_collapse": {
            "ad1": {"collapsed": collapsed, "reasons": []},
            "c3": {"collapsed": collapsed, "reasons": []},
        },
    }


def test_config_loads_and_identity_is_reproducible():
    first = load_config()
    second = load_config(DEFAULT_CONFIG)
    assert experiment_identity(first) == experiment_identity(second)
    assert first["steps"] == 100
    assert first["lr"] == 1e-05
    assert first["seed"] == 20261107
    assert first["device"] == "cpu"
    assert first["combined_train_examples"] == 252
    assert first["combined_validation_examples"] == 84


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "steps": 101},
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


def test_checkpoint_gate_passes_and_tamper_fails():
    config = load_config()
    assert check_checkpoint_gate(config) == config["baseline_checkpoint_sha256"]
    tampered = {**config, "baseline_checkpoint_sha256": "0" * 64}
    with pytest.raises(GateError, match="checkpoint gate"):
        check_checkpoint_gate(tampered)


def test_digest_gates_pass_and_tamper_fails():
    config = load_config()
    digests = check_digest_gates(config)
    assert digests["binding"] == "bc533a187a631e46d5efc03603c40d604ebddea430a8eb8dd5b54f5a8dc07b7b"
    assert digests["functional"] == "8c3f8d10c37a1f64add826bab759a4c9688ee4a7ec9661cfc65614ee01db0b60"
    tampered = {**config, "curricula": {**config["curricula"], "functional": {**config["curricula"]["functional"], "digest": "0" * 64}}}
    with pytest.raises(GateError, match="curriculum digest gate"):
        check_digest_gates(tampered)


def test_curriculum_gates_combined_pass():
    config = load_config()
    result = check_curriculum_gates(config)
    assert result["quality"]["train_examples"] == 252
    assert result["quality"]["validation_examples"] == 84
    assert result["quality"]["heldout_overlap"] == 0


def test_functional_gate_passes_and_tamper_fails():
    config = load_config()
    assert check_functional_gates(config)["functional_examples_verified"] == 96
    from training.functional_curriculum import build_training_examples

    examples = build_training_examples()
    broken = examples[0]
    broken.metadata = {**broken.metadata, "expected_outputs": ["wrong-value"]}
    with pytest.raises(GateError, match="functional gate"):
        check_functional_gates(config, examples=[broken])


def test_fixture_gates_pass_and_tamper_fails():
    config = load_config()
    fixtures = check_fixture_gates(config)
    assert fixtures["ad1"]["cases"] == 29
    assert fixtures["c3"]["cases"] == 54
    tampered = {
        **config,
        "frozen_fixtures": {
            **config["frozen_fixtures"],
            "ad1": {**config["frozen_fixtures"]["ad1"], "sha256": "0" * 64},
        },
    }
    with pytest.raises(GateError, match="fixture gate"):
        check_fixture_gates(tampered)


def test_trainer_consumes_combined_curriculum():
    from tokenizers import Tokenizer

    from training.supervised_train import build_batches

    config = load_config()
    tokenizer = Tokenizer.from_file(str(config["tokenizer"]))
    sample = combined_training_examples(config)[:6] + combined_training_examples(config)[-6:]
    val_sample = combined_validation_examples(config)[:4]
    # Tokenization only: no training loop is initialized here.
    train_batches = build_batches(tokenizer, sample, max_length=256)
    val_batches = build_batches(tokenizer, val_sample, max_length=256)
    assert len(train_batches) == len(sample)
    assert len(val_batches) == len(val_sample)
    assert all(example.full_text and example.prompt_text and example.target_text for example in sample)


def test_training_requires_explicit_authorization(tmp_path):
    config = {**load_config(), "experiment_root": str(tmp_path)}
    with pytest.raises(GateError, match="authorization gate"):
        run_preflight(config, authorized=False, for_training=True)
    report = run_preflight(config, authorized=False, for_training=False)
    assert report["mode"] == "dry_run"
    assert not (tmp_path / config["run_id"]).exists()


# --- Promotion (functional-first) ------------------------------------------

def test_promotion_requires_functional_improvement():
    baseline = _summary(24, 0.0, 0.0)
    candidate = _summary(24, 0.0, 0.0)
    decision = decide_promotion5(baseline, candidate)
    assert decision["promote"] is False
    assert any("functional accuracy did not improve" in r for r in decision["reasons"])


def test_promotion_requires_conditional_improvement():
    baseline = _summary(24, 0.0, 0.0)
    candidate = _summary(24, 0.02, 0.0)
    decision = decide_promotion5(baseline, candidate)
    assert decision["promote"] is False
    assert any("conditional on binding" in r for r in decision["reasons"])


def test_promotion_accepts_functional_improvement():
    baseline = _summary(24, 0.0, 0.0)
    candidate = _summary(24, 0.0222, 0.1)
    assert decide_promotion5(baseline, candidate)["promote"] is True


def test_promotion_rejects_ad1_parse_regression():
    baseline = _summary(24, 0.0, 0.0)
    candidate = _summary(23, 0.0222, 0.1)
    decision = decide_promotion5(baseline, candidate)
    assert decision["promote"] is False
    assert any("AD-1 parse rate fell" in r for r in decision["reasons"])


def test_promotion_rejects_collapse_missing_and_hash_failures():
    baseline = _summary(24, 0.0, 0.0)
    collapsed = _summary(24, 0.0222, 0.1, collapsed=True)
    assert decide_promotion5(baseline, collapsed)["promote"] is False
    missing = _summary(24, 0.0222, 0.1, cases=(29, 40))
    decision = decide_promotion5(baseline, missing, hashes_ok=False)
    assert decision["promote"] is False
    assert any("required evaluation cases missing" in r for r in decision["reasons"])
    assert any("hash verification failed" in r for r in decision["reasons"])


def test_training_loss_is_never_consulted():
    baseline = _summary(24, 0.0, 0.0)
    candidate = _summary(24, 0.0222, 0.1)
    assert decide_promotion5(baseline, candidate)["promote"] is True


# --- CLI dry runs (never train) ---------------------------------------------

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
    assert plan["combined_train_examples"] == 252
    assert plan["launch_command_after_approval"].startswith("python scripts/run_v120_increment5.py --train --authorize")
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


def test_promoter_rejects_increment3_candidate():
    if not INCREMENT3_EVAL.is_file():
        pytest.skip("Increment 3 evaluation JSON not present")
    result = subprocess.run(
        [sys.executable, str(PROMOTER), "--evaluation", str(INCREMENT3_EVAL)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    decision = json.loads(result.stdout)["decision"]
    assert decision["promote"] is False
    reasons = " ".join(decision["reasons"])
    assert "AD-1 parse rate fell" in reasons
