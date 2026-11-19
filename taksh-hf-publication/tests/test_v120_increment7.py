"""Focused tests for Increment 7 targeted preparation. No training."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.experiment_artifacts import file_sha256
from training.targeted_curriculum import (
    build_training_examples,
    build_validation_examples,
    novel_pairs,
    novel_singles,
    targeted_digest,
)
from training.v120_experiment import GateError
from training.v120_experiment7 import (
    DEFAULT_CONFIG,
    check_checkpoint_gate,
    check_curriculum_gates,
    check_digest_gates,
    check_fixture_gates,
    check_functional_gates,
    decide_promotion7,
    experiment_identity,
    load_config,
    run_preflight,
    validate_config,
)

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_v120_increment7.py"
PROMOTER = REPO / "scripts" / "promote_v120_candidate7.py"
BASELINE = Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v210\runE\checkpoints\trained.pt")
INCREMENT5_EVAL = Path(
    r"C:\Users\Lenovo\AppData\Local\Temp\opencode\v120\increment5\increment5-functional-100\evaluations\heldout_comparison.json"
)


def _param_names(example) -> list[str]:
    inside = example.signature.split("(", 1)[1].split(")", 1)[0]
    return [part.strip() for part in inside.split(",")]


def test_targeted_curriculum_deterministic_and_complete():
    first = build_training_examples()
    second = build_training_examples()
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert targeted_digest() == targeted_digest()
    assert len(first) == 8 * 16
    assert len(build_validation_examples()) == 8 * 4


def test_novel_identifier_pools_are_large_and_disjoint():
    pairs = novel_pairs()
    singles = novel_singles()
    assert len(set(pairs)) == 48
    assert len(set(singles)) == 48
    names = {name for pair in pairs for name in pair} | set(singles)
    assert len(names) >= 64
    fixed = {"text", "left", "right", "values", "items", "value", "data", "word", "source", "target", "entry"}
    assert names.isdisjoint(fixed)


def test_prompt_variation_and_semantic_separation():
    train = build_training_examples()
    val = build_validation_examples()
    assert {(e.instruction, e.signature) for e in train}.isdisjoint(
        {(e.instruction, e.signature) for e in val}
    )
    train_names = {n for e in train for n in _param_names(e)}
    val_names = {n for e in val for n in _param_names(e)}
    assert train_names.isdisjoint(val_names)
    assert {e.instruction for e in train}.isdisjoint({e.instruction for e in val})


def test_targeted_examples_execute_correctly():
    from evaluation.functional_eval import evaluate_functional

    for example in build_training_examples() + build_validation_examples():
        tests = [
            {"args": list(args), "kwargs": {}, "expected": expected}
            for args, expected in zip(example.metadata["inputs"], example.metadata["expected_outputs"])
        ]
        name = example.signature.split("(")[0].removeprefix("def ").strip()
        result = evaluate_functional(example.signature + "\n" + example.body + "\n", name, tests)
        assert result.passed is True, (example.example_id, result)


def test_config_loads_and_identity_is_reproducible():
    first = load_config()
    second = load_config(DEFAULT_CONFIG)
    assert experiment_identity(first) == experiment_identity(second)
    assert first["steps"] == 100
    assert first["lr"] == 1e-05
    assert first["seed"] == 20261108
    assert first["combined_train_examples"] == 284


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: {**c, "steps": 101},
        lambda c: {**c, "lr": 0},
        lambda c: {**c, "lr": 0.001},
        lambda c: {**c, "device": "cuda"},
        lambda c: {**c, "run_id": "../escape"},
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
    fixtures = check_fixture_gates(config)
    assert fixtures["ad1"]["cases"] == 29
    assert fixtures["c3"]["cases"] == 54
    with pytest.raises(GateError, match="checkpoint gate"):
        check_checkpoint_gate({**config, "baseline_checkpoint_sha256": "0" * 64})
    tampered = {
        **config,
        "curricula": {**config["curricula"], "targeted": {**config["curricula"]["targeted"], "digest": "0" * 64}},
    }
    with pytest.raises(GateError, match="curriculum digest gate"):
        check_digest_gates(tampered)


def test_combined_quality_leakage_and_functional_gates():
    import json as json_module
    from pathlib import Path as _Path

    from training.curriculum_quality import check_curriculum

    config = load_config()
    result = check_curriculum_gates(config)
    assert result["quality"]["train_examples"] == 284
    assert result["quality"]["heldout_overlap"] == 0
    assert check_functional_gates(config)["targeted_examples_verified"] == 128
    # Supplemental suite must not appear in training data.
    from evaluation.behavior_suite import build_cases

    supplemental = {(c["instruction"], c["signature"]) for c in build_cases()}
    trained = {(e.instruction, e.signature) for e in
               [x for x in __import__("training.v120_experiment7", fromlist=["combined_training_examples"]).combined_training_examples(config)]}
    assert trained.isdisjoint(supplemental)


def test_trainer_consumes_combined_curriculum():
    from tokenizers import Tokenizer

    from training.supervised_train import build_batches
    from training.v120_experiment7 import combined_training_examples

    config = load_config()
    tokenizer = Tokenizer.from_file(str(config["tokenizer"]))
    sample = combined_training_examples(config)[:6] + combined_training_examples(config)[-6:]
    batches = build_batches(tokenizer, sample, max_length=256)
    assert len(batches) == len(sample)


def test_training_requires_explicit_authorization(tmp_path):
    config = {**load_config(), "experiment_root": str(tmp_path)}
    with pytest.raises(GateError, match="authorization gate"):
        run_preflight(config, authorized=False, for_training=True)
    report = run_preflight(config, authorized=False, for_training=False)
    assert report["mode"] == "dry_run"
    assert not (tmp_path / config["run_id"]).exists()


# --- Promotion ---------------------------------------------------------------

def _summary(ad1_parses, ad1_binding, supp_passes, supp_passing_behaviors, collapsed=False, cases=(29, 54, 48)):
    ad1_cases, c3_cases, supp_cases = cases
    supp_rate = supp_passes / supp_cases if supp_cases else 0.0
    return {
        "ad1": {"cases": ad1_cases, "parses": ad1_parses, "binding": ad1_binding},
        "c3_heldout": {"cases": c3_cases, "functional_passed": 0, "functional_cases": 45},
        "supplemental": {
            "cases": supp_cases,
            "report": {"functional_rate": supp_rate},
            "by_behavior": {f"b{k}": {"functional_rate": (1.0 if k < supp_passing_behaviors else 0.0)} for k in range(8)},
            "records": [{"functional": {"passed": True}} for _ in range(supp_passes)]
            + [{"functional": {"passed": False}} for _ in range(supp_cases - supp_passes)],
        },
        "mode_collapse": {"ad1": {"collapsed": collapsed, "reasons": []}, "c3": {"collapsed": collapsed, "reasons": []}},
    }


def test_promotion_accepts_broad_generalization():
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 5, 4, 3)
    assert decide_promotion7(baseline, candidate)["promote"] is True


def test_promotion_rejects_single_template_pass():
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 6, 1, 1)
    decision = decide_promotion7(baseline, candidate)
    assert decision["promote"] is False
    assert any("minimum" in r for r in decision["reasons"])


def test_promotion_rejects_ad1_binding_regression():
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 2, 6, 4)
    decision = decide_promotion7(baseline, candidate)
    assert decision["promote"] is False
    assert any("AD-1 binding loss" in r for r in decision["reasons"])


def test_promotion_rejects_no_supplemental_improvement():
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 6, 0, 0)
    assert decide_promotion7(baseline, candidate)["promote"] is False


def test_promotion_rejects_collapse_and_uses_hash_flag():
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 6, 6, 4, collapsed=True)
    assert decide_promotion7(baseline, candidate)["promote"] is False
    good = _summary(24, 6, 6, 4)
    assert decide_promotion7(baseline, good, hashes_ok=False)["promote"] is False


# --- CLI dry runs (never train) -----------------------------------------------

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
    assert plan["combined_train_examples"] == 284
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
    baseline = _summary(24, 6, 0, 0)
    candidate = _summary(24, 5, 4, 3)
    payload = {
        "baseline_checkpoint": str(BASELINE),
        "candidate_checkpoint": str(BASELINE),
        "summary": {"baseline": baseline, "candidate": candidate},
    }
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(PROMOTER), "--evaluation", str(path)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["decision"]["promote"] is True
