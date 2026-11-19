"""Increment 5 experiment preparation: gates and functional promotion.

Preparation only: this module validates the combined Increment 2 +
Increment 4 curriculum (including executing every functional example's
stored input/output relationship through the isolated safe gate), runs the
ten pre-training gates, and decides promotion with functional correctness
as the primary criterion. It never initializes a training loop.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.functional_eval import evaluate_functional
from training.binding_curriculum import (
    build_training_examples as build_binding,
    build_validation_examples as build_binding_val,
    curriculum_digest as binding_digest,
)
from training.curriculum_quality import check_curriculum
from training.experiment_artifacts import canonical_digest, file_sha256
from training.functional_curriculum import (
    build_training_examples as build_functional,
    build_validation_examples as build_functional_val,
    functional_digest as functional_digest_fn,
)
from training.v120_experiment import (
    GateError,
    available_gib,
    check_memory_gate,
    check_output_gate,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "v120_increment5.json"

_REQUIRED_KEYS = (
    "increment",
    "architecture",
    "baseline_checkpoint",
    "baseline_checkpoint_sha256",
    "tokenizer",
    "curricula",
    "combined_train_examples",
    "combined_validation_examples",
    "experiment_root",
    "run_id",
    "steps",
    "lr",
    "seed",
    "device",
    "min_available_gib",
    "frozen_fixtures",
)


def load_config(path=DEFAULT_CONFIG) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    for key in _REQUIRED_KEYS:
        if key not in config:
            raise GateError(f"configuration check failed: missing key {key!r}")
    steps = config["steps"]
    if not isinstance(steps, int) or not (1 <= steps <= 100):
        raise GateError("configuration check failed: steps must be an integer in 1..100")
    lr = config["lr"]
    if not isinstance(lr, (int, float)) or not (0 < lr <= 1e-4):
        raise GateError("configuration check failed: lr must be in (0, 1e-4]")
    if not isinstance(config["seed"], int):
        raise GateError("configuration check failed: seed must be an integer")
    if config["device"] != "cpu":
        raise GateError("configuration check failed: device must be cpu for Increment 5")
    if not isinstance(config["min_available_gib"], (int, float)) or config["min_available_gib"] <= 0:
        raise GateError("configuration check failed: min_available_gib must be positive")
    run_id = config["run_id"]
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise GateError(f"configuration check failed: invalid run_id {run_id!r}")
    root = Path(config["experiment_root"]).resolve()
    if root == REPO or REPO in root.parents:
        raise GateError("configuration check failed: experiment root is inside the repository")
    curricula = config["curricula"]
    for name in ("binding", "functional"):
        if name not in curricula or "digest" not in curricula[name]:
            raise GateError(f"configuration check failed: curriculum {name!r} is incomplete")
    for name in ("ad1", "c3"):
        if name not in config["frozen_fixtures"]:
            raise GateError(f"configuration check failed: frozen fixture {name!r} is missing")


def experiment_identity(config: dict) -> str:
    return canonical_digest(config)


def combined_training_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding(seed=seed) + build_functional(seed=seed)


def combined_validation_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding_val(seed=seed + 1) + build_functional_val(seed=seed + 1)


def check_checkpoint_gate(config: dict) -> str:
    path = Path(config["baseline_checkpoint"])
    if not path.is_file():
        raise GateError(f"checkpoint gate failed: baseline checkpoint missing: {path}")
    observed = file_sha256(path)
    if observed != config["baseline_checkpoint_sha256"]:
        raise GateError(
            f"checkpoint gate failed: baseline hash {observed} does not match "
            f"the recorded {config['baseline_checkpoint_sha256']}"
        )
    return observed


def check_digest_gates(config: dict) -> dict:
    binding_observed = binding_digest()
    if binding_observed != config["curricula"]["binding"]["digest"]:
        raise GateError(
            f"curriculum digest gate failed: binding digest {binding_observed} does not match "
            f"{config['curricula']['binding']['digest']}"
        )
    functional_observed = functional_digest_fn()
    if functional_observed != config["curricula"]["functional"]["digest"]:
        raise GateError(
            f"curriculum digest gate failed: functional digest {functional_observed} does not match "
            f"{config['curricula']['functional']['digest']}"
        )
    return {"binding": binding_observed, "functional": functional_observed}


def _ad1_pairs(path: str) -> set[tuple[str, str]]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(c["instruction"], c["signature"]) for c in suite.get("cases", [])}


def check_curriculum_gates(config: dict) -> dict:
    digests = check_digest_gates(config)
    train = combined_training_examples(config)
    val = combined_validation_examples(config)
    if len(train) != config["combined_train_examples"]:
        raise GateError(
            f"curriculum count gate failed: {len(train)} combined training examples, "
            f"expected {config['combined_train_examples']}"
        )
    if len(val) != config["combined_validation_examples"]:
        raise GateError(
            f"curriculum count gate failed: {len(val)} combined validation examples, "
            f"expected {config['combined_validation_examples']}"
        )
    heldout = _ad1_pairs(config["frozen_fixtures"]["ad1"]["path"])
    c3_path = config["frozen_fixtures"]["c3"]["path"]
    c3 = json.loads(Path(c3_path).read_text(encoding="utf-8"))
    heldout |= {(c["instruction"], c["signature"]) for c in c3["cases"]}
    report = check_curriculum(train, val, heldout_pairs=heldout)
    return {**digests, "quality": report}


def check_functional_gates(config: dict, *, examples=None) -> dict:
    """Execute every functional training example's I/O relationship safely."""
    if examples is None:
        examples = build_functional(seed=config["seed"])
    failures = []
    for example in examples:
        tests = [
            {"args": list(args), "kwargs": {}, "expected": expected}
            for args, expected in zip(
                example.metadata.get("inputs", []),
                example.metadata.get("expected_outputs", []),
            )
        ]
        if not tests:
            failures.append((example.example_id, "missing input/output metadata"))
            continue
        name = example.signature.split("(")[0].removeprefix("def ").strip()
        result = evaluate_functional(example.signature + "\n" + example.body + "\n", name, tests)
        if not result.passed:
            failures.append((example.example_id, result.category + ": " + result.reason))
    if failures:
        raise GateError(
            f"functional gate failed: {len(failures)}/{len(examples)} functional examples do not "
            f"satisfy their stored input/output relationship; first: {failures[0]}"
        )
    return {"functional_examples_verified": len(examples)}


def check_fixture_gates(config: dict) -> dict:
    results = {}
    for name, fixture in config["frozen_fixtures"].items():
        path = Path(fixture["path"])
        if not path.is_file():
            raise GateError(f"fixture gate failed: {name} suite missing: {path}")
        observed = file_sha256(path)
        if observed != fixture["sha256"]:
            raise GateError(
                f"fixture gate failed: {name} hash {observed} does not match {fixture['sha256']}"
            )
        suite = json.loads(path.read_text(encoding="utf-8"))
        cases = len(suite.get("cases", []))
        if cases != fixture["cases"]:
            raise GateError(f"fixture gate failed: {name} has {cases} cases, expected {fixture['cases']}")
        results[name] = {"path": str(path), "sha256": observed, "cases": cases}
    return results


def run_preflight(config: dict, *, authorized: bool = False, for_training: bool = False) -> dict:
    gates: list[dict] = []

    def record(name, passed, detail):
        gates.append({"gate": name, "passed": bool(passed), "detail": detail})

    if for_training and not authorized:
        raise GateError(
            "authorization gate failed: training requires the explicit "
            "--authorize flag in addition to --train"
        )

    try:
        validate_config(config)
        record("configuration", True, f"identity {experiment_identity(config)}")
    except GateError as exc:
        record("configuration", False, str(exc))
        if for_training:
            raise

    try:
        observed = check_checkpoint_gate(config)
        record("baseline_checkpoint", True, observed)
    except GateError as exc:
        record("baseline_checkpoint", False, str(exc))
        if for_training:
            raise

    try:
        curriculum = check_curriculum_gates(config)
        record(
            "curriculum_digests_quality_separation",
            True,
            f"binding {config['curricula']['binding']['digest']}, "
            f"functional {config['curricula']['functional']['digest']}, "
            f"252 train / 84 validation, leakage 0",
        )
    except GateError as exc:
        record("curriculum_digests_quality_separation", False, str(exc))
        if for_training:
            raise

    try:
        functional = check_functional_gates(config)
        record("functional_relationships", True, f"{functional['functional_examples_verified']} examples verified")
    except GateError as exc:
        record("functional_relationships", False, str(exc))
        if for_training:
            raise

    try:
        fixtures = check_fixture_gates(config)
        record("frozen_fixtures", True, f"AD-1 {fixtures['ad1']['cases']} cases, C3 {fixtures['c3']['cases']} cases")
    except GateError as exc:
        record("frozen_fixtures", False, str(exc))
        if for_training:
            raise

    observed_memory = available_gib()
    try:
        check_memory_gate(config["min_available_gib"], available=observed_memory)
        record("memory", True, f"available {observed_memory:.2f} GiB >= {config['min_available_gib']:.2f} GiB")
    except GateError as exc:
        record("memory", False, str(exc))
        if for_training:
            raise

    try:
        directory = check_output_gate(config)
        record("output_directory", True, f"new run directory {directory}")
    except GateError as exc:
        record("output_directory", False, str(exc))
        if for_training:
            raise

    record("authorization", True, "authorized" if authorized else "not authorized (dry run only)")
    return {
        "mode": "train" if for_training else "dry_run",
        "experiment_identity": experiment_identity(config),
        "gates": gates,
        "passed": all(gate["passed"] for gate in gates if gate["gate"] != "memory") if not for_training
        else all(gate["passed"] for gate in gates),
    }


def _rate(numerator: int, denominator: int):
    return (numerator / denominator) if denominator else None


def _func_rate(side: dict):
    functional = side.get("functional_analysis", {}).get("c3", {})
    if functional.get("functional_rate") is not None:
        return functional["functional_rate"]
    c3 = side.get("c3_heldout", {})
    return _rate(c3.get("functional_passed", 0), c3.get("functional_cases", 0))


def _func_given_binding(side: dict):
    functional = side.get("functional_analysis", {}).get("c3", {})
    if functional.get("functional_given_binding") is not None:
        return functional["functional_given_binding"]
    return None


def decide_promotion5(
    baseline: dict,
    candidate: dict,
    *,
    expected_ad1_cases: int = 29,
    expected_c3_cases: int = 54,
    hashes_ok: bool = True,
) -> dict:
    """Promotion with functional correctness as the primary criterion.

    ``baseline``/``candidate`` are per-checkpoint summaries from
    ``scripts/evaluate_v120_increment1.py`` (keys ``ad1``, ``c3_heldout``,
    ``functional_analysis``, ``mode_collapse``). A baseline conditional
    rate of None (no binding passes) counts as 0.0. Training loss is
    never consulted.
    """
    reasons: list[str] = []
    if not hashes_ok:
        reasons.append("checkpoint or evaluation hash verification failed")

    for suite, expected in (("ad1", expected_ad1_cases), ("c3_heldout", expected_c3_cases)):
        base_cases = baseline[suite]["cases"]
        cand_cases = candidate[suite]["cases"]
        if base_cases != expected or cand_cases != expected:
            reasons.append(
                f"required evaluation cases missing in {suite}: baseline {base_cases}, "
                f"candidate {cand_cases}, expected {expected}"
            )

    base_ad1 = baseline["ad1"]
    cand_ad1 = candidate["ad1"]
    if base_ad1["cases"] and cand_ad1["cases"]:
        base_rate = _rate(base_ad1["parses"], base_ad1["cases"])
        cand_rate = _rate(cand_ad1["parses"], cand_ad1["cases"])
        if cand_rate < base_rate:
            reasons.append(
                f"AD-1 parse rate fell: candidate {cand_ad1['parses']}/{cand_ad1['cases']} "
                f"({cand_rate:.1%}) below baseline {base_ad1['parses']}/{base_ad1['cases']} ({base_rate:.1%})"
            )

    base_func = _func_rate(baseline) or 0.0
    cand_func = _func_rate(candidate) or 0.0
    if cand_func <= base_func:
        reasons.append(
            f"functional accuracy did not improve: candidate {cand_func:.1%} vs baseline {base_func:.1%}"
        )

    base_cond = _func_given_binding(baseline)
    cand_cond = _func_given_binding(candidate)
    base_cond = 0.0 if base_cond is None else base_cond
    cand_cond = 0.0 if cand_cond is None else cand_cond
    if cand_cond <= base_cond:
        reasons.append(
            "functional accuracy conditional on binding did not improve: "
            f"candidate {cand_cond:.1%} vs baseline {base_cond:.1%}"
        )

    for suite in ("ad1", "c3"):
        collapse = candidate.get("mode_collapse", {}).get(suite, {})
        if collapse.get("collapsed"):
            reasons.append(f"mode collapse detected in candidate {suite}: {collapse.get('reasons')}")

    return {
        "promote": not reasons,
        "reasons": reasons,
        "baseline": {
            "ad1_parses": baseline["ad1"]["parses"],
            "ad1_cases": baseline["ad1"]["cases"],
            "c3_functional_rate": base_func,
            "c3_functional_given_binding": base_cond,
        },
        "candidate": {
            "ad1_parses": candidate["ad1"]["parses"],
            "ad1_cases": candidate["ad1"]["cases"],
            "c3_functional_rate": cand_func,
            "c3_functional_given_binding": cand_cond,
        },
    }
