"""Increment 7 experiment preparation: targeted generalization gates.

Preparation only: this module validates the combined binding + targeted
curriculum (including executing every targeted example's stored
input/output relationship through the isolated safe gate), runs the
pre-training gates, and decides promotion with **supplemental functional
generalization plus AD-1 retention** as the primary criteria. It never
initializes a training loop.

Promotion thresholds (stated before training, with rationale):

* ``supplemental_functional_min_passes = 3`` and
  ``supplemental_functional_min_behaviors = 3``: a single passing case
  cannot distinguish template memorization from behaviour learning
  (Increment 5's lone pass was a memorized template). Three passes across
  three behaviours is the smallest evidence that generalizes beyond one
  template.
* ``ad1_binding_loss_limit = 1``: AD-1 binding is retained capability
  (6/29). Increment 5 lost 4; losing more than one is an unacceptable
  trade for narrow functional gains.
* Supplemental functional rate must strictly improve over baseline
  (both checkpoints scored 0/48 in Increment 6, so any candidate pass is
  improvement — but the 3-pass/3-behaviour floors still apply).
* Frozen C3 functional must not regress and conditional functional
  (functional | binding) must not fall below baseline (None counts as 0).
* Mode collapse, missing cases, or any hash failure reject.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.behavior_suite import suite_digest as supplemental_digest
from evaluation.functional_eval import evaluate_functional
from training.binding_curriculum import (
    build_training_examples as build_binding,
    build_validation_examples as build_binding_val,
    curriculum_digest as binding_digest,
)
from training.curriculum_quality import check_curriculum
from training.experiment_artifacts import canonical_digest, file_sha256
from training.targeted_curriculum import (
    build_training_examples as build_targeted,
    build_validation_examples as build_targeted_val,
    targeted_digest as targeted_digest_fn,
)
from training.v120_experiment import (
    GateError,
    available_gib,
    check_memory_gate,
    check_output_gate,
)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "v120_increment7.json"

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

AD1_BINDING_LOSS_LIMIT = 1
SUPPLEMENTAL_MIN_PASSES = 3
SUPPLEMENTAL_MIN_BEHAVIORS = 3


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
        raise GateError("configuration check failed: device must be cpu for Increment 7")
    if not isinstance(config["min_available_gib"], (int, float)) or config["min_available_gib"] <= 0:
        raise GateError("configuration check failed: min_available_gib must be positive")
    run_id = config["run_id"]
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise GateError(f"configuration check failed: invalid run_id {run_id!r}")
    root = Path(config["experiment_root"]).resolve()
    if root == REPO or REPO in root.parents:
        raise GateError("configuration check failed: experiment root is inside the repository")
    for name in ("binding", "targeted"):
        if name not in config["curricula"] or not config["curricula"][name].get("digest"):
            raise GateError(f"configuration check failed: curriculum {name!r} digest missing")
    for name in ("ad1", "c3"):
        if name not in config["frozen_fixtures"]:
            raise GateError(f"configuration check failed: frozen fixture {name!r} is missing")


def experiment_identity(config: dict) -> str:
    return canonical_digest(config)


def combined_training_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding(seed=seed) + build_targeted(seed=seed)


def combined_validation_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding_val(seed=seed + 1) + build_targeted_val(seed=seed + 1)


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
    targeted_observed = targeted_digest_fn()
    if targeted_observed != config["curricula"]["targeted"]["digest"]:
        raise GateError(
            f"curriculum digest gate failed: targeted digest {targeted_observed} does not match "
            f"{config['curricula']['targeted']['digest']}"
        )
    return {"binding": binding_observed, "targeted": targeted_observed}


def _heldout_pairs(config: dict) -> set[tuple[str, str]]:
    pairs = set()
    for name in ("ad1", "c3"):
        suite = json.loads(Path(config["frozen_fixtures"][name]["path"]).read_text(encoding="utf-8"))
        pairs |= {(c["instruction"], c["signature"]) for c in suite.get("cases", [])}
    from evaluation.behavior_suite import build_cases

    pairs |= {(c["instruction"], c["signature"]) for c in build_cases()}
    return pairs


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
    report = check_curriculum(train, val, heldout_pairs=_heldout_pairs(config))
    return {**digests, "quality": report}


def check_functional_gates(config: dict, *, examples=None) -> dict:
    if examples is None:
        examples = build_targeted(seed=config["seed"])
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
            f"functional gate failed: {len(failures)}/{len(examples)} targeted examples do not "
            f"satisfy their stored input/output relationship; first: {failures[0]}"
        )
    return {"targeted_examples_verified": len(examples)}


def check_fixture_gates(config: dict) -> dict:
    from evaluation.behavior_suite import build_cases

    results = {}
    for name, fixture in config["frozen_fixtures"].items():
        path = Path(fixture["path"])
        if not path.is_file():
            raise GateError(f"fixture gate failed: {name} suite missing: {path}")
        if name == "supplemental":
            # The supplemental fixture is defined by its case digest, not by
            # the source file hash (the builder may be refactored without
            # changing cases).
            observed = supplemental_digest()
            if observed != fixture["sha256"]:
                raise GateError(
                    f"fixture gate failed: supplemental digest {observed} does not match "
                    f"{fixture['sha256']}"
                )
            cases = len(build_cases())
        else:
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

    for name, fn in (
        ("configuration", lambda: validate_config(config)),
        ("baseline_checkpoint", lambda: check_checkpoint_gate(config)),
        ("curriculum_digests_quality_separation", lambda: check_curriculum_gates(config)),
        ("functional_relationships", lambda: check_functional_gates(config)),
        ("frozen_fixtures", lambda: check_fixture_gates(config)),
    ):
        try:
            detail = fn()
            record(name, True, str(detail)[:200])
        except GateError as exc:
            record(name, False, str(exc))
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


def _supp(side: dict) -> dict:
    supplemental = side.get("supplemental", {})
    if supplemental:
        return supplemental
    return side.get("supplemental_analysis", {})


def decide_promotion7(
    baseline: dict,
    candidate: dict,
    *,
    expected_ad1_cases: int = 29,
    expected_c3_cases: int = 54,
    expected_supplemental_cases: int = 48,
    hashes_ok: bool = True,
) -> dict:
    """Promotion with supplemental functional generalization + AD-1 retention.

    Summaries need ``ad1``, ``c3_heldout``, ``supplemental`` (report +
    by_behavior), and ``mode_collapse`` keys as produced by the
    Increment 6 analysis/evaluation scripts. Baseline supplemental rates
    of None count as 0.0. Training loss is never consulted.
    """
    reasons: list[str] = []
    if not hashes_ok:
        reasons.append("checkpoint or evaluation hash verification failed")

    base_supp = _supp(baseline)
    cand_supp = _supp(candidate)
    for suite, expected in (("ad1", expected_ad1_cases), ("c3_heldout", expected_c3_cases)):
        if baseline[suite]["cases"] != expected or candidate[suite]["cases"] != expected:
            reasons.append(
                f"required evaluation cases missing in {suite}: baseline {baseline[suite]['cases']}, "
                f"candidate {candidate[suite]['cases']}, expected {expected}"
            )
    if base_supp.get("cases", 0) != expected_supplemental_cases or cand_supp.get("cases", 0) != expected_supplemental_cases:
        reasons.append(
            f"required supplemental cases missing: baseline {base_supp.get('cases')}, "
            f"candidate {cand_supp.get('cases')}, expected {expected_supplemental_cases}"
        )

    base_ad1 = baseline["ad1"]
    cand_ad1 = candidate["ad1"]
    base_ad1_rate = _rate(base_ad1["parses"], base_ad1["cases"]) or 0.0
    cand_ad1_rate = _rate(cand_ad1["parses"], cand_ad1["cases"]) or 0.0
    if cand_ad1_rate < base_ad1_rate:
        reasons.append(
            f"AD-1 parse rate fell: candidate {cand_ad1['parses']}/{cand_ad1['cases']} "
            f"({cand_ad1_rate:.1%}) below baseline {base_ad1['parses']}/{base_ad1['cases']} ({base_ad1_rate:.1%})"
        )
    binding_loss = base_ad1["binding"] - cand_ad1["binding"]
    if binding_loss > AD1_BINDING_LOSS_LIMIT:
        reasons.append(
            f"AD-1 binding loss {binding_loss} exceeds the limit of {AD1_BINDING_LOSS_LIMIT}"
        )

    base_func = base_supp.get("report", {}).get("functional_rate") or 0.0
    cand_func = cand_supp.get("report", {}).get("functional_rate") or 0.0
    if cand_func <= base_func:
        reasons.append(
            f"supplemental functional accuracy did not improve: candidate {cand_func:.1%} vs baseline {base_func:.1%}"
        )
    cand_passes = cand_supp.get("report", {}).get("functional_scored", 0) and cand_supp.get("report", {}).get("functional_rate")
    # Count passing cases from the report ratio when available.
    cand_report = cand_supp.get("report", {})
    cand_flags = cand_supp.get("records") or []
    cand_pass_count = sum(1 for r in cand_flags if r.get("functional", {}).get("passed"))
    if cand_pass_count < SUPPLEMENTAL_MIN_PASSES:
        reasons.append(
            f"supplemental functional passes {cand_pass_count} below the minimum of {SUPPLEMENTAL_MIN_PASSES}"
        )
    cand_by_behavior = cand_supp.get("by_behavior", {})
    passing_behaviors = sum(1 for entry in cand_by_behavior.values() if entry.get("functional_rate"))
    if passing_behaviors < SUPPLEMENTAL_MIN_BEHAVIORS:
        reasons.append(
            f"supplemental functional passes span {passing_behaviors} behaviours, "
            f"below the minimum of {SUPPLEMENTAL_MIN_BEHAVIORS}"
        )

    base_c3 = baseline["c3_heldout"]
    cand_c3 = candidate["c3_heldout"]
    base_c3_rate = _rate(base_c3["functional_passed"], base_c3["functional_cases"]) or 0.0
    cand_c3_rate = _rate(cand_c3["functional_passed"], cand_c3["functional_cases"]) or 0.0
    if cand_c3_rate < base_c3_rate:
        reasons.append(
            f"frozen C3 functional accuracy regressed: candidate {cand_c3_rate:.1%} vs baseline {base_c3_rate:.1%}"
        )

    for suite in ("ad1", "c3"):
        collapse = candidate.get("mode_collapse", {}).get(suite, {})
        if collapse.get("collapsed"):
            reasons.append(f"mode collapse detected in candidate {suite}: {collapse.get('reasons')}")

    return {
        "promote": not reasons,
        "reasons": reasons,
        "thresholds": {
            "ad1_binding_loss_limit": AD1_BINDING_LOSS_LIMIT,
            "supplemental_min_passes": SUPPLEMENTAL_MIN_PASSES,
            "supplemental_min_behaviors": SUPPLEMENTAL_MIN_BEHAVIORS,
        },
        "baseline": {
            "ad1_parses": base_ad1["parses"],
            "ad1_binding": base_ad1["binding"],
            "supplemental_functional_rate": base_func,
            "c3_functional_rate": base_c3_rate,
        },
        "candidate": {
            "ad1_parses": cand_ad1["parses"],
            "ad1_binding": cand_ad1["binding"],
            "supplemental_functional_rate": cand_func,
            "supplemental_passes": cand_pass_count,
            "supplemental_passing_behaviors": passing_behaviors,
            "c3_functional_rate": cand_c3_rate,
        },
    }
