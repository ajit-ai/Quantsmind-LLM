"""Increment 10 experiment preparation: identifier-aware training gates.

Preparation only (gates + promotion rules). The runner
``scripts/run_v120_increment10.py`` reuses the existing trainer with the
Increment 9 identifier-supervision sidecar enabled. Note on quality
thresholds: copy-drill stages intentionally teach single exact templates
(that is the lesson), so the template-diversity minimum is relaxed to 1
for this experiment while identifier diversity, leakage, and functional
execution remain strict. Loss is never used as evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.behavior_suite import build_cases as build_supp_cases
from evaluation.behavior_suite import suite_digest as supplemental_digest
from evaluation.diagnostics import COPY_CASES
from evaluation.functional_eval import evaluate_functional
from training.binding_curriculum import (
    build_training_examples as build_binding,
    build_validation_examples as build_binding_val,
    curriculum_digest as binding_digest,
)
from training.copy_curriculum import (
    build_training_examples as build_copy,
    build_validation_examples as build_copy_val,
    copy_curriculum_digest as copy_digest,
)
from training.curriculum_quality import QualityThresholds, check_curriculum
from training.experiment_artifacts import canonical_digest, file_sha256
from training.identifier_supervision import IdentifierSupervisionConfig
from training.v120_experiment import GateError, available_gib, check_memory_gate, check_output_gate

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "v120_increment10.json"

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
    "identifier_supervision",
)

COPY_MINIMUM = 10
SUPP_MINIMUM = 3
SUPP_MIN_BEHAVIORS = 3
AD1_BINDING_LOSS_LIMIT = 1


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
        raise GateError("configuration check failed: device must be cpu")
    if not isinstance(config["min_available_gib"], (int, float)) or config["min_available_gib"] <= 0:
        raise GateError("configuration check failed: min_available_gib must be positive")
    run_id = config["run_id"]
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise GateError(f"configuration check failed: invalid run_id {run_id!r}")
    root = Path(config["experiment_root"]).resolve()
    if root == REPO or REPO in root.parents:
        raise GateError("configuration check failed: experiment root is inside the repository")
    # Identifier supervision config must be valid.
    id_conf = config["identifier_supervision"]
    try:
        IdentifierSupervisionConfig(
            copy_weight=id_conf["copy_weight"],
            propagation_weight=id_conf["propagation_weight"],
            max_weight=id_conf["max_weight"],
            decay=id_conf["decay"],
        )
    except ValueError as exc:
        raise GateError(f"configuration check failed: identifier supervision: {exc}") from exc
    for name in ("binding", "copy"):
        if name not in config["curricula"] or not config["curricula"][name].get("digest"):
            raise GateError(f"configuration check failed: curriculum {name!r} digest missing")
    for name in ("ad1", "c3"):
        if name not in config["frozen_fixtures"]:
            raise GateError(f"configuration check failed: frozen fixture {name!r} is missing")


def identifier_config(config: dict) -> IdentifierSupervisionConfig:
    id_conf = config["identifier_supervision"]
    return IdentifierSupervisionConfig(
        copy_weight=id_conf["copy_weight"],
        propagation_weight=id_conf["propagation_weight"],
        max_weight=id_conf["max_weight"],
        decay=id_conf["decay"],
    )


def experiment_identity(config: dict) -> str:
    return canonical_digest(config)


def combined_training_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding(seed=seed) + build_copy()


def combined_validation_examples(config: dict) -> list:
    seed = config["seed"]
    return build_binding_val(seed=seed + 1) + build_copy_val()


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
    observed_binding = binding_digest()
    if observed_binding != config["curricula"]["binding"]["digest"]:
        raise GateError("curriculum digest gate failed: binding digest mismatch")
    observed_copy = copy_digest()
    if observed_copy != config["curricula"]["copy"]["digest"]:
        raise GateError(
            f"curriculum digest gate failed: copy digest {observed_copy} does not match "
            f"{config['curricula']['copy']['digest']}"
        )
    return {"binding": observed_binding, "copy": observed_copy}


def _heldout_pairs(config: dict) -> set[tuple[str, str]]:
    # Copy probes are excluded by design: their familiar control cases
    # deliberately reuse trained identifier pairs (that is what "familiar"
    # means), and the unfamiliar probes do not occur in any corpus.
    pairs = set()
    for name in ("ad1", "c3"):
        suite = json.loads(Path(config["frozen_fixtures"][name]["path"]).read_text(encoding="utf-8"))
        pairs |= {(c["instruction"], c["signature"]) for c in suite.get("cases", [])}
    pairs |= {(c["instruction"], c["signature"]) for c in build_supp_cases()}
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
    thresholds = QualityThresholds(
        min_templates_per_operation=1,
        max_template_share=0.30,
        max_variant_share_within_operation=1.0,
    )
    report = check_curriculum(train, val, thresholds=thresholds, heldout_pairs=_heldout_pairs(config))
    return {**digests, "quality": report}


def check_functional_gates(config: dict, *, examples=None) -> dict:
    if examples is None:
        examples = build_copy()
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
            f"functional gate failed: {len(failures)}/{len(examples)} copy examples do not "
            f"satisfy their stored input/output relationship; first: {failures[0]}"
        )
    return {"copy_examples_verified": len(examples)}


def check_fixture_gates(config: dict) -> dict:
    results = {}
    for name, fixture in config["frozen_fixtures"].items():
        path = Path(fixture["path"])
        if not path.is_file():
            raise GateError(f"fixture gate failed: {name} suite missing: {path}")
        if name == "supplemental":
            observed = supplemental_digest()
            if observed != fixture["sha256"]:
                raise GateError("fixture gate failed: supplemental digest mismatch")
            cases = len(build_supp_cases())
        elif name == "copy_probes":
            cases = len(COPY_CASES)
            observed = str(cases)
        else:
            observed = file_sha256(path)
            if observed != fixture["sha256"]:
                raise GateError(f"fixture gate failed: {name} hash mismatch")
            suite = json.loads(path.read_text(encoding="utf-8"))
            cases = len(suite.get("cases", []))
        if cases != fixture["cases"]:
            raise GateError(f"fixture gate failed: {name} has {cases} cases, expected {fixture['cases']}")
        results[name] = {"cases": cases}
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
    return (numerator / denominator) if denominator else 0.0


def decide_promotion10(
    *,
    copy_exact: int,
    copy_cases: int = 13,
    supplemental_passes: int,
    supplemental_behaviors: int,
    supplemental_cases: int = 48,
    baseline_supplemental_passes: int = 0,
    baseline_supplemental_behaviors: int = 0,
    baseline_ad1_binding: int,
    candidate_ad1_binding: int,
    baseline_ad1_parses: int,
    candidate_ad1_parses: int,
    baseline_ad1_cases: int = 29,
    baseline_c3_functional: float = 0.0,
    candidate_c3_functional: float = 0.0,
    candidate_collapsed: bool = False,
    hashes_ok: bool = True,
) -> dict:
    """Promotion gates (owner table, predeclared in Increment 9 §9)."""
    reasons: list[str] = []
    if not hashes_ok:
        reasons.append("checkpoint or evaluation hash verification failed")
    if copy_exact < COPY_MINIMUM:
        reasons.append(f"exact identifier-copy probes {copy_exact}/{copy_cases} below the minimum of {COPY_MINIMUM}")
    if supplemental_passes < SUPP_MINIMUM:
        reasons.append(
            f"supplemental functional passes {supplemental_passes}/{supplemental_cases} "
            f"below the minimum of {SUPP_MINIMUM}"
        )
    if supplemental_passes <= baseline_supplemental_passes:
        reasons.append("supplemental functional did not improve over baseline")
    if supplemental_behaviors < SUPP_MIN_BEHAVIORS:
        reasons.append(
            f"supplemental passes span {supplemental_behaviors} behaviours, "
            f"below the minimum of {SUPP_MIN_BEHAVIORS}"
        )
    binding_loss = baseline_ad1_binding - candidate_ad1_binding
    if binding_loss > AD1_BINDING_LOSS_LIMIT:
        reasons.append(f"AD-1 binding loss {binding_loss} exceeds the limit of {AD1_BINDING_LOSS_LIMIT}")
    if candidate_ad1_parses < baseline_ad1_parses:
        reasons.append("AD-1 parse rate fell below baseline")
    if candidate_c3_functional < baseline_c3_functional:
        reasons.append("frozen C3 functional accuracy regressed")
    if candidate_collapsed:
        reasons.append("mode collapse detected in candidate")
    return {
        "promote": not reasons,
        "reasons": reasons,
        "thresholds": {
            "copy_exact_minimum": COPY_MINIMUM,
            "supplemental_minimum": SUPP_MINIMUM,
            "supplemental_min_behaviors": SUPP_MIN_BEHAVIORS,
            "ad1_binding_loss_limit": AD1_BINDING_LOSS_LIMIT,
        },
    }
