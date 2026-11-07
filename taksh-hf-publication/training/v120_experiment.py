"""Increment 3 experiment configuration, pre-training gates, and promotion.

This module is preparation only: nothing here initializes a training loop,
loads a model for training, or writes a checkpoint. The runner
(:mod:`scripts.run_v120_increment3` style entry point) calls
:func:`run_preflight` before any training path and refuses to proceed
unless every gate passes and explicit authorization was supplied.

Promotion (:func:`decide_promotion`) is a pure function over the two
summaries produced by ``scripts/evaluate_v120_increment1.py`` so it can be
tested without loading a model.
"""

from __future__ import annotations

import json
from pathlib import Path

from training.binding_curriculum import (
    build_training_examples,
    build_validation_examples,
    curriculum_digest,
)
from training.curriculum_quality import check_curriculum
from training.experiment_artifacts import canonical_digest, file_sha256

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "v120_increment3.json"

_REQUIRED_KEYS = (
    "increment",
    "architecture",
    "baseline_checkpoint",
    "baseline_checkpoint_sha256",
    "tokenizer",
    "curriculum_digest",
    "experiment_root",
    "run_id",
    "steps",
    "lr",
    "seed",
    "device",
    "min_available_gib",
    "frozen_fixtures",
)


class GateError(ValueError):
    """A pre-training or promotion gate failed."""


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
        raise GateError(
            f"configuration check failed: steps {steps!r} is outside the "
            "Increment 3 bound of 1..100"
        )
    lr = config["lr"]
    if not isinstance(lr, (int, float)) or not (0 < lr <= 1e-4):
        raise GateError(
            f"configuration check failed: lr {lr!r} is outside the conservative "
            "bound (0, 1e-4]"
        )
    if not isinstance(config["seed"], int):
        raise GateError("configuration check failed: seed must be an integer")
    if config["device"] != "cpu":
        raise GateError(
            f"configuration check failed: device {config['device']!r} is not the "
            "configured CPU device for Increment 3"
        )
    if not isinstance(config["min_available_gib"], (int, float)) or config["min_available_gib"] <= 0:
        raise GateError("configuration check failed: min_available_gib must be positive")
    run_id = config["run_id"]
    if not run_id or "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        raise GateError(f"configuration check failed: invalid run_id {run_id!r}")
    root = Path(config["experiment_root"]).resolve()
    if root == REPO or REPO in root.parents:
        raise GateError(
            "configuration check failed: experiment root is inside the repository; "
            "experiments must not write into the source tree"
        )
    fixtures = config["frozen_fixtures"]
    for name in ("ad1", "c3"):
        if name not in fixtures or "path" not in fixtures[name] or "sha256" not in fixtures[name]:
            raise GateError(f"configuration check failed: frozen fixture {name!r} is incomplete")


def experiment_identity(config: dict) -> str:
    """Stable hash of the experiment configuration."""
    return canonical_digest(config)


def available_gib():
    """Available RAM in GiB, or None when it cannot be determined."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 1024**3


_MEASURE_MEMORY = object()


def check_memory_gate(minimum_gib: float, available=_MEASURE_MEMORY) -> float:
    observed = available_gib() if available is _MEASURE_MEMORY else available
    if observed is None:
        raise GateError("memory gate failed: available memory could not be determined; refusing to train")
    if observed < minimum_gib:
        raise GateError(
            f"memory gate failed: available {observed:.2f} GiB is below the "
            f"required {minimum_gib:.2f} GiB"
        )
    return observed


def check_output_gate(config: dict) -> Path:
    directory = Path(config["experiment_root"]).resolve() / config["run_id"]
    if directory.exists():
        raise GateError(
            f"output gate failed: run directory already exists: {directory}; "
            "a run identifier may be used only once"
        )
    return directory


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


def _ad1_pairs(path: str) -> set[tuple[str, str]]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(case["instruction"], case["signature"]) for case in suite.get("cases", [])}


def check_curriculum_gates(config: dict, *, seed: int | None = None) -> dict:
    digest = curriculum_digest()
    if digest != config["curriculum_digest"]:
        raise GateError(
            f"curriculum digest gate failed: observed {digest} does not match "
            f"the approved Increment 2 digest {config['curriculum_digest']}"
        )
    run_seed = config["seed"] if seed is None else seed
    train = build_training_examples(seed=run_seed)
    val = build_validation_examples(seed=run_seed + 1)
    heldout = _ad1_pairs(config["frozen_fixtures"]["ad1"]["path"])
    report = check_curriculum(train, val, heldout_pairs=heldout)
    if len(train) != config.get("train_examples", len(train)):
        raise GateError(
            f"curriculum count gate failed: {len(train)} training examples, "
            f"expected {config['train_examples']}"
        )
    if len(val) != config.get("validation_examples", len(val)):
        raise GateError(
            f"curriculum count gate failed: {len(val)} validation examples, "
            f"expected {config['validation_examples']}"
        )
    return report


def check_fixture_gates(config: dict) -> dict:
    results = {}
    for name, fixture in config["frozen_fixtures"].items():
        path = Path(fixture["path"])
        if not path.is_file():
            raise GateError(f"fixture gate failed: {name} suite missing: {path}")
        observed = file_sha256(path)
        if observed != fixture["sha256"]:
            raise GateError(
                f"fixture gate failed: {name} hash {observed} does not match "
                f"the frozen hash {fixture['sha256']}"
            )
        suite = json.loads(path.read_text(encoding="utf-8"))
        cases = len(suite.get("cases", []))
        if cases != fixture["cases"]:
            raise GateError(
                f"fixture gate failed: {name} has {cases} cases, expected {fixture['cases']}"
            )
        results[name] = {"path": str(path), "sha256": observed, "cases": cases}
    return results


def run_preflight(config: dict, *, authorized: bool = False, for_training: bool = False) -> dict:
    """Run all gates. Dry runs (``for_training=False``) never write anything.

    With ``for_training=True`` every gate must pass and ``authorized`` must
    be true, otherwise :class:`GateError` is raised and no training can
    start. With ``for_training=False`` the report is returned and memory
    is reported as observed (it only blocks an actual training start).
    """
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
        record("baseline_checkpoint_hash", True, observed)
    except GateError as exc:
        record("baseline_checkpoint_hash", False, str(exc))
        if for_training:
            raise

    try:
        report = check_curriculum_gates(config)
        record("curriculum", True, f"digest {config['curriculum_digest']}, {report['train_examples']} train")
    except GateError as exc:
        record("curriculum", False, str(exc))
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


def decide_promotion(
    baseline: dict,
    candidate: dict,
    *,
    expected_ad1_cases: int = 29,
    expected_c3_cases: int = 54,
    hashes_ok: bool = True,
) -> dict:
    """Decide whether a candidate may be promoted over the baseline.

    ``baseline`` and ``candidate`` are the per-checkpoint summaries from
    ``scripts/evaluate_v120_increment1.py`` (keys ``ad1``, ``c3_heldout``,
    ``mode_collapse``). Promotion fails on parse-rate or functional-rate
    regression, detected mode collapse, missing cases, or failed hashes.
    Lower training loss is never consulted here.
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
        elif cand_cases != base_cases:
            reasons.append(f"case count mismatch in {suite}: baseline {base_cases}, candidate {cand_cases}")

    for suite in ("ad1", "c3_heldout"):
        base = baseline[suite]
        cand = candidate[suite]
        if base["cases"] and cand["cases"]:
            base_rate = _rate(base["parses"], base["cases"])
            cand_rate = _rate(cand["parses"], cand["cases"])
            if cand_rate < base_rate:
                reasons.append(
                    f"parse rate fell in {suite}: candidate {cand['parses']}/{cand['cases']} "
                    f"({cand_rate:.1%}) below baseline {base['parses']}/{base['cases']} ({base_rate:.1%})"
                )

    base_func = baseline["c3_heldout"]
    cand_func = candidate["c3_heldout"]
    if not base_func["functional_cases"] or not cand_func["functional_cases"]:
        reasons.append("functional evaluation cases missing in c3_heldout")
    else:
        base_rate = _rate(base_func["functional_passed"], base_func["functional_cases"])
        cand_rate = _rate(cand_func["functional_passed"], cand_func["functional_cases"])
        if cand_rate < base_rate:
            reasons.append(
                f"functional accuracy fell in c3_heldout: candidate "
                f"{cand_func['functional_passed']}/{cand_func['functional_cases']} ({cand_rate:.1%}) "
                f"below baseline {base_func['functional_passed']}/{base_func['functional_cases']} ({base_rate:.1%})"
            )

    for suite in ("ad1", "c3"):
        collapse = candidate.get("mode_collapse", {}).get(suite, {})
        if collapse.get("collapsed"):
            reasons.append(
                f"mode collapse detected in candidate {suite}: {collapse.get('reasons')}"
            )

    return {
        "promote": not reasons,
        "reasons": reasons,
        "baseline": {
            "ad1_parses": baseline["ad1"]["parses"],
            "ad1_cases": baseline["ad1"]["cases"],
            "c3_parses": baseline["c3_heldout"]["parses"],
            "c3_cases": baseline["c3_heldout"]["cases"],
            "c3_functional": f"{baseline['c3_heldout']['functional_passed']}/{baseline['c3_heldout']['functional_cases']}",
        },
        "candidate": {
            "ad1_parses": candidate["ad1"]["parses"],
            "ad1_cases": candidate["ad1"]["cases"],
            "c3_parses": candidate["c3_heldout"]["parses"],
            "c3_cases": candidate["c3_heldout"]["cases"],
            "c3_functional": f"{candidate['c3_heldout']['functional_passed']}/{candidate['c3_heldout']['functional_cases']}",
        },
    }
