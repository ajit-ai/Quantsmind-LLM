"""Validate the capability-evaluation contract without evaluating a model.

This module reads the contract, training-authorization and frozen-benchmark
evidence. It never loads a model, runs a benchmark, trains, creates a
checkpoint or publishes anything. Capability remains NOT EVALUATED.

Run:

    python -m evaluation.capability_evaluation
    python -m evaluation.capability_evaluation --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CONTRACT = "configs/roadmap/capability_evaluation_contract.yaml"
DEFAULT_OUT = "reports/taksh_capability_evaluation.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


class CapabilityEvaluationError(RuntimeError):
    """The contract is malformed or claims an unauthorized capability."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_contract(path: str | Path = CONTRACT) -> dict[str, Any]:
    contract_path = _path(path)
    if not contract_path.is_file():
        raise CapabilityEvaluationError(f"contract not found: {contract_path}")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise CapabilityEvaluationError(f"contract is not a mapping: {contract_path}")
    return document


def _validate(contract: dict[str, Any]) -> None:
    errors: list[str] = []
    if contract.get("schema") != "taksh.capability_evaluation_contract/1":
        errors.append(f"unexpected schema: {contract.get('schema')!r}")

    policy = contract.get("policy") or {}
    for key in (
        "evaluation_executed",
        "capability_claimed",
        "fail_closed",
        "partial_scoring_allowed",
        "silent_skip_allowed",
        "requires_canonical_checkpoint",
        "requires_frozen_benchmarks",
        "requires_owner_authorization",
        "requires_raw_outputs",
        "preserves_historical_results",
    ):
        if not isinstance(policy.get(key), bool):
            errors.append(f"policy.{key} must be a boolean")

    if policy.get("evaluation_executed") is not False:
        errors.append("evaluation_executed must be false until a run occurs")
    if policy.get("capability_claimed") is not False:
        errors.append("capability_claimed must be false")
    if policy.get("fail_closed") is not True:
        errors.append("fail_closed must be true")
    if policy.get("partial_scoring_allowed") is not False:
        errors.append("partial_scoring_allowed must be false")
    if policy.get("silent_skip_allowed") is not False:
        errors.append("silent_skip_allowed must be false")
    for key in (
        "requires_canonical_checkpoint",
        "requires_frozen_benchmarks",
        "requires_owner_authorization",
        "requires_raw_outputs",
        "preserves_historical_results",
    ):
        if policy.get(key) is not True:
            errors.append(f"policy.{key} must be true")

    spec = contract.get("contract") or {}
    if not spec.get("supported_benchmarks"):
        errors.append("contract.supported_benchmarks must be non-empty")
    scoring = spec.get("scoring") or {}
    if scoring.get("raw_outputs_required") is not True:
        errors.append("raw outputs must be required")
    if scoring.get("score_without_raw_outputs") is not False:
        errors.append("scoring without raw outputs must be prohibited")
    if scoring.get("reuse_historical_score") is not False:
        errors.append("historical scores must not be reused")
    if not spec.get("fail_closed_rules"):
        errors.append("fail_closed_rules must be non-empty")

    if errors:
        raise CapabilityEvaluationError("; ".join(errors))


def _training_report() -> dict[str, Any]:
    try:
        from evaluation.l120_training_authorization import inspect as inspect_training

        return inspect_training()
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}


def _frozen_report() -> dict[str, Any]:
    try:
        from evaluation.frozen_benchmarks import inspect as inspect_frozen

        return inspect_frozen()
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}


def _owner_authorization(training_report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    packet_decisions = {
        str(decision.get("id")): decision.get("status")
        for decision in training_report.get("owner_decisions", [])
    }
    granted = packet_decisions.get("D-EVAL-5") == "GRANTED"
    authorization_flag = bool(
        training_report.get("authorization", {}).get("capability_evaluation_authorized")
    )
    status = "PASS" if granted and authorization_flag else "BLOCKED"
    return status, {
        "decision": "D-EVAL-5",
        "status": packet_decisions.get("D-EVAL-5", "NOT GRANTED"),
        "authorization_flag": authorization_flag,
    }


def inspect(path: str | Path = CONTRACT) -> dict[str, Any]:
    """Return deterministic capability-evaluation evidence; no model is run."""
    contract = load_contract(path)
    _validate(contract)

    training_report = _training_report()
    frozen_report = _frozen_report()
    owner_status, owner_detail = _owner_authorization(training_report)

    checkpoint_status = str(
        training_report.get("gates", {}).get("checkpoint", {}).get("status", "BLOCKED")
    )
    model_status = str(
        training_report.get("gates", {}).get("model", {}).get("status", "BLOCKED")
    )
    frozen_status = str(frozen_report.get("status", "BLOCKED"))

    gates = {
        "contract_schema": "PASS",
        "fail_closed_policy": "PASS",
        "checkpoint_prerequisite": checkpoint_status,
        "model_identity_prerequisite": model_status,
        "frozen_benchmark_prerequisite": frozen_status,
        "owner_authorization": owner_status,
        "implementation": "BLOCKED",
        "historical_result_preservation": "PASS",
        "raw_output_policy": "PASS",
    }

    overall = "PARTIAL" if gates["contract_schema"] == "PASS" else "BLOCKED"
    if overall not in ALLOWED_STATUSES:
        raise CapabilityEvaluationError(f"invalid overall status: {overall}")

    blockers = []
    if checkpoint_status != "PASS":
        blockers.append("checkpoint_prerequisite: no canonical checkpoint")
    if model_status != "PASS":
        blockers.append("model_identity_prerequisite: model contract not verified")
    if frozen_status != "PASS":
        blockers.append("frozen_benchmark_prerequisite: frozen benchmarks absent or unverified")
    if owner_status != "PASS":
        blockers.append("owner_authorization: capability evaluation not authorized")
    blockers.append("implementation: no capability evaluation has been executed")

    return {
        "schema": "taksh.capability_evaluation_contract_report/1",
        "generated_by": "evaluation/capability_evaluation.py",
        "contract_manifest": _relative(_path(path)),
        "status": overall,
        "verdict": "CAPABILITY_NOT_EVALUATED",
        "evaluation_executed": False,
        "capability_claimed": False,
        "score": None,
        "raw_outputs": None,
        "policy": contract["policy"],
        "contract": contract["contract"],
        "prerequisites": {
            "canonical_checkpoint": {
                "status": checkpoint_status,
                "target": training_report.get("gates", {})
                .get("checkpoint", {})
                .get("target"),
                "exists": training_report.get("gates", {})
                .get("checkpoint", {})
                .get("exists"),
            },
            "model_identity": {
                "status": model_status,
                "id": training_report.get("gates", {}).get("model", {}).get("id"),
                "parameters": training_report.get("gates", {})
                .get("model", {})
                .get("parameters"),
            },
            "frozen_benchmarks": {
                "status": frozen_status,
                "absent": frozen_report.get("absent", []),
                "hash_mismatches": frozen_report.get("hash_mismatches", []),
            },
            "owner_authorization": owner_detail,
        },
        "gates": gates,
        "blockers": blockers,
        "policy_result": {
            "no_evaluation_executed": True,
            "no_capability_claim": True,
            "no_partial_score": True,
            "historical_results_preserved": True,
            "no_publication": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=CONTRACT,
        help=f"capability-evaluation contract (default: {CONTRACT})",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = inspect(args.contract)
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"status: {report['status']}")
        print(f"verdict: {report['verdict']}")
        print(f"evaluation executed: {report['evaluation_executed']}")
        print(f"capability claimed: {report['capability_claimed']}")
        print("gates:")
        for name, status in report["gates"].items():
            print(f"  {status:<10} {name}")
        print(f"report: {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
