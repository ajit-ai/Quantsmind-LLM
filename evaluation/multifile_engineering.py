"""Validate the multi-file engineering contract without implementing it.

This module reads the contract and capability-evaluation evidence. It never
mutates a repository, runs tests, trains, creates a checkpoint or publishes
anything. Multi-file engineering remains unimplemented.

Run:

    python -m evaluation.multifile_engineering
    python -m evaluation.multifile_engineering --json
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

CONTRACT = "configs/roadmap/multifile_engineering_contract.yaml"
DEFAULT_OUT = "reports/taksh_multifile_engineering.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


class MultifileEngineeringError(RuntimeError):
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
        raise MultifileEngineeringError(f"contract not found: {contract_path}")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise MultifileEngineeringError(f"contract is not a mapping: {contract_path}")
    return document


def _validate(contract: dict[str, Any]) -> None:
    errors: list[str] = []
    if contract.get("schema") != "taksh.multifile_engineering_contract/1":
        errors.append(f"unexpected schema: {contract.get('schema')!r}")

    policy = contract.get("policy") or {}
    for key in (
        "multifile_editing_supported",
        "fail_closed",
        "partial_apply_allowed",
        "rollback_required",
        "approval_required",
        "sandbox_required",
        "capability_required",
        "owner_authorization_required",
    ):
        if not isinstance(policy.get(key), bool):
            errors.append(f"policy.{key} must be a boolean")

    if policy.get("multifile_editing_supported") is not False:
        errors.append("multifile_editing_supported must be false until implemented")
    if policy.get("fail_closed") is not True:
        errors.append("fail_closed must be true")
    if policy.get("partial_apply_allowed") is not False:
        errors.append("partial_apply_allowed must be false")
    for key in ("rollback_required", "approval_required", "sandbox_required", "capability_required", "owner_authorization_required"):
        if policy.get(key) is not True:
            errors.append(f"policy.{key} must be true")

    spec = contract.get("contract") or {}
    workflows = spec.get("workflows") or []
    if not workflows:
        errors.append("contract.workflows must be non-empty")
    for required in ("apply", "validation", "rollback"):
        if required not in spec:
            errors.append(f"contract.{required} is required")
    apply_spec = spec.get("apply") or {}
    if apply_spec.get("requires_approval") is not True:
        errors.append("apply.requires_approval must be true")
    if apply_spec.get("requires_sandbox") is not True:
        errors.append("apply.requires_sandbox must be true")
    if apply_spec.get("no_partial_apply") is not True:
        errors.append("apply.no_partial_apply must be true")
    validation = spec.get("validation") or {}
    if validation.get("tests_required") is not True:
        errors.append("validation.tests_required must be true")
    if validation.get("no_network") is not True:
        errors.append("validation.no_network must be true")
    rollback = spec.get("rollback") or {}
    if rollback.get("required") is not True:
        errors.append("rollback.required must be true")
    if rollback.get("automatic_on_validation_failure") is not True:
        errors.append("rollback.automatic_on_validation_failure must be true")
    forbidden = spec.get("forbidden") or []
    if "unapproved_patch_application" not in forbidden:
        errors.append("forbidden must include unapproved_patch_application")

    if errors:
        raise MultifileEngineeringError("; ".join(errors))


def _capability_report() -> dict[str, Any]:
    try:
        from evaluation.capability_evaluation import inspect as inspect_capability

        return inspect_capability()
    except Exception as exc:  # noqa: BLE001
        return {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}


def _owner_authorization() -> tuple[str, dict[str, Any]]:
    try:
        from evaluation.l120_training_authorization import load_packet

        packet = load_packet()
        decisions = {
            str(decision.get("id")): decision.get("status")
            for decision in packet.get("owner_decisions", [])
        }
        granted = decisions.get("D-EVAL-5") == "GRANTED"
        status = "PASS" if granted else "BLOCKED"
        return status, {
            "decision": "D-EVAL-5",
            "status": decisions.get("D-EVAL-5", "NOT GRANTED"),
        }
    except Exception as exc:  # noqa: BLE001
        return "BLOCKED", {"error": f"{type(exc).__name__}: {exc}"}


def inspect(path: str | Path = CONTRACT) -> dict[str, Any]:
    """Return deterministic contract evidence; no repository is mutated."""
    contract = load_contract(path)
    _validate(contract)

    capability_report = _capability_report()
    owner_status, owner_detail = _owner_authorization()
    capability_status = str(capability_report.get("status", "BLOCKED"))

    gates = {
        "contract_schema": "PASS",
        "fail_closed_policy": "PASS",
        "approval_policy": "PASS",
        "rollback_policy": "PASS",
        "implementation": "BLOCKED",
        "sandbox_approval": "BLOCKED",
        "capability_prerequisite": capability_status if capability_status == "PASS" else "BLOCKED",
        "owner_authorization": owner_status,
    }

    overall = "PARTIAL" if gates["contract_schema"] == "PASS" else "BLOCKED"
    if overall not in ALLOWED_STATUSES:
        raise MultifileEngineeringError(f"invalid overall status: {overall}")

    blockers = [
        "implementation: no multi-file apply workflow is implemented",
        "sandbox_approval: no sandbox approval recorded",
    ]
    if capability_status != "PASS":
        blockers.append("capability_prerequisite: capability is not evaluated")
    if owner_status != "PASS":
        blockers.append("owner_authorization: multi-file engineering is not authorized")

    return {
        "schema": "taksh.multifile_engineering_contract_report/1",
        "generated_by": "evaluation/multifile_engineering.py",
        "contract_manifest": _relative(_path(path)),
        "status": overall,
        "verdict": "MULTIFILE_ENGINEERING_NOT_IMPLEMENTED",
        "multifile_editing_supported": False,
        "ready": False,
        "policy": contract["policy"],
        "contract": contract["contract"],
        "prerequisites": {
            "capability_evaluation": {
                "status": capability_status,
                "verdict": capability_report.get("verdict"),
            },
            "owner_authorization": owner_detail,
        },
        "gates": gates,
        "blockers": blockers,
        "policy_result": {
            "no_repository_mutation": True,
            "no_unapproved_apply": True,
            "no_partial_side_effects": True,
            "no_capability_claim": True,
            "no_publication": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=CONTRACT,
        help=f"multi-file engineering contract (default: {CONTRACT})",
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
        print(f"multi-file editing supported: {report['multifile_editing_supported']}")
        print("gates:")
        for name, status in report["gates"].items():
            print(f"  {status:<10} {name}")
        print(f"report: {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
