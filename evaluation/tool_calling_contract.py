"""Validate the fail-closed tool-calling contract without implementing it.

This module reads the contract manifest and the serving agent profile. It never
executes a tool, starts a server, trains, creates a checkpoint or publishes
anything. Tool calling remains unsupported.

Run:

    python -m evaluation.tool_calling_contract
    python -m evaluation.tool_calling_contract --json
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

CONTRACT = "configs/roadmap/tool_calling_contract.yaml"
DEFAULT_OUT = "reports/taksh_tool_calling_contract.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


class ToolCallingContractError(RuntimeError):
    """The contract manifest is malformed or claims a prohibited capability."""


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
        raise ToolCallingContractError(f"contract not found: {contract_path}")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ToolCallingContractError(f"contract is not a mapping: {contract_path}")
    return document


def _validate(contract: dict[str, Any]) -> None:
    errors: list[str] = []
    if contract.get("schema") != "taksh.tool_calling_contract/1":
        errors.append(f"unexpected schema: {contract.get('schema')!r}")

    policy = contract.get("policy") or {}
    for key in (
        "tool_calling_supported",
        "fail_closed",
        "partial_execution_allowed",
        "silent_ignore_allowed",
        "requires_canonical_checkpoint",
        "requires_capability_evaluation",
        "requires_owner_authorization",
        "requires_sandbox_approval",
    ):
        if not isinstance(policy.get(key), bool):
            errors.append(f"policy.{key} must be a boolean")

    if policy.get("tool_calling_supported") is not False:
        errors.append("tool_calling_supported must be false until implemented")
    if policy.get("fail_closed") is not True:
        errors.append("fail_closed must be true")
    if policy.get("partial_execution_allowed") is not False:
        errors.append("partial_execution_allowed must be false")
    if policy.get("silent_ignore_allowed") is not False:
        errors.append("silent_ignore_allowed must be false")
    for key in (
        "requires_canonical_checkpoint",
        "requires_capability_evaluation",
        "requires_owner_authorization",
        "requires_sandbox_approval",
    ):
        if policy.get(key) is not True:
            errors.append(f"policy.{key} must be true")

    spec = contract.get("contract") or {}
    if not isinstance(spec.get("request_fields"), list) or not spec["request_fields"]:
        errors.append("contract.request_fields must be a non-empty list")
    rejection = spec.get("rejection") or {}
    if rejection.get("http_status") != 400:
        errors.append("rejection.http_status must be 400")
    if rejection.get("error_type") != "invalid_request_error":
        errors.append("rejection.error_type must be invalid_request_error")
    if not rejection.get("code"):
        errors.append("rejection.code is required")
    if spec.get("allowed_tools") != []:
        errors.append("allowed_tools must be empty until a tool is implemented")
    sandbox = spec.get("sandbox") or {}
    if sandbox.get("network") != "disabled":
        errors.append("sandbox.network must be disabled")
    if sandbox.get("filesystem") != "jailed":
        errors.append("sandbox.filesystem must be jailed")
    if not isinstance(sandbox.get("timeout_seconds"), int) or sandbox["timeout_seconds"] <= 0:
        errors.append("sandbox.timeout_seconds must be positive")
    if sandbox.get("no_untrusted_code_execution") is not True:
        errors.append("sandbox.no_untrusted_code_execution must be true")
    if spec.get("approval_required_for_mutation") is not True:
        errors.append("mutation approval must be required")

    if errors:
        raise ToolCallingContractError("; ".join(errors))


def _server_profile_alignment() -> tuple[bool, dict[str, Any]]:
    from serving.agent_profile import default_profile

    profile = default_profile()
    payload = profile.to_dict()
    aligned = (
        payload.get("tool_calling_supported") is False
        and str(payload.get("tool_calling_status", "")).startswith("unsupported")
    )
    return aligned, {
        "tool_calling_supported": payload.get("tool_calling_supported"),
        "tool_calling_status": payload.get("tool_calling_status"),
    }


def inspect(path: str | Path = CONTRACT) -> dict[str, Any]:
    """Return deterministic contract evidence; never implements tool calling."""
    contract = load_contract(path)
    _validate(contract)

    policy = contract["policy"]
    spec = contract["contract"]
    aligned, profile = _server_profile_alignment()

    gates = {
        "contract_schema": "PASS",
        "fail_closed_policy": "PASS",
        "server_profile_alignment": "PASS" if aligned else "BLOCKED",
        "implementation": "BLOCKED",
        "sandbox_approval": "BLOCKED",
        "capability_prerequisite": "BLOCKED",
        "owner_authorization": "BLOCKED",
    }
    overall = "PARTIAL" if gates["contract_schema"] == "PASS" else "BLOCKED"
    if overall not in ALLOWED_STATUSES:
        raise ToolCallingContractError(f"invalid overall status: {overall}")

    return {
        "schema": "taksh.tool_calling_contract_report/1",
        "generated_by": "evaluation/tool_calling_contract.py",
        "contract_manifest": _relative(_path(path)),
        "status": overall,
        "verdict": "TOOL_CALLING_NOT_IMPLEMENTED",
        "contract_supported": False,
        "tool_calling_supported": False,
        "fail_closed": policy["fail_closed"],
        "ready": False,
        "policy": policy,
        "contract": {
            "request_fields": spec["request_fields"],
            "rejection": spec["rejection"],
            "allowed_tools": spec["allowed_tools"],
            "approval_required_for_mutation": spec["approval_required_for_mutation"],
            "sandbox": spec["sandbox"],
            "input_validation": spec["input_validation"],
            "output_handling": spec["output_handling"],
            "fail_closed_rules": spec["fail_closed_rules"],
        },
        "server_profile": profile,
        "gates": gates,
        "blockers": [
            "implementation: no tool is implemented",
            "sandbox_approval: no sandbox approval recorded",
            "capability_prerequisite: no canonical checkpoint or capability evaluation",
            "owner_authorization: tool calling is not authorized",
        ],
        "policy_result": {
            "no_tool_execution": True,
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
        help=f"tool-calling contract (default: {CONTRACT})",
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
        print(f"tool calling supported: {report['tool_calling_supported']}")
        print(f"fail closed: {report['fail_closed']}")
        print("gates:")
        for name, status in report["gates"].items():
            print(f"  {status:<10} {name}")
        print(f"report: {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
