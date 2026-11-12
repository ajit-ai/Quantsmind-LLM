"""Validate the GA security, sandbox and deployment contract.

This module reads the contract only. It does not perform a security review,
approve a sandbox, automate deployment, train, create a checkpoint or publish
anything. GA remains NOT STARTED.

Run:

    python -m evaluation.ga_security_contract
    python -m evaluation.ga_security_contract --json
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

CONTRACT = "configs/roadmap/ga_security_contract.yaml"
DEFAULT_OUT = "reports/taksh_ga_security_contract.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


class GaSecurityContractError(RuntimeError):
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
        raise GaSecurityContractError(f"contract not found: {contract_path}")
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise GaSecurityContractError(f"contract is not a mapping: {contract_path}")
    return document


def _validate(contract: dict[str, Any]) -> None:
    errors: list[str] = []
    if contract.get("schema") != "taksh.ga_security_contract/1":
        errors.append(f"unexpected schema: {contract.get('schema')!r}")
    if contract.get("status") != "NOT STARTED":
        errors.append("GA status must remain NOT STARTED")

    policy = contract.get("policy") or {}
    for key in ("ga_ready", "security_review_completed", "sandbox_approved", "deployment_automated", "fail_closed", "no_publication"):
        if not isinstance(policy.get(key), bool):
            errors.append(f"policy.{key} must be a boolean")
    if policy.get("ga_ready") is not False:
        errors.append("ga_ready must be false")
    if policy.get("security_review_completed") is not False:
        errors.append("security_review_completed must be false")
    if policy.get("sandbox_approved") is not False:
        errors.append("sandbox_approved must be false")
    if policy.get("deployment_automated") is not False:
        errors.append("deployment_automated must be false")
    if policy.get("fail_closed") is not True:
        errors.append("fail_closed must be true")
    if policy.get("no_publication") is not True:
        errors.append("no_publication must be true")

    spec = contract.get("contract") or {}
    security = spec.get("security") or {}
    for key in ("workspace_sandboxing", "permission_model_documented", "secret_scanner_configured", "dependency_audit", "supply_chain_provenance"):
        if security.get(key) != "required":
            errors.append(f"security.{key} must be required")
    sandbox = spec.get("sandbox") or {}
    if sandbox.get("network") != "disabled":
        errors.append("sandbox.network must be disabled")
    if sandbox.get("filesystem") != "jailed":
        errors.append("sandbox.filesystem must be jailed")
    if sandbox.get("no_untrusted_code_execution") is not True:
        errors.append("sandbox.no_untrusted_code_execution must be true")
    if sandbox.get("no_secret_leakage") is not True:
        errors.append("sandbox.no_secret_leakage must be true")
    deployment = spec.get("deployment") or {}
    if deployment.get("local_cpu_profile") != "required":
        errors.append("deployment.local_cpu_profile must be required")
    endpoints = deployment.get("health_endpoints") or []
    if "/health" not in endpoints or "/v1/models" not in endpoints:
        errors.append("deployment.health_endpoints must include /health and /v1/models")
    if not spec.get("fail_closed_rules"):
        errors.append("fail_closed_rules must be non-empty")

    if errors:
        raise GaSecurityContractError("; ".join(errors))


def inspect(path: str | Path = CONTRACT) -> dict[str, Any]:
    """Return deterministic GA security evidence; GA remains not started."""
    contract = load_contract(path)
    _validate(contract)

    policy = contract["policy"]
    spec = contract["contract"]

    gates = {
        "contract_schema": "PASS",
        "fail_closed_policy": "PASS",
        "security_review": "BLOCKED",
        "sandbox_approval": "BLOCKED",
        "dependency_audit": "BLOCKED",
        "deployment_automation": "BLOCKED",
        "capability_prerequisite": "BLOCKED",
        "owner_authorization": "BLOCKED",
    }

    overall = "PARTIAL"
    if overall not in ALLOWED_STATUSES:
        raise GaSecurityContractError(f"invalid overall status: {overall}")

    return {
        "schema": "taksh.ga_security_contract_report/1",
        "generated_by": "evaluation/ga_security_contract.py",
        "contract_manifest": _relative(_path(path)),
        "ga_status": contract["status"],
        "status": overall,
        "verdict": "GA_NOT_READY",
        "ga_ready": False,
        "policy": policy,
        "contract": spec,
        "gates": gates,
        "blockers": [
            "security_review: not completed",
            "sandbox_approval: not recorded",
            "dependency_audit: not completed",
            "deployment_automation: not implemented",
            "capability_prerequisite: capability not evaluated",
            "owner_authorization: GA release not authorized",
        ],
        "policy_result": {
            "no_security_claim": True,
            "no_deployment_claim": True,
            "no_publication": True,
            "no_ga_readiness_claim": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        default=CONTRACT,
        help=f"GA security contract (default: {CONTRACT})",
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
        print(f"GA status: {report['ga_status']}")
        print(f"status: {report['status']}")
        print(f"verdict: {report['verdict']}")
        print("gates:")
        for name, status in report["gates"].items():
            print(f"  {status:<10} {name}")
        print(f"report: {out_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
