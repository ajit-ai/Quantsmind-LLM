"""Focused tests for the GA security, sandbox and deployment contract.

The tests never perform a security review, deploy or publish anything. They
verify that GA remains NOT STARTED and fail-closed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def test_ga_remains_not_started():
    from evaluation.ga_security_contract import inspect

    report = inspect()
    assert report["schema"] == "taksh.ga_security_contract_report/1"
    assert report["ga_status"] == "NOT STARTED"
    assert report["status"] in ALLOWED
    assert report["status"] == "PARTIAL"
    assert report["verdict"] == "GA_NOT_READY"
    assert report["ga_ready"] is False


def test_contract_policy_is_fail_closed():
    from evaluation.ga_security_contract import inspect

    report = inspect()
    assert report["policy"] == {
        "ga_ready": False,
        "security_review_completed": False,
        "sandbox_approved": False,
        "deployment_automated": False,
        "fail_closed": True,
        "no_publication": True,
    }
    assert report["gates"]["contract_schema"] == "PASS"
    assert report["gates"]["fail_closed_policy"] == "PASS"
    assert report["gates"]["security_review"] == "BLOCKED"
    assert report["gates"]["sandbox_approval"] == "BLOCKED"
    assert report["gates"]["dependency_audit"] == "BLOCKED"
    assert report["gates"]["deployment_automation"] == "BLOCKED"
    assert report["gates"]["capability_prerequisite"] == "BLOCKED"
    assert report["gates"]["owner_authorization"] == "BLOCKED"
    assert report["policy_result"] == {
        "no_security_claim": True,
        "no_deployment_claim": True,
        "no_publication": True,
        "no_ga_readiness_claim": True,
    }
    assert report["blockers"]


def test_report_is_deterministic():
    from evaluation.ga_security_contract import inspect

    assert inspect() == inspect()


def test_contract_cannot_claim_ga_readiness(tmp_path):
    from evaluation.ga_security_contract import GaSecurityContractError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/ga_security_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["ga_ready"] = True
    path = tmp_path / "ga.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(GaSecurityContractError, match="ga_ready"):
        inspect(path)


def test_contract_cannot_disable_fail_closed_behavior(tmp_path):
    from evaluation.ga_security_contract import GaSecurityContractError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/ga_security_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["fail_closed"] = False
    path = tmp_path / "ga.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(GaSecurityContractError, match="fail_closed"):
        inspect(path)


def test_cli_writes_report_and_exits_nonzero_while_blocked(tmp_path, capsys):
    from evaluation.ga_security_contract import main

    out = tmp_path / "ga_security.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "GA_NOT_READY" in captured.out
    assert "NOT STARTED" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.ga_security_contract_report/1"
    assert written["ga_ready"] is False
    assert written["ga_status"] == "NOT STARTED"
