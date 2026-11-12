"""Focused tests for the fail-closed tool-calling contract design.

The tests never implement or execute a tool. They verify that the contract is
explicitly unsupported, fail-closed and aligned with the serving profile.
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


def test_contract_is_explicitly_unsupported_and_fail_closed():
    from evaluation.tool_calling_contract import inspect

    report = inspect()
    assert report["schema"] == "taksh.tool_calling_contract_report/1"
    assert report["status"] in ALLOWED
    assert report["status"] == "PARTIAL"
    assert report["verdict"] == "TOOL_CALLING_NOT_IMPLEMENTED"
    assert report["contract_supported"] is False
    assert report["tool_calling_supported"] is False
    assert report["fail_closed"] is True
    assert report["ready"] is False
    assert report["policy"] == {
        "tool_calling_supported": False,
        "fail_closed": True,
        "partial_execution_allowed": False,
        "silent_ignore_allowed": False,
        "requires_canonical_checkpoint": True,
        "requires_capability_evaluation": True,
        "requires_owner_authorization": True,
        "requires_sandbox_approval": True,
    }


def test_contract_exposes_no_tools_and_rejects_requests():
    from evaluation.tool_calling_contract import inspect

    contract = inspect()["contract"]
    assert contract["allowed_tools"] == []
    assert contract["approval_required_for_mutation"] is True
    assert contract["rejection"] == {
        "http_status": 400,
        "error_type": "invalid_request_error",
        "code": "tool_calling_not_implemented",
    }
    assert contract["sandbox"] == {
        "network": "disabled",
        "filesystem": "jailed",
        "timeout_seconds": 30,
        "no_untrusted_code_execution": True,
    }


def test_contract_is_aligned_with_the_serving_profile():
    from evaluation.tool_calling_contract import inspect

    report = inspect()
    assert report["server_profile"] == {
        "tool_calling_supported": False,
        "tool_calling_status": (
            "unsupported: the server rejects tools/functions/tool_choice with "
            "an explicit error rather than ignoring them"
        ),
    }
    assert report["gates"]["contract_schema"] == "PASS"
    assert report["gates"]["fail_closed_policy"] == "PASS"
    assert report["gates"]["server_profile_alignment"] == "PASS"
    assert report["gates"]["implementation"] == "BLOCKED"
    assert report["gates"]["sandbox_approval"] == "BLOCKED"
    assert report["blockers"]


def test_report_preserves_fail_closed_policy_result():
    from evaluation.tool_calling_contract import inspect

    report = inspect()
    assert report["policy_result"] == {
        "no_tool_execution": True,
        "no_partial_side_effects": True,
        "no_capability_claim": True,
        "no_publication": True,
    }


def test_report_is_deterministic():
    from evaluation.tool_calling_contract import inspect

    assert inspect() == inspect()


def test_contract_cannot_claim_tool_support_without_implementation(tmp_path):
    from evaluation.tool_calling_contract import ToolCallingContractError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/tool_calling_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["tool_calling_supported"] = True
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ToolCallingContractError, match="tool_calling_supported"):
        inspect(path)


def test_contract_cannot_disable_fail_closed_behavior(tmp_path):
    from evaluation.tool_calling_contract import ToolCallingContractError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/tool_calling_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["partial_execution_allowed"] = True
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ToolCallingContractError, match="partial_execution_allowed"):
        inspect(path)


def test_cli_writes_report_and_exits_nonzero_while_unimplemented(tmp_path, capsys):
    from evaluation.tool_calling_contract import main

    out = tmp_path / "tool_contract.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "TOOL_CALLING_NOT_IMPLEMENTED" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.tool_calling_contract_report/1"
    assert written["tool_calling_supported"] is False
    assert written["fail_closed"] is True
