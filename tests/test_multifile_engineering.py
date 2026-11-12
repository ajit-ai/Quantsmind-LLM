"""Focused tests for the multi-file engineering contract.

The tests never mutate a repository. They verify that multi-file editing
remains unimplemented and fail-closed.
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


def test_multifile_engineering_is_unimplemented():
    from evaluation.multifile_engineering import inspect

    report = inspect()
    assert report["schema"] == "taksh.multifile_engineering_contract_report/1"
    assert report["status"] in ALLOWED
    assert report["status"] == "PARTIAL"
    assert report["verdict"] == "MULTIFILE_ENGINEERING_NOT_IMPLEMENTED"
    assert report["multifile_editing_supported"] is False
    assert report["ready"] is False


def test_contract_policy_is_fail_closed():
    from evaluation.multifile_engineering import inspect

    report = inspect()
    assert report["policy"] == {
        "multifile_editing_supported": False,
        "fail_closed": True,
        "partial_apply_allowed": False,
        "rollback_required": True,
        "approval_required": True,
        "sandbox_required": True,
        "capability_required": True,
        "owner_authorization_required": True,
    }
    assert report["gates"]["contract_schema"] == "PASS"
    assert report["gates"]["fail_closed_policy"] == "PASS"
    assert report["gates"]["approval_policy"] == "PASS"
    assert report["gates"]["rollback_policy"] == "PASS"
    assert report["gates"]["implementation"] == "BLOCKED"
    assert report["gates"]["sandbox_approval"] == "BLOCKED"
    assert report["gates"]["capability_prerequisite"] == "BLOCKED"
    assert report["gates"]["owner_authorization"] == "BLOCKED"
    assert report["policy_result"] == {
        "no_repository_mutation": True,
        "no_unapproved_apply": True,
        "no_partial_side_effects": True,
        "no_capability_claim": True,
        "no_publication": True,
    }
    assert report["blockers"]


def test_report_is_deterministic():
    from evaluation.multifile_engineering import inspect

    assert inspect() == inspect()


def test_contract_cannot_claim_multifile_support(tmp_path):
    from evaluation.multifile_engineering import MultifileEngineeringError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/multifile_engineering_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["multifile_editing_supported"] = True
    path = tmp_path / "multifile.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(MultifileEngineeringError, match="multifile_editing_supported"):
        inspect(path)


def test_contract_cannot_disable_fail_closed_behavior(tmp_path):
    from evaluation.multifile_engineering import MultifileEngineeringError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/multifile_engineering_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["partial_apply_allowed"] = True
    path = tmp_path / "multifile.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(MultifileEngineeringError, match="partial_apply_allowed"):
        inspect(path)


def test_cli_writes_report_and_exits_nonzero_while_unimplemented(tmp_path, capsys):
    from evaluation.multifile_engineering import main

    out = tmp_path / "multifile.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "MULTIFILE_ENGINEERING_NOT_IMPLEMENTED" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.multifile_engineering_contract_report/1"
    assert written["multifile_editing_supported"] is False
    assert written["ready"] is False
