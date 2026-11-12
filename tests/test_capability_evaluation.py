"""Focused tests for the capability-evaluation contract.

The tests never evaluate a model. They verify that capability remains
NOT EVALUATED while prerequisites are missing and that the contract is
fail-closed.
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


def test_capability_remains_not_evaluated():
    from evaluation.capability_evaluation import inspect

    report = inspect()
    assert report["schema"] == "taksh.capability_evaluation_contract_report/1"
    assert report["status"] in ALLOWED
    assert report["status"] == "PARTIAL"
    assert report["verdict"] == "CAPABILITY_NOT_EVALUATED"
    assert report["evaluation_executed"] is False
    assert report["capability_claimed"] is False
    assert report["score"] is None
    assert report["raw_outputs"] is None


def test_prerequisites_are_fail_closed():
    from evaluation.capability_evaluation import inspect

    report = inspect()
    assert report["prerequisites"]["canonical_checkpoint"]["status"] == "BLOCKED"
    assert report["prerequisites"]["canonical_checkpoint"]["exists"] is False
    assert report["prerequisites"]["model_identity"]["status"] == "PASS"
    assert report["prerequisites"]["frozen_benchmarks"]["status"] == "BLOCKED"
    assert report["prerequisites"]["frozen_benchmarks"]["absent"] == [
        "l02-python-v1",
        "l051-multilingual-v1",
    ]
    assert report["prerequisites"]["owner_authorization"]["status"] == "NOT GRANTED"


def test_contract_policy_is_fail_closed():
    from evaluation.capability_evaluation import inspect

    report = inspect()
    assert report["policy"] == {
        "evaluation_executed": False,
        "capability_claimed": False,
        "fail_closed": True,
        "partial_scoring_allowed": False,
        "silent_skip_allowed": False,
        "requires_canonical_checkpoint": True,
        "requires_frozen_benchmarks": True,
        "requires_owner_authorization": True,
        "requires_raw_outputs": True,
        "preserves_historical_results": True,
    }
    assert report["gates"]["contract_schema"] == "PASS"
    assert report["gates"]["fail_closed_policy"] == "PASS"
    assert report["gates"]["checkpoint_prerequisite"] == "BLOCKED"
    assert report["gates"]["frozen_benchmark_prerequisite"] == "BLOCKED"
    assert report["gates"]["owner_authorization"] == "BLOCKED"
    assert report["gates"]["implementation"] == "BLOCKED"
    assert report["policy_result"] == {
        "no_evaluation_executed": True,
        "no_capability_claim": True,
        "no_partial_score": True,
        "historical_results_preserved": True,
        "no_publication": True,
    }
    assert report["blockers"]


def test_report_is_deterministic():
    from evaluation.capability_evaluation import inspect

    assert inspect() == inspect()


def test_contract_cannot_claim_evaluation_without_execution(tmp_path):
    from evaluation.capability_evaluation import CapabilityEvaluationError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/capability_evaluation_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["evaluation_executed"] = True
    path = tmp_path / "capability.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(CapabilityEvaluationError, match="evaluation_executed"):
        inspect(path)


def test_contract_cannot_claim_capability(tmp_path):
    from evaluation.capability_evaluation import CapabilityEvaluationError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/capability_evaluation_contract.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["capability_claimed"] = True
    path = tmp_path / "capability.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(CapabilityEvaluationError, match="capability_claimed"):
        inspect(path)


def test_cli_writes_report_and_exits_nonzero_while_blocked(tmp_path, capsys):
    from evaluation.capability_evaluation import main

    out = tmp_path / "capability.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "CAPABILITY_NOT_EVALUATED" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.capability_evaluation_contract_report/1"
    assert written["evaluation_executed"] is False
    assert written["capability_claimed"] is False
