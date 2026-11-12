"""Focused tests for GA implementation hardening."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.logging import get_logger
from taksh.recovery import recover_tasks
from taksh.release import check_version_consistency, verify_release_artifacts
from evaluation.dependency_inventory import dependency_inventory


def test_structured_logger_emits_json(capsys):
    logger = get_logger("taksh.test")
    logger.info("hello")
    captured = capsys.readouterr()
    payload = json.loads(captured.out.strip())
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"


def test_recovery_detects_interrupted_and_corrupt_tasks(tmp_path):
    from taksh.agent import AgentTask, AgentState
    import json as _json

    directory = tmp_path / "tasks"
    directory.mkdir()
    (directory / "task.json").write_text(_json.dumps({"task_id": "t1", "objective": "o", "root": str(tmp_path), "state": "PLANNED", "history": [], "provenance": {}, "result": {}, "cancelled": False}), encoding="utf-8")
    (directory / "bad.json").write_text("not json", encoding="utf-8")
    result = recover_tasks(directory)
    assert result["status"] == "ok"
    assert result["interrupted_count"] == 2
    assert result["corrupt_count"] == 1


def test_release_checks_are_available():
    consistency = check_version_consistency()
    assert consistency["package_version"] == "1.0.0"
    artifacts = verify_release_artifacts()
    assert artifacts["all_present"] is True


def test_dependency_inventory_records_hashes_without_network():
    inventory = dependency_inventory()
    assert inventory["vulnerability_scan"] == "NOT_EVALUATED"
    assert inventory["license_review"] == "NOT_EVALUATED"
    assert len(inventory["pyproject_dependencies"]) > 0
    assert inventory["pyproject_sha256"]
    assert inventory["requirements_sha256"]
