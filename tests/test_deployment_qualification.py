"""Deployment qualification tests for TAKSH self-hosted path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.deployment import (
    backup_state,
    restore_state,
    validate_deployment_config,
    readiness_check,
    graceful_shutdown_report,
)
from taksh.memory import ProjectMemory
from taksh.recovery import recover_tasks


def _temp_config(tmp_path, token="a-strong-random-token-with-32-characters"):
    return {
        "TAKSH_ENGINEERING_TOKEN": token,
        "TAKSH_TOKEN_SCOPES": "read,execute",
        "TAKSH_WORKSPACE": str(tmp_path),
    }


def test_fresh_deployment_configuration_validates(tmp_path):
    result = validate_deployment_config(_temp_config(tmp_path))
    assert result["status"] == "valid"
    assert result["errors"] == []


def test_configuration_validation_failure(tmp_path):
    result = validate_deployment_config({"TAKSH_WORKSPACE": str(tmp_path)})
    assert result["status"] == "invalid"
    assert result["errors"]


def test_placeholder_token_is_rejected(tmp_path):
    result = validate_deployment_config(_temp_config(tmp_path, token="replace-with-strong-random-token"))
    assert result["status"] == "invalid"
    assert any("placeholder" in error for error in result["errors"])


def test_readiness_succeeds_for_valid_workspace(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = readiness_check(tmp_path)
    assert result["status"] == "ready"


def test_backup_and_restore_round_trip(tmp_path):
    memory_path = tmp_path / "memory.json"
    memory = ProjectMemory(memory_path)
    memory.store("deploy-test", "fact", "persistent value", source="test", verified=True)
    backup_path = tmp_path / "backup.json"
    backup = backup_state(memory_path, backup_path)
    assert backup["status"] == "backed_up"
    memory_path.unlink()
    restored = restore_state(backup_path, memory_path)
    assert restored["status"] == "restored"
    recovered = ProjectMemory(memory_path)
    values = [fact.value for fact in recovered.get("deploy-test")]
    assert "persistent value" in values


def test_restore_rejects_invalid_backup(tmp_path):
    backup = tmp_path / "backup.json"
    backup.write_text("not json", encoding="utf-8")
    result = restore_state(backup, tmp_path / "memory.json")
    assert result["status"] == "invalid"


def test_interrupted_task_recovery_and_shutdown(tmp_path):
    from taksh.agent import AgentTask, AgentState
    import json as _json

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "t.json").write_text(_json.dumps({
        "task_id": "t", "objective": "o", "root": str(tmp_path), "state": "PLANNED",
        "history": [], "provenance": {}, "result": {}, "cancelled": False,
    }), encoding="utf-8")
    recovered = recover_tasks(tasks)
    assert recovered["interrupted_count"] == 1
    report = graceful_shutdown_report(active_tasks=1)
    assert report["active_tasks"] == 1
