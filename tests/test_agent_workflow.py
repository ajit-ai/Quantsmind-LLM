"""Focused tests for the agent workflow."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.agent import AgentWorkflow, AgentState
from taksh.engineering import DeterministicTestProvider, FileChange


def _make_project(tmp_path):
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "math.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_math.py").write_text("from pkg.math import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
    return root


def test_agent_workflow_completes_with_deterministic_provider(tmp_path):
    root = _make_project(tmp_path)
    changes = [FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b + 0\n", reason="adjust")]
    provider = DeterministicTestProvider(changes)
    workflow = AgentWorkflow(storage_path=tmp_path / "memory.json")
    task = workflow.run("agent-1", "adjust add", str(root), provider, apply=True, validate=True)
    assert task.state == AgentState.REPORTED
    assert task.result["task"]["state"] == "COMPLETED"
    assert task.provenance["provider"] == "deterministic-test"


def test_agent_workflow_blocks_unavailable_provider(tmp_path):
    root = _make_project(tmp_path)
    from taksh.engineering import NativeTakshProvider

    workflow = AgentWorkflow()
    task = workflow.run("agent-2", "do work", str(root), NativeTakshProvider(), apply=False)
    assert task.state == AgentState.BLOCKED


def test_agent_workflow_persists_and_loads(tmp_path):
    root = _make_project(tmp_path)
    changes = [FileChange(path="pkg/math.py", operation="modify", content="def add(a, b):\n    return a + b\n", reason="noop")]
    provider = DeterministicTestProvider(changes)
    workflow = AgentWorkflow()
    task = workflow.run("agent-3", "noop", str(root), provider, apply=True, validate=True)
    path = tmp_path / "agent.json"
    workflow.persist(task, path)
    loaded = workflow.load(path)
    assert loaded.task_id == task.task_id
    assert loaded.state == task.state
