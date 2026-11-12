"""Focused tests for the shared engineering service."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.service import EngineeringService
from taksh.engineering import DeterministicTestProvider, FileChange


def _make_project(tmp_path):
    root = tmp_path / "project"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "math.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return root


def test_service_inspect_and_validate(tmp_path):
    root = _make_project(tmp_path)
    service = EngineeringService(root)
    info = service.inspect_project()
    assert info["summary"]["python_files"] == 1
    validation = service.validate()
    assert validation["status"] == "PASS"


def test_service_tool_catalog_and_invoke(tmp_path):
    root = _make_project(tmp_path)
    service = EngineeringService(root)
    catalog = service.tool_catalog()
    assert any(tool["name"] == "inspect_project" for tool in catalog)
    result = service.invoke_tool("inspect_project", {})
    assert result["status"] == "success"


def test_service_memory_round_trip(tmp_path):
    root = _make_project(tmp_path)
    service = EngineeringService(root, memory_path=tmp_path / "memory.json")
    fact = service.memory_store("project-a", "note", "hello", source="test", verified=True)
    assert fact["value"] == "hello"
    inspected = service.memory_inspect("project-a")
    assert inspected["fact_count"] == 1
    assert service.memory_invalidate("project-a", "changed") == 1
