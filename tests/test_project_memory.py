"""Focused tests for project-scoped memory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.memory import ProjectMemory


def test_memory_is_project_scoped(tmp_path):
    memory = ProjectMemory(tmp_path / "memory.json")
    memory.store("project-a", "fact", "value-a", source="test", verified=True)
    memory.store("project-b", "fact", "value-b", source="test", verified=True)
    assert [f.value for f in memory.get("project-a")] == ["value-a"]
    assert [f.value for f in memory.get("project-b")] == ["value-b"]


def test_memory_persists_and_recovers(tmp_path):
    path = tmp_path / "memory.json"
    memory = ProjectMemory(path)
    memory.store("project-a", "fact", "value-a", source="test", verified=True)
    recovered = ProjectMemory(path)
    assert [f.value for f in recovered.get("project-a")] == ["value-a"]


def test_memory_invalidation_and_deletion(tmp_path):
    memory = ProjectMemory(tmp_path / "memory.json")
    fact = memory.store("project-a", "fact", "value-a", source="test", verified=True)
    memory.invalidate(fact.fact_id, "source changed")
    assert memory.get("project-a")[0].invalidated is True
    assert memory.delete(fact.fact_id) is True
    assert memory.get("project-a") == []


def test_memory_refuses_secret_like_values(tmp_path):
    memory = ProjectMemory(tmp_path / "memory.json")
    with pytest.raises(ValueError):
        memory.store("project-a", "fact", "password=secret", source="test")


def test_memory_clears_project(tmp_path):
    memory = ProjectMemory(tmp_path / "memory.json")
    memory.store("project-a", "fact", "value-a", source="test")
    memory.store("project-b", "fact", "value-b", source="test")
    assert memory.clear_project("project-a") == 1
    assert memory.get("project-a") == []
    assert len(memory.get("project-b")) == 1
