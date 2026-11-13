"""Focused tests for the provider-to-orchestration bridge."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from serving.model_change_provider import DeterministicPatchProvider, ModelChangeProvider, parse_structured_changes
from serving.coding_providers import DeterministicCodingProvider
from taksh.engineering import FileChange, create_task, run_engineering_task


def test_parse_structured_changes():
    text = "FILE: a.py\nOPERATION: create\nREASON: add module\nCONTENT:\nx = 1\nEND\n"
    changes = parse_structured_changes(text)
    assert len(changes) == 1
    assert changes[0].path == "a.py"
    assert changes[0].reason == "add module"


def test_deterministic_patch_provider_generates_structured_text():
    from serving.providers import GenerationConfig, GenerationRequest

    provider = DeterministicPatchProvider([
        FileChange(path="a.py", operation="create", content="x = 1\n", reason="test"),
    ])
    result = provider.generate(GenerationRequest("add a", GenerationConfig(max_new_tokens=1, seed=1)))
    assert "FILE: a.py" in result.text


def test_model_change_provider_end_to_end(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")

    changes = [
        FileChange(path="app.py", operation="modify", content="def f():\n    return 2\n", reason="update"),
    ]
    bridge = ModelChangeProvider(DeterministicPatchProvider(changes))
    task = create_task("model-provider", "change f", root)
    task = run_engineering_task(task, bridge, apply=True, validate=True, run_project_tests=False)

    assert task.state.value == "COMPLETED"
    assert task.plan.provider == "deterministic-patch"
    assert "return 2" in (root / "app.py").read_text(encoding="utf-8")
    assert task.capability_claim is None
