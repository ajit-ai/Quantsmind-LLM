"""Focused tests for the tool-calling runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.tools import ToolRuntime, build_engineering_tool_registry, ToolContext


def test_tool_catalog_contains_required_tools():
    registry = build_engineering_tool_registry()
    names = registry.names()
    for required in ("inspect_project", "file_lookup", "patch_apply", "validate_project", "test_execution"):
        assert required in names


def test_unknown_tool_is_rejected(tmp_path):
    registry = build_engineering_tool_registry()
    runtime = ToolRuntime(registry)
    invocation = runtime.invoke("not_a_tool", {}, ToolContext(root=str(tmp_path)))
    assert invocation.status == "error"
    assert "unknown tool" in invocation.error


def test_invalid_arguments_are_rejected(tmp_path):
    registry = build_engineering_tool_registry()
    runtime = ToolRuntime(registry)
    invocation = runtime.invoke("file_lookup", {}, ToolContext(root=str(tmp_path)))
    assert invocation.status == "error"
    assert "missing required argument" in invocation.error


def test_tool_call_limit_is_enforced(tmp_path):
    registry = build_engineering_tool_registry()
    runtime = ToolRuntime(registry)
    runtime._max_calls = 1
    first = runtime.invoke("inspect_project", {}, ToolContext(root=str(tmp_path)))
    second = runtime.invoke("inspect_project", {}, ToolContext(root=str(tmp_path)))
    assert first.status == "success"
    assert second.status == "error"
    assert "limit exceeded" in second.error


def test_tool_audit_history_is_recorded(tmp_path):
    registry = build_engineering_tool_registry()
    runtime = ToolRuntime(registry)
    runtime.invoke("inspect_project", {}, ToolContext(root=str(tmp_path)))
    audit = runtime.audit()
    assert len(audit) == 1
    assert audit[0]["tool"] == "inspect_project"
    assert audit[0]["status"] == "success"
