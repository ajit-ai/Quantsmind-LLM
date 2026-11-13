"""Focused tests for the controlled execution sandbox."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from taksh.sandbox import Sandbox, SandboxPolicy


def test_allowlisted_command_runs(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"], timeout_seconds=10)
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "print('hello')"], cwd=str(tmp_path))
    assert result.status == "PASS"
    assert "hello" in result.stdout


def test_disallowed_command_is_rejected(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"])
    sandbox = Sandbox(policy)
    result = sandbox.run(["rm", "-rf", "/"], cwd=str(tmp_path))
    assert result.status == "rejected"
    assert "not allowlisted" in result.stderr


def test_workspace_escape_is_rejected(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"])
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "print('x')"], cwd=str(tmp_path.parent))
    assert result.status == "rejected"


def test_timeout_is_reported(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"], timeout_seconds=1)
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "import time; time.sleep(5)"], cwd=str(tmp_path))
    assert result.status == "TIMEOUT"
    assert result.timed_out is True


def test_output_is_bounded(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"], max_output_bytes=20)
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "print('a' * 100)"], cwd=str(tmp_path))
    assert result.truncated is True
    assert len(result.stdout) <= 20


def test_strict_isolation_fails_closed_when_unavailable(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"], isolation="strict")
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "print('should not run')"], cwd=str(tmp_path))
    assert result.status == "rejected"
    assert "strict OS isolation" in result.stderr
    assert result.returncode is None


def test_application_isolation_remains_available(tmp_path):
    policy = SandboxPolicy(workspace=str(tmp_path), allowed_commands=["python"], isolation="application")
    sandbox = Sandbox(policy)
    result = sandbox.run(["python", "-c", "print('ok')"], cwd=str(tmp_path))
    assert result.status == "PASS"
