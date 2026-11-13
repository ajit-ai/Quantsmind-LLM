"""Focused CLI tests for the provider-independent coding workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import taksh
from taksh.engineering import FileChange


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "app.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return root


def test_cli_lists_providers(capsys):
    assert taksh.main(["code", "--list-providers"]) == 0
    output = capsys.readouterr().out
    assert "openai-compatible" in output
    assert "ollama" in output
    assert "native-taksh" in output


def test_cli_refuses_test_provider_without_explicit_flag(tmp_path, capsys):
    root = _project(tmp_path)
    code = taksh.main(["code", "--project", str(root), "--task", "change f", "--provider", "deterministic-mock"])
    assert code == 2
    assert "deterministic test provider" in capsys.readouterr().err


def test_cli_runs_plan_only_workflow(tmp_path, capsys):
    root = _project(tmp_path)
    code = taksh.main([
        "code",
        "--project", str(root),
        "--task", "change f",
        "--provider", "deterministic-mock",
        "--allow-test-provider",
        "--json",
    ])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["workflow"]["status"] == "PLANNED"
    assert report["workflow"]["approved"] is False


def test_cli_diagnose_is_non_secret(capsys, monkeypatch):
    monkeypatch.delenv("TAKSH_OPENAI_API_KEY", raising=False)
    assert taksh.main(["code", "--diagnose", "openai-compatible"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unauthenticated"
    assert "api_key" not in report
