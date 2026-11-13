"""Regression tests for the evaluation-script syntax defect."""

from __future__ import annotations

import ast
import py_compile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "eval_v210_agent.py"


def test_eval_script_parses_and_compiles():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, filename=str(SCRIPT))
    py_compile.compile(str(SCRIPT), doraise=True)


def test_eval_script_has_main_guard():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    guards = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if isinstance(left, ast.Name) and left.id == "__name__":
                guards.append(node)
    assert guards, "evaluation logic must stay behind an __main__ guard"
