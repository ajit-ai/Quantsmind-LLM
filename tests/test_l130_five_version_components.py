"""Focused tests for L1.2.0–2.0.0 components (non-training, non-execution)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_coding_engine_runs_without_applier():
    from taksh.coding_engine import TaskEngine, TaskSpec

    def generator(task):
        return "def add(a, b):\n    return a + b\n"

    engine = TaskEngine()
    result = engine.run(
        TaskSpec("t1", "write add"),
        generator=generator,
        applier=None,
    )
    assert result.status.value in ("COMPLETED", "FAILED")


def test_coding_validation_syntax_ok():
    from taksh.coding_validation import check_syntax

    result = check_syntax("def add(a, b):\n    return a + b\n")
    assert result.status == "PASS"
    assert not result.issues


def test_coding_validation_syntax_error():
    from taksh.coding_validation import check_syntax

    result = check_syntax("def add(:\n    return\n")
    assert result.status == "FAIL"
    assert result.issues[0].category == "syntax"


def test_specialists_registered():
    from taksh.specialists import SPECIALISTS

    expected = {
        "python_generation", "python_completion", "python_explanation",
        "python_debugging", "python_refactoring", "test_generation",
        "test_failure_diagnosis", "code_review", "dependency_analysis",
        "documentation_generation",
    }
    assert set(SPECIALISTS) == expected


def test_generation_worker_returns_template():
    from taksh.specialists import SPECIALISTS

    out = SPECIALISTS["python_generation"].run("def f():\n    return 1\n", target="def f():")
    assert out.status == "OK"


def test_dependency_analysis_extracts_imports():
    from taksh.specialists import SPECIALISTS

    src = "import os\nimport sys\nfrom pathlib import Path\n\ndef f():\n    return os.getcwd()\n"
    out = SPECIALISTS["dependency_analysis"].run(src)
    assert "os" in out.output


def test_code_review_flags_bare_except():
    from taksh.specialists import SPECIALISTS

    src = "def f():\n    try:\n        return 1\n    except:\n        pass\n"
    out = SPECIALISTS["code_review"].run(src)
    assert any("bare except" in f for f in out.findings)


def test_l120_training_readiness_runs():
    from evaluation.l120_training_readiness import verify

    report = verify()
    assert report["schema"] == "taksh.training_readiness/1"
    assert report["artifacts"]["foundation_config"]["parameters"] == 110_658_816
    assert report["artifacts"]["tokenizer"]["vocab_size"] == 5822
    assert report["authorization_status"]["training_authorized"] is False
