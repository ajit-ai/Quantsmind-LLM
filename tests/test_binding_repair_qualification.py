"""Tests for the binding-repair qualification suite and execution gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evaluation.safe_execute import execute_tests, validate_source
from scripts.run_binding_repair_qualification import canonical_digest, score_static
from training.binding_repair import repair_completion

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "evaluation" / "binding_repair_qualification_v1.json"
EXPECTED_SHA = "961936f8b5f098e0d9d863a8753d7430929ba87b0a6880fe27acb81c461f3d2f"


def load_suite():
    return json.loads(SUITE.read_text(encoding="utf-8"))


def test_qualification_dataset_is_frozen():
    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == EXPECTED_SHA
    suite = load_suite()
    assert suite["status"] == "frozen_before_evaluation"
    assert suite["case_count"] == len(suite["cases"]) == 46
    assert canonical_digest(suite["cases"]) == suite["digest"]
    pairs = [(case["instruction"], case["signature"]) for case in suite["cases"]]
    assert len(pairs) == len(set(pairs))


def test_execution_gate_accepts_pure_direct_function():
    source = "def add(a, b):\n    return a + b\n"
    tree, reason = validate_source(source)
    assert tree is not None
    assert reason == "ok"
    result = execute_tests(source, "add", [{"args": [2, 3], "kwargs": {}, "expected": 5}])
    assert result.passed is True


@pytest.mark.parametrize("source", [
    "import os\ndef f(x):\n    return x\n",
    "def f(x):\n    return f(x)\n",
    "def f(x):\n    return x.__class__\n",
    "def f(x):\n    print(x)\n    return x\n",
    "def f(x):\n    open('x').read()\n    return x\n",
    "def f(x):\n    while True:\n        return x\n",
])
def test_execution_gate_rejects_unsafe_source(source):
    tree, reason = validate_source(source)
    assert tree is None
    assert reason


def test_qualification_scoring_handles_reference_body():
    suite = load_suite()
    case = next(item for item in suite["cases"] if item["case_id"] == "Q-add-000")
    completion = "    return left + right\n"
    raw = score_static(case, completion)
    repaired = score_static(case, repair_completion(case["signature"], completion))
    correct = score_static(case, "    return alpha + beta\n")
    assert raw["parses"] is True
    assert raw["exact"] is False
    assert repaired == raw
    assert correct["exact"] is True


def test_negative_cases_are_never_execution_approved():
    suite = load_suite()
    negatives = [case for case in suite["cases"] if case["negative"]]
    assert negatives
    assert all(case["execution_safe"] is False for case in negatives)
