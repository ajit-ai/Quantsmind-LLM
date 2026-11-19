"""Tests for TAKSH Native v0.1 baseline capability evaluation.

Covers task inventory, deterministic scoring/extraction, safe isolated execution,
result schema/summarization, and reproducibility of scoring. Does not require the
real checkpoint (real-model execution is covered by the benchmark run/report).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.native_v01.tasks import TASKS, get_tasks  # noqa: E402
from evaluation.native_v01 import runner  # noqa: E402


REQUIRED_CATEGORIES = {
    "basic_function_generation",
    "arithmetic_string_manipulation",
    "lists_dicts_iteration",
    "conditional_logic_edge_cases",
    "simple_algorithm",
    "function_completion",
    "instruction_following",
    "code_correction_debugging",
}


def test_task_inventory_size_and_categories():
    tasks = get_tasks()
    assert len(tasks) == 20
    assert len(TASKS) == 20
    ids = [t.task_id for t in tasks]
    assert len(set(ids)) == 20
    assert all(tid.startswith("NV01-") for tid in ids)
    cats = {t.category for t in tasks}
    assert REQUIRED_CATEGORIES.issubset(cats)
    for t in tasks:
        assert t.prompt and t.expected_behavior
        assert t.acceptance_criteria
        assert t.entry_point
        assert len(t.tests) >= 2
        d = t.to_dict()
        assert d["task_id"] == t.task_id
        assert isinstance(d["acceptance_criteria"], list)
        assert isinstance(d["tests"], list)


def test_extract_code_from_fence_and_prefix():
    text = "Here you go:\n```python\ndef add(a,b):\n    return a+b\n```\nThanks"
    assert "def add" in runner.extract_code(text)
    plain = "def add(a,b):\n    return a+b"
    assert runner.extract_code(plain).startswith("def add")


def test_repetition_stats_detects_degenerate():
    rep = runner._repetition_stats("( ( ( ( ( ( ( (")
    assert rep["repeated"] is True
    ok = runner._repetition_stats("def add(a, b): return a + b")
    assert ok["repeated"] is False


def test_evaluate_output_syntax_and_target():
    task = get_tasks()[0]  # add
    good = runner.evaluate_output(task, "```python\ndef add(a, b):\n    return a + b\n```")
    assert good["syntax_ok"] is True
    assert good["has_target_definition"] is True
    assert good["instruction_followed"] is True
    bad = runner.evaluate_output(task, "def add(a b)\n    return a+b")
    assert bad["syntax_ok"] is False
    assert bad["syntax_error"]
    empty = runner.evaluate_output(task, "", {"finish_reason": "length"})
    assert empty["empty_output"] is True
    assert empty["truncated"] is True


def test_safe_execution_correct_and_incorrect():
    task = get_tasks()[0]  # add
    ok = runner.execute_task_tests(task, "def add(a, b):\n    return a + b\n", timeout_s=5.0)
    assert ok["executable"] is True
    assert ok["defined"] is True
    assert ok["correct"] is True
    assert ok["passed"] == ok["total"] == 3

    bad = runner.execute_task_tests(task, "def add(a, b):\n    return a - b\n", timeout_s=5.0)
    assert bad["executable"] is True
    assert bad["defined"] is True
    assert bad["correct"] is False
    assert bad["passed"] < bad["total"]


def test_safe_execution_syntax_error_and_timeout():
    task = get_tasks()[0]
    syn = runner.execute_task_tests(task, "def add(a, b) return a+b", timeout_s=5.0)
    assert syn["executable"] is False
    assert syn["correct"] is False

    hang = runner.execute_task_tests(task, "def add(a,b):\n    while True:\n        pass\n", timeout_s=0.2)
    assert hang["correct"] is False
    assert hang["error"] in {"timeout", None} or "timeout" in str(hang.get("error"))


def test_score_task_does_not_count_syntax_as_correct():
    task = get_tasks()[0]
    scored = runner.score_task(task, "def add(a, b):\n    return a - b\n", {"finish_reason": "stop"})
    assert scored["syntax_ok"] is True
    assert scored["executable"] is True
    assert scored["correct"] is False


def test_completion_task_harness_accepts_body_only():
    # NV01-COMP-015 count_items has completion_prefix
    task = next(t for t in get_tasks() if t.task_id == "NV01-COMP-015")
    body_only = "    return len(items)\n"
    res = runner.execute_task_tests(task, body_only, timeout_s=5.0)
    assert res["correct"] is True


def test_summarize_and_schema():
    results = [
        {"category": "basic_function_generation", "syntax_ok": True, "executable": True, "correct": True, "instruction_followed": True, "truncated": False, "empty_output": False, "degenerate_output": False, "inference_failed": False, "inference_error": None},
        {"category": "basic_function_generation", "syntax_ok": False, "executable": False, "correct": False, "instruction_followed": False, "truncated": False, "empty_output": True, "degenerate_output": True, "inference_failed": False, "inference_error": None},
    ]
    s = runner.summarize(results)
    assert s["task_count"] == 2
    assert s["syntax_ok"] == 1 and s["correct"] == 1
    assert 0.0 <= s["syntax_rate"] <= 1.0
    assert "basic_function_generation" in s["by_category"]
    assert s["by_category"]["basic_function_generation"]["count"] == 2


def test_run_evaluation_blocks_on_missing_checkpoint(tmp_path):
    report = runner.run_evaluation(
        checkpoint_path=tmp_path / "missing.pt",
        tokenizer_path=tmp_path / "missing_tok.json",
        tasks=get_tasks()[:1],
    )
    assert report["status"] == "BLOCKED"
    assert "checkpoint_missing" in (report.get("blocker") or "")
    assert report["summary"] == {}


def test_scoring_determinism_same_input():
    task = get_tasks()[0]
    raw = "def add(a, b):\n    return a + b\n"
    meta = {"finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 7}
    a = runner.score_task(task, raw, dict(meta), timeout_s=5.0)
    b = runner.score_task(task, raw, dict(meta), timeout_s=5.0)
    # strip any non-deterministic incidental fields if ever added; core scores stable
    for k in ("syntax_ok", "executable", "correct", "tests_passed", "instruction_followed", "empty_output", "degenerate_output"):
        assert a[k] == b[k]
    assert a["case_results"] == b["case_results"]
