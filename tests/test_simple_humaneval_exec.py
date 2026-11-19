"""Tests for v1.7.0 execution-based simple_humaneval evaluation."""

from __future__ import annotations

from evaluation.simple_humaneval_exec import (
    extract_function_name,
    build_test_cases,
    run_exec_eval,
    summarize_exec,
)
from evaluation.simple_humaneval import PROBLEMS


def test_extract_function_name():
    """Function name extraction works for standard def prompts."""
    assert extract_function_name("def add(a, b):\n    return ") == "add"
    assert extract_function_name("def multiply(x, y):\n    return ") == "multiply"
    assert extract_function_name("def is_even(n):\n    return ") == "is_even"
    assert extract_function_name("def maximum(a, b):\n    if a > b:") == "maximum"


def test_build_test_cases():
    """Test case generation returns deterministic cases per problem."""
    cases = build_test_cases(PROBLEMS[0])  # p01_add
    assert len(cases) == 3
    assert cases[0]["name"] == "add_2_3"
    assert cases[0]["args"] == [2, 3]
    assert cases[0]["expected"] == 5

    # All problems should have test cases
    for p in PROBLEMS:
        cases = build_test_cases(p)
        assert len(cases) > 0, f"{p.id} has no test cases"
        for c in cases:
            assert "name" in c
            assert "args" in c
            assert "expected" in c


def test_run_exec_eval_with_passing_mock():
    """run_exec_eval with a correct generator passes all tests."""
    problems = PROBLEMS[:3]

    def correct_generate(prompt: str) -> str:
        # Return the expected tail for known prompts
        for p in problems:
            if prompt == p.prompt:
                return p.expected_tail + "\n"
        return ""

    results = run_exec_eval(correct_generate, problems=problems)
    assert len(results) == 3
    for r in results:
        assert r["syntax_valid"] is True
        assert r["passed"] is True
        assert r["tests_passed"] == r["tests_total"]


def test_run_exec_eval_with_failing_mock():
    """run_exec_eval with a wrong generator reports failures."""
    problems = PROBLEMS[:2]

    def wrong_generate(prompt: str) -> str:
        return "return 0\n"

    results = run_exec_eval(wrong_generate, problems=problems)
    assert len(results) == 2
    for r in results:
        assert r["passed"] is False


def test_run_exec_eval_syntax_error():
    """run_exec_eval handles syntax errors gracefully."""
    problems = PROBLEMS[:1]

    def broken_generate(prompt: str) -> str:
        return "this is not valid python !!!\n"

    results = run_exec_eval(broken_generate, problems=problems)
    assert len(results) == 1
    assert results[0]["syntax_valid"] is False
    assert results[0]["passed"] is False
    assert results[0]["execution_error"] == "syntax_error"


def test_summarize_exec():
    """summarize_exec computes correct aggregate metrics."""
    results = [
        {"passed": True, "syntax_valid": True, "category": "arithmetic",
         "tests_passed": 3, "tests_total": 3},
        {"passed": False, "syntax_valid": True, "category": "arithmetic",
         "tests_passed": 1, "tests_total": 3},
        {"passed": True, "syntax_valid": True, "category": "logic",
         "tests_passed": 2, "tests_total": 2},
        {"passed": False, "syntax_valid": False, "category": "logic",
         "tests_passed": 0, "tests_total": 0},
    ]
    summary = summarize_exec(results)
    assert summary["total"] == 4
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["syntax_valid"] == 3
    assert summary["syntax_valid_rate"] == 0.75
    assert summary["total_tests"] == 8
    assert summary["passed_tests"] == 6
    assert summary["test_pass_rate"] == 0.75

    # Category breakdown
    assert summary["by_category"]["arithmetic"]["total"] == 2
    assert summary["by_category"]["arithmetic"]["passed"] == 1
    assert summary["by_category"]["logic"]["total"] == 2
    assert summary["by_category"]["logic"]["passed"] == 1


def test_summarize_exec_empty():
    """summarize_exec on empty results."""
    summary = summarize_exec([])
    assert summary["total"] == 0
    assert "error" in summary
