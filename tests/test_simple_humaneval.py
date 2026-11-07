"""Tests for v1.5.0 simple HumanEval-style benchmark.

Verifies that:
1. All 20 problems are well-formed (prompt ends mid-statement, non-empty tail).
2. The check() method correctly identifies valid and invalid generations.
3. run_exec produces correct results with a mock generation function.
4. summarize() computes correct pass rates and category breakdowns.
5. No problem is trivially solvable by empty output.
"""

from __future__ import annotations

from evaluation.simple_humaneval import (
    CATEGORIES,
    EvalProblem,
    PROBLEMS,
    run_eval,
    summarize,
)


def test_problem_count():
    """There are exactly 20 problems."""
    assert len(PROBLEMS) == 20


def test_problems_well_formed():
    """Every problem has a non-empty prompt ending mid-statement and a non-empty tail."""
    for p in PROBLEMS:
        assert p.id, f"problem {p.id} has no id"
        assert p.prompt, f"problem {p.id} has no prompt"
        assert p.expected_tail, f"problem {p.id} has no expected tail"
        # Prompt should end in a way that expects completion (not a complete line)
        stripped = p.prompt.rstrip()
        assert not stripped.endswith(":"), (
            f"problem {p.id} prompt ends with colon, may be complete"
        )
        assert not stripped.endswith("return\n"), (
            f"problem {p.id} prompt ends with bare return"
        )


def test_categories_non_empty():
    """Categories are defined and cover all problems."""
    assert len(CATEGORIES) > 0
    problem_cats = {p.category for p in PROBLEMS}
    assert problem_cats == set(CATEGORIES)


def test_check_pass():
    """check() returns passed=True when syntax valid and tail present."""
    p = EvalProblem(
        id="test_add",
        prompt="def add(a, b):\n    return ",
        expected_tail="a + b",
    )
    result = p.check("a + b\n")
    assert result["syntax_valid"] is True
    assert result["expected_tail_present"] is True
    assert result["passed"] is True
    assert result["failure_reason"] == []


def test_check_syntax_fail():
    """check() returns passed=False on syntax error."""
    p = EvalProblem(
        id="test_syntax",
        prompt="def foo(x):\n    return ",
        expected_tail="x + 1",
    )
    result = p.check("x + \n    broken syntax !!!\n")
    assert result["syntax_valid"] is False
    assert result["passed"] is False
    assert "syntax_error" in result["failure_reason"]


def test_check_tail_missing():
    """check() returns passed=False when expected tail is missing."""
    p = EvalProblem(
        id="test_tail",
        prompt="def foo(x):\n    return ",
        expected_tail="x + 1",
    )
    result = p.check("x * 2\n")
    assert result["syntax_valid"] is True
    assert result["expected_tail_present"] is False
    assert result["passed"] is False
    assert "expected_tail_missing" in result["failure_reason"]


def test_check_empty_output():
    """check() returns passed=False on empty generation."""
    p = EvalProblem(
        id="test_empty",
        prompt="def foo(x):\n    return ",
        expected_tail="x",
    )
    result = p.check("")
    assert result["passed"] is False


def test_run_eval_with_mock():
    """run_exec produces correct results with a deterministic mock generator."""
    problems = PROBLEMS[:3]  # first 3 for speed

    def mock_generate(prompt: str) -> str:
        # Perfect mock: returns expected tail for known problems
        for p in problems:
            if prompt == p.prompt:
                return p.expected_tail + "\n"
        return ""

    results = run_eval(mock_generate, problems=problems)
    assert len(results) == 3
    for r in results:
        assert r["passed"] is True
        assert r["syntax_valid"] is True


def test_run_eval_with_bad_mock():
    """run_exec reports failures when mock always returns garbage."""
    problems = PROBLEMS[:2]

    def bad_generate(prompt: str) -> str:
        return "garbage that does not parse !!!\n"

    results = run_eval(bad_generate, problems=problems)
    assert len(results) == 2
    for r in results:
        assert r["passed"] is False


def test_summarize_correct():
    """summarize() computes correct pass rate and category breakdown."""
    results = [
        {"id": "p01", "passed": True, "syntax_valid": True, "category": "arithmetic"},
        {"id": "p02", "passed": False, "syntax_valid": True, "category": "arithmetic"},
        {"id": "p03", "passed": True, "syntax_valid": True, "category": "logic"},
        {"id": "p04", "passed": False, "syntax_valid": False, "category": "logic"},
    ]
    summary = summarize(results)
    assert summary["total"] == 4
    assert summary["passed"] == 2
    assert summary["pass_rate"] == 0.5
    assert summary["syntax_valid"] == 3
    assert summary["syntax_valid_rate"] == 0.75

    # Category breakdown
    assert summary["by_category"]["arithmetic"]["total"] == 2
    assert summary["by_category"]["arithmetic"]["passed"] == 1
    assert summary["by_category"]["arithmetic"]["pass_rate"] == 0.5
    assert summary["by_category"]["logic"]["total"] == 2
    assert summary["by_category"]["logic"]["passed"] == 1
    assert summary["by_category"]["logic"]["pass_rate"] == 0.5


def test_summarize_empty():
    """summarize() on empty results returns error dict."""
    summary = summarize([])
    assert summary["total"] == 0
    assert "error" in summary


def test_all_problems_have_unique_ids():
    """No duplicate problem IDs."""
    ids = [p.id for p in PROBLEMS]
    assert len(ids) == len(set(ids))
