"""Taksh v1.5.0 — Simple HumanEval-style coding benchmark.

20 small Python code-completion problems for measuring neural model progress.
Each problem provides a partial function (prompt ending mid-statement) and the
expected completion tail. A generation passes if:
  1. prompt + generated_code parses as valid Python (ast.parse), AND
  2. the expected tail string appears in the generated output.

Usage:
    from evaluation.simple_humaneval import PROBLEMS, run_eval

    results = run_eval(generate_fn)
    # generate_fn(prompt: str) -> str (generated completion only)

Results are a list of dicts with keys: id, syntax_valid, expected_tail_present,
passed, generated, full_code.

This is deterministic and does not execute generated code (safety).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class EvalProblem:
    """A single code-completion problem."""

    id: str
    prompt: str
    expected_tail: str
    category: str = "general"

    def check(self, generated: str) -> Dict[str, Any]:
        """Check a generation against this problem.

        Returns a dict with: syntax_valid, expected_tail_present, passed,
        full_code, failure_reason.
        """
        full = self.prompt + generated
        syntax_valid = False
        try:
            ast.parse(full)
            syntax_valid = True
        except SyntaxError:
            pass

        expected_present = self.expected_tail.strip() in generated
        passed = syntax_valid and expected_present

        failure_reason: List[str] = []
        if not syntax_valid:
            failure_reason.append("syntax_error")
        if not expected_present:
            failure_reason.append("expected_tail_missing")

        return {
            "id": self.id,
            "syntax_valid": syntax_valid,
            "expected_tail_present": expected_present,
            "passed": passed,
            "generated": generated,
            "full_code": full,
            "failure_reason": failure_reason,
            "category": self.category,
        }


PROBLEMS: List[EvalProblem] = [
    EvalProblem(
        id="p01_add",
        prompt="def add(a, b):\n    return ",
        expected_tail="a + b",
        category="arithmetic",
    ),
    EvalProblem(
        id="p02_mul",
        prompt="def multiply(x, y):\n    return ",
        expected_tail="x * y",
        category="arithmetic",
    ),
    EvalProblem(
        id="p03_is_even",
        prompt="def is_even(n):\n    return ",
        expected_tail="n % 2 == 0",
        category="logic",
    ),
    EvalProblem(
        id="p04_max",
        prompt="def maximum(a, b):\n    if a > b:\n        return a\n    else:\n        return ",
        expected_tail="b",
        category="logic",
    ),
    EvalProblem(
        id="p05_factorial",
        prompt="def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return ",
        expected_tail="n * factorial(n-1)",
        category="recursion",
    ),
    EvalProblem(
        id="p06_reverse",
        prompt="def reverse_string(s):\n    return ",
        expected_tail="s[::-1]",
        category="string",
    ),
    EvalProblem(
        id="p07_sum_list",
        prompt="def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return ",
        expected_tail="total",
        category="list",
    ),
    EvalProblem(
        id="p08_filter_pos",
        prompt="def filter_positive(nums):\n    result = []\n    for n in nums:\n        if n > 0:\n            result.",
        expected_tail="append(n)",
        category="list",
    ),
    EvalProblem(
        id="p09_len",
        prompt="def string_length(s):\n    return ",
        expected_tail="len(s)",
        category="string",
    ),
    EvalProblem(
        id="p10_is_pal",
        prompt="def is_palindrome(s):\n    return ",
        expected_tail="s == s[::-1]",
        category="string",
    ),
    EvalProblem(
        id="p11_count_vowels",
        prompt="def count_vowels(s):\n    count = 0\n    for ch in s:\n        if ch in 'aeiou':\n            count += 1\n    return ",
        expected_tail="count",
        category="string",
    ),
    EvalProblem(
        id="p12_abs_val",
        prompt="def absolute(x):\n    if x < 0:\n        return \n    else:\n        return x",
        expected_tail="-x",
        category="arithmetic",
    ),
    EvalProblem(
        id="p13_caps",
        prompt="def capitalize_words(s):\n    return ",
        expected_tail="s.title()",
        category="string",
    ),
    EvalProblem(
        id="p14_merge",
        prompt="def merge_dicts(d1, d2):\n    result = {}\n    for k, v in d1.items():\n        result[k] = v\n    for k, v in d2.items():\n        result[k] = v\n    return ",
        expected_tail="result",
        category="dict",
    ),
    EvalProblem(
        id="p15_flatten",
        prompt="def flatten_list(nested):\n    flat = []\n    for sub in nested:\n        for item in sub:\n            flat.",
        expected_tail="append(item)",
        category="list",
    ),
    EvalProblem(
        id="p16_dedup",
        prompt="def remove_duplicates(items):\n    return ",
        expected_tail="list(set(items))",
        category="list",
    ),
    EvalProblem(
        id="p17_c2f",
        prompt="def celsius_to_fahrenheit(c):\n    return ",
        expected_tail="c * 9/5 + 32",
        category="arithmetic",
    ),
    EvalProblem(
        id="p18_prime",
        prompt="def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return ",
        expected_tail="False",
        category="logic",
    ),
    EvalProblem(
        id="p19_sort",
        prompt="def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = ",
        expected_tail="arr[j+1], arr[j]",
        category="list",
    ),
    EvalProblem(
        id="p20_mod",
        prompt="def mod(a, b):\n    return ",
        expected_tail="a % b",
        category="arithmetic",
    ),
]

# Problem categories for breakdown reporting
CATEGORIES = sorted({p.category for p in PROBLEMS})


def run_eval(
    generate_fn: Callable[[str], str],
    problems: List[EvalProblem] | None = None,
) -> List[Dict[str, Any]]:
    """Run all problems through a generation function.

    Args:
        generate_fn: Callable taking a prompt string and returning the
            generated completion (prompt NOT included in the return).
        problems: Optional subset of problems to run. Defaults to all 20.

    Returns:
        List of result dicts (one per problem) as returned by EvalProblem.check.
    """
    if problems is None:
        problems = PROBLEMS

    results: List[Dict[str, Any]] = []
    for problem in problems:
        generated = generate_fn(problem.prompt)
        result = problem.check(generated)
        results.append(result)
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a summary dict from run_eval results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "no results"}

    passed = sum(1 for r in results if r.get("passed", False))
    syntax_valid = sum(1 for r in results if r.get("syntax_valid", False))
    expected_present = sum(1 for r in results if r.get("expected_tail_present", False))

    # Per-category breakdown
    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "syntax_valid": 0}
        by_category[cat]["total"] += 1
        if r.get("passed", False):
            by_category[cat]["passed"] += 1
        if r.get("syntax_valid", False):
            by_category[cat]["syntax_valid"] += 1

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "syntax_valid": syntax_valid,
        "syntax_valid_rate": round(syntax_valid / total, 4) if total else 0.0,
        "expected_tail_present": expected_present,
        "expected_tail_rate": round(expected_present / total, 4) if total else 0.0,
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                "syntax_valid": v["syntax_valid"],
                "syntax_valid_rate": round(
                    v["syntax_valid"] / v["total"], 4
                )
                if v["total"]
                else 0.0,
            }
            for cat, v in sorted(by_category.items())
        },
    }


__all__ = ["EvalProblem", "PROBLEMS", "CATEGORIES", "run_eval", "summarize"]
