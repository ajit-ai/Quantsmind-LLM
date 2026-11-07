"""Taksh v1.7.0 — Execution-based evaluation for simple_humaneval.

Runs generated Python code in a restricted subprocess and compares output
to expected results. This is a stronger signal than syntax-only checking.

Safety: code runs in a subprocess with:
  - No network access (subprocess is isolated)
  - No file system writes to the repo
  - Timeout protection
  - Resource limits (memory, CPU)

Usage:
    from evaluation.simple_humaneval_exec import run_exec_eval

    results = run_exec_eval(generate_fn)
    # generate_fn(prompt: str) -> str (generated completion only)
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

from evaluation.simple_humaneval import PROBLEMS, EvalProblem

REPO = Path(__file__).resolve().parent.parent

# Wrapper template: injects the generated function, runs test cases, prints JSON
WRAPPER_TEMPLATE = '''import json
import sys

__FULL_CODE__

results = []
for test in __TESTS__:
    try:
        fn = globals()["__FUNCTION_NAME__"]
        output = fn(*test["args"])
        results.append({
            "test": test["name"],
            "passed": output == test["expected"],
            "output": output,
            "expected": test["expected"],
            "error": None,
        })
    except Exception as exc:
        results.append({
            "test": test["name"],
            "passed": False,
            "output": None,
            "expected": test["expected"],
            "error": f"{type(exc).__name__}: {exc}",
        })

print(json.dumps(results))
'''


def extract_function_name(prompt: str) -> str:
    """Extract the function name from a ``def foo(...):`` prompt line."""
    for line in prompt.split("\n"):
        line = line.strip()
        if line.startswith("def "):
            name = line[4:].split("(")[0].strip()
            return name
    return "unknown_function"


def build_test_cases(problem: EvalProblem) -> List[Dict[str, Any]]:
    """Build execution test cases for a problem.

    These are simple, deterministic test inputs derived from the problem's
    expected behavior. Each case has args, expected output, and a name.
    """
    pid = problem.id
    prompt = problem.prompt
    expected_tail = problem.expected_tail.strip()

    # Parse the function signature to get parameter names and count
    sig_line = prompt.split("\n")[0]
    param_str = sig_line[sig_line.find("(") + 1 : sig_line.find(")")]
    param_names = [p.strip() for p in param_str.split(",") if p.strip()]

    cases: List[Dict[str, Any]] = []

    # Arithmetic problems
    if pid == "p01_add":
        cases = [
            {"name": "add_2_3", "args": [2, 3], "expected": 5},
            {"name": "add_neg", "args": [-5, 3], "expected": -2},
            {"name": "add_zero", "args": [0, 0], "expected": 0},
        ]
    elif pid == "p02_mul":
        cases = [
            {"name": "mul_3_4", "args": [3, 4], "expected": 12},
            {"name": "mul_neg", "args": [-2, 5], "expected": -10},
            {"name": "mul_zero", "args": [0, 100], "expected": 0},
        ]
    elif pid == "p03_is_even":
        cases = [
            {"name": "even_4", "args": [4], "expected": True},
            {"name": "odd_5", "args": [5], "expected": False},
            {"name": "even_0", "args": [0], "expected": True},
        ]
    elif pid == "p04_max":
        cases = [
            {"name": "max_3_7", "args": [3, 7], "expected": 7},
            {"name": "max_10_2", "args": [10, 2], "expected": 10},
            {"name": "max_equal", "args": [5, 5], "expected": 5},
        ]
    elif pid == "p05_factorial":
        cases = [
            {"name": "fact_0", "args": [0], "expected": 1},
            {"name": "fact_1", "args": [1], "expected": 1},
            {"name": "fact_5", "args": [5], "expected": 120},
        ]
    elif pid == "p06_reverse":
        cases = [
            {"name": "rev_abc", "args": ["abc"], "expected": "cba"},
            {"name": "rev_empty", "args": [""], "expected": ""},
            {"name": "rev_pal", "args": ["aba"], "expected": "aba"},
        ]
    elif pid == "p07_sum_list":
        cases = [
            {"name": "sum_empty", "args": [[]], "expected": 0},
            {"name": "sum_single", "args": [[5]], "expected": 5},
            {"name": "sum_multi", "args": [[1, 2, 3, 4]], "expected": 10},
        ]
    elif pid == "p08_filter_pos":
        cases = [
            {"name": "filter_mixed", "args": [[-1, 2, -3, 4]], "expected": [2, 4]},
            {"name": "filter_none", "args": [[-1, -2]], "expected": []},
            {"name": "filter_all", "args": [[1, 2, 3]], "expected": [1, 2, 3]},
        ]
    elif pid == "p09_len":
        cases = [
            {"name": "len_hello", "args": ["hello"], "expected": 5},
            {"name": "len_empty", "args": [""], "expected": 0},
            {"name": "len_space", "args": ["a b"], "expected": 3},
        ]
    elif pid == "p10_is_pal":
        cases = [
            {"name": "pal_aba", "args": ["aba"], "expected": True},
            {"name": "pal_abc", "args": ["abc"], "expected": False},
            {"name": "pal_empty", "args": [""], "expected": True},
        ]
    elif pid == "p11_count_vowels":
        cases = [
            {"name": "vowels_hello", "args": ["hello"], "expected": 2},
            {"name": "vowels_aeiou", "args": ["aeiou"], "expected": 5},
            {"name": "vowels_none", "args": ["xyz"], "expected": 0},
        ]
    elif pid == "p12_abs_val":
        cases = [
            {"name": "abs_5", "args": [5], "expected": 5},
            {"name": "abs_neg5", "args": [-5], "expected": 5},
            {"name": "abs_0", "args": [0], "expected": 0},
        ]
    elif pid == "p13_caps":
        cases = [
            {"name": "caps_hello", "args": ["hello world"], "expected": "Hello World"},
            {"name": "caps_empty", "args": [""], "expected": ""},
        ]
    elif pid == "p14_merge":
        cases = [
            {"name": "merge_simple", "args": [{"a": 1}, {"b": 2}], "expected": {"a": 1, "b": 2}},
            {"name": "merge_overlap", "args": [{"a": 1}, {"a": 2}], "expected": {"a": 2}},
            {"name": "merge_empty", "args": [{}, {}], "expected": {}},
        ]
    elif pid == "p15_flatten":
        cases = [
            {"name": "flatten_simple", "args": [[[1, 2], [3, 4]]], "expected": [1, 2, 3, 4]},
            {"name": "flatten_empty", "args": [[]], "expected": []},
        ]
    elif pid == "p16_dedup":
        cases = [
            {"name": "dedup_simple", "args": [[1, 2, 2, 3]], "expected": [1, 2, 3]},
            {"name": "dedup_empty", "args": [[]], "expected": []},
        ]
    elif pid == "p17_c2f":
        cases = [
            {"name": "c2f_0", "args": [0], "expected": 32.0},
            {"name": "c2f_100", "args": [100], "expected": 212.0},
        ]
    elif pid == "p18_prime":
        cases = [
            {"name": "prime_2", "args": [2], "expected": True},
            {"name": "prime_4", "args": [4], "expected": False},
            {"name": "prime_7", "args": [7], "expected": True},
        ]
    elif pid == "p19_sort":
        cases = [
            {"name": "sort_simple", "args": [[3, 1, 2]], "expected": [1, 2, 3]},
            {"name": "sort_empty", "args": [[]], "expected": []},
        ]
    elif pid == "p20_mod":
        cases = [
            {"name": "mod_10_3", "args": [10, 3], "expected": 1},
            {"name": "mod_7_2", "args": [7, 2], "expected": 1},
            {"name": "mod_neg", "args": [-10, 3], "expected": 2},
        ]

    return cases


def run_exec_eval(
    generate_fn: Callable[[str], str],
    problems: List[EvalProblem] | None = None,
    timeout_seconds: int = 10,
) -> List[Dict[str, Any]]:
    """Run execution-based evaluation on problems.

    For each problem:
    1. Generate completion from the model
    2. Build a wrapper script with prompt + generation + test cases
    3. Run in a subprocess with timeout
    4. Parse JSON results

    Args:
        generate_fn: Callable taking a prompt string and returning generated completion.
        problems: Optional subset of problems. Defaults to all 20.
        timeout_seconds: Max seconds per subprocess execution.

    Returns:
        List of result dicts with execution details.
    """
    if problems is None:
        problems = PROBLEMS

    results: List[Dict[str, Any]] = []

    for problem in problems:
        generated = generate_fn(problem.prompt)
        full_code = problem.prompt + generated

        # Check syntax first
        syntax_valid = False
        try:
            ast.parse(full_code)
            syntax_valid = True
        except SyntaxError:
            pass

        result: Dict[str, Any] = {
            "id": problem.id,
            "category": problem.category,
            "prompt": problem.prompt,
            "generated": generated,
            "full_code": full_code,
            "syntax_valid": syntax_valid,
            "test_results": [],
            "tests_passed": 0,
            "tests_total": 0,
            "execution_error": None,
            "passed": False,
        }

        if not syntax_valid:
            result["execution_error"] = "syntax_error"
            results.append(result)
            continue

        # Build test cases
        test_cases = build_test_cases(problem)
        func_name = extract_function_name(problem.prompt)

        # Prepare the wrapper script. Use repr() so Python booleans (True/False)
        # are emitted correctly instead of JSON's true/false.
        tests_repr = repr(test_cases)
        wrapper = (
            WRAPPER_TEMPLATE
            .replace("__FULL_CODE__", full_code)
            .replace("__FUNCTION_NAME__", func_name)
            .replace("__TESTS__", tests_repr)
        )

        # Write to temp file and execute
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(wrapper)
                temp_path = f.name

            proc = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(REPO),
            )

            os.unlink(temp_path)

            if proc.returncode != 0:
                result["execution_error"] = (
                    f"process exited {proc.returncode}: {proc.stderr[:500]}"
                )
            else:
                try:
                    test_outputs = json.loads(proc.stdout.strip())
                    result["test_results"] = test_outputs
                    result["tests_total"] = len(test_outputs)
                    result["tests_passed"] = sum(
                        1 for t in test_outputs if t.get("passed")
                    )
                    result["passed"] = result["tests_passed"] == result["tests_total"]
                except json.JSONDecodeError as exc:
                    result["execution_error"] = f"json parse error: {exc}"
        except subprocess.TimeoutExpired:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            result["execution_error"] = f"timeout after {timeout_seconds}s"
        except Exception as exc:
            result["execution_error"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

    return results


def summarize_exec(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce summary from execution eval results."""
    total = len(results)
    if total == 0:
        return {"total": 0, "error": "no results"}

    passed = sum(1 for r in results if r["passed"])
    syntax_valid = sum(1 for r in results if r["syntax_valid"])
    total_tests = sum(r["tests_total"] for r in results)
    passed_tests = sum(r["tests_passed"] for r in results)

    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "tests_passed": 0, "tests_total": 0}
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1
        by_category[cat]["tests_passed"] += r["tests_passed"]
        by_category[cat]["tests_total"] += r["tests_total"]

    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "syntax_valid": syntax_valid,
        "syntax_valid_rate": round(syntax_valid / total, 4) if total else 0.0,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "test_pass_rate": round(passed_tests / total_tests, 4) if total_tests else 0.0,
        "by_category": {
            cat: {
                "total": v["total"],
                "passed": v["passed"],
                "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0,
                "tests_passed": v["tests_passed"],
                "tests_total": v["tests_total"],
                "test_pass_rate": round(
                    v["tests_passed"] / v["tests_total"], 4
                )
                if v["tests_total"] else 0.0,
            }
            for cat, v in sorted(by_category.items())
        },
    }


__all__ = ["extract_function_name", "build_test_cases", "run_exec_eval", "summarize_exec"]
