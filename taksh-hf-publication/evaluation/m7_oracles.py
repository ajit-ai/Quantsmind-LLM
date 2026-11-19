"""Reference-execution oracles for the M7 locked functional probe set.

The oracle contract is deliberately narrow:

* the task names a real function from the held-out source repository;
* the test inputs are hand-specified;
* the expected values were produced by executing the **licensed reference
  implementation** at dataset-build time (see ``data.build_m7_dataset``);
* the candidate is executed under the identical harness, in a separate process,
  with a timeout.

The reference implementation is never shown in the prompt, so copying is not an
available strategy. A candidate that reimplements the specified behaviour passes;
one that emits plausible-looking code with wrong values does not.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile

from data.build_m7_dataset import _HARNESS

TIMEOUT_SECONDS = 15


def extract_function(text, name):
    """Return ``(code, status)`` for a candidate response.

    The M5 extractor already distinguishes ambiguous fences from no code; this
    wrapper adds the one requirement the oracle has: a definition with the
    expected name must exist at module level.
    """
    from evaluation.m5_scorers import extract_code

    code, extraction = extract_code(text)
    if code is None:
        return None, extraction
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code, "SYNTAX_INVALID"
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
        for node in tree.body
    ):
        return code, "FUNCTION_MISSING"
    return code, "OK"


def run_oracle(output, oracle):
    """Run one oracle against model output. Returns a JSON-safe record."""
    name = oracle["function"]
    code, extraction = extract_function(output, name)
    record = {
        "function": name,
        "extraction": extraction,
        "status": "NOT_RUN",
        "passed": 0,
        "attempted": len(oracle["cases"]),
        "cases": [],
    }
    if code is None:
        record["status"] = "NO_CODE"
        return record
    if extraction == "SYNTAX_INVALID":
        record["status"] = "SYNTAX_INVALID"
        return record
    if extraction == "FUNCTION_MISSING":
        record["status"] = "FUNCTION_MISSING"
        return record

    cases = [{"args": case["args"], "kwargs": case["kwargs"]}
             for case in oracle["cases"]]
    script = (
        oracle.get("support", "") + "\n" + code + "\n"
        + _HARNESS.replace("__TARGET__", name)
        .replace("__CASES__", repr(cases))
    )
    env = dict(os.environ)
    env.update({"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
                "NO_PROXY": "*", "no_proxy": "*"})
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", script],
                capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                cwd=tmp, env=env,
            )
    except subprocess.TimeoutExpired:
        record["status"] = "TIMEOUT"
        return record

    if proc.returncode != 0:
        record["status"] = "EXECUTION_ERROR"
        record["error"] = (proc.stderr or proc.stdout).strip()[-300:]
        return record

    try:
        observed = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        record["status"] = "UNPARSEABLE_RESULT"
        record["error"] = proc.stdout.strip()[-300:]
        return record

    passed = 0
    for case, expected, actual in zip(oracle["cases"], expected_of(oracle),
                                      observed):
        ok = actual == expected
        passed += int(ok)
        record["cases"].append({
            "args": case["args"],
            "kwargs": case["kwargs"],
            "expected": expected,
            "actual": actual,
            "pass": ok,
        })
    record["passed"] = passed
    if passed == len(observed) and observed:
        record["status"] = "PASS"
    elif passed:
        record["status"] = "PARTIAL"
    else:
        record["status"] = "FAIL"
    return record


def expected_of(oracle):
    return [case["expected"] for case in oracle["cases"]]
