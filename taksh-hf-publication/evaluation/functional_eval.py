"""Deterministic functional evaluation layer (Taksh v1.2.0 Increment 4).

Functional correctness is measured independently from syntax, structure,
and parameter binding. Generated code is executed only through
:func:`evaluation.safe_execute.execute_tests`, which runs it in an isolated,
bounded subprocess (``python -I``, AST allowlist, 2s timeout, no imports,
no ``open``/``eval``/``exec``). This module never calls ``exec`` or
``eval`` itself and never falls back to in-process execution: anything the
validator rejects, anything that times out, and any subprocess error is a
failure. If the safe boundary cannot be established (validator rejects the
source), execution is refused rather than attempted.

Metrics are reported separately — parse rate, exact match, structural
validity, binding, functional accuracy, functional accuracy conditional on
parsing, functional accuracy conditional on binding, and the
failure-category distribution. They are never combined into one score.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from dataclasses import dataclass

from evaluation.safe_execute import execute_tests, validate_source
from training.program_analysis import analyse

__all__ = [
    "FunctionalResult",
    "evaluate_functional",
    "classify_failure",
    "functional_report",
]

# Failure categories (stable strings; documented in the Increment 4 report).
PASS = "pass"
SYNTAX_STRUCTURAL = "syntax_structural_failure"
BINDING = "incorrect_parameter_binding"
RETURN_EXPR = "incorrect_return_expression"
OPERATION = "incorrect_operation"
IO = "incorrect_input_output_relationship"
TYPE = "incorrect_type_handling"
CONDITIONAL = "missing_conditional_behavior"
COLLECTION = "incorrect_collection_transformation"
PLAUSIBLE = "superficially_plausible_but_functionally_incorrect"
VALIDATION = "rejected_by_safe_execution_gate"
TIMEOUT = "execution_timeout"
EXECUTION_ERROR = "execution_error"


@dataclass(frozen=True)
class FunctionalResult:
    passed: bool
    category: str
    reason: str


def evaluate_functional(source: str, function_name: str, tests: list[dict]) -> FunctionalResult:
    """Execute ``source`` against ``tests`` in the isolated safe gate.

    ``tests`` are dicts with ``args``/``kwargs``/``expected``. Timeouts and
    execution errors are failures, never exceptions.
    """
    _tree, reason = validate_source(source)
    if _tree is None:
        if "syntax error" in reason:
            return FunctionalResult(False, SYNTAX_STRUCTURAL, reason)
        return FunctionalResult(False, VALIDATION, reason)
    result = execute_tests(source, function_name, tests)
    if result.passed:
        return FunctionalResult(True, PASS, result.reason)
    reason = result.reason or ""
    if "timed out" in reason:
        return FunctionalResult(False, TIMEOUT, reason)
    if reason.startswith("AssertionError"):
        return FunctionalResult(False, IO, reason)
    if "NameError" in reason:
        return FunctionalResult(False, BINDING, reason)
    if "TypeError" in reason:
        return FunctionalResult(False, TYPE, reason)
    if "SyntaxError" in reason:
        return FunctionalResult(False, SYNTAX_STRUCTURAL, reason)
    return FunctionalResult(False, EXECUTION_ERROR, reason)


def _parse_signature(signature: str) -> str:
    tree_source = signature + "\n    pass"
    import ast

    return tree_source


def classify_failure(
    signature: str,
    reference_body: str | None,
    completion: str,
    *,
    functional: FunctionalResult | None = None,
    function_name: str | None = None,
    tests: list[dict] | None = None,
) -> str:
    """Classify one generated completion into a failure category.

    ``reference_body`` may be None (static classification only). When
    ``tests`` and ``function_name`` are supplied, functional evidence is
    obtained through the safe gate; otherwise ``functional`` may be passed
    directly. Classification order: syntax, binding, then functional
    shape (conditional / collection / operation / return / plausible).
    """
    expected = analyse(signature, reference_body) if reference_body else None
    indent = completion
    if completion and not completion[0].isspace():
        indent = "\n".join("    " + line if line else line for line in completion.splitlines())
    generated = analyse(signature, indent)
    if not generated["parses"]:
        return SYNTAX_STRUCTURAL
    if generated["undefined_names"]:
        return BINDING
    if expected is not None and expected["parses"]:
        if generated["params_used"] != expected["params_used"]:
            return BINDING
    if functional is None and function_name is not None and tests is not None:
        source = signature + "\n" + indent
        functional = evaluate_functional(source, function_name, tests)
    if functional is not None and functional.passed:
        return PASS
    if functional is not None and functional.category in (TYPE, BINDING, TIMEOUT, VALIDATION, SYNTAX_STRUCTURAL):
        return functional.category
    if expected is not None and expected["parses"]:
        expected_ops = set(expected["operations"])
        generated_ops = set(generated["operations"])
        if ("comparison_gt" in expected_ops) or ("comparison_lt" in expected_ops) or ("if_selection" in expected_ops):
            if "comparison_gt" not in generated_ops and "comparison_lt" not in generated_ops and "if_selection" not in generated_ops:
                return CONDITIONAL
        collection_ops = {"comprehension", "iteration", "length", "count_items", "sum_items"}
        if expected_ops & collection_ops:
            if not (generated_ops & collection_ops):
                return COLLECTION
        if expected_ops and not expected_ops.issubset(generated_ops):
            return OPERATION
    if generated["parses"] and (expected is None or generated["params_used"] == expected["params_used"]):
        # Parses and binds, matches the operation shape, but produces the
        # wrong value: correct shape, wrong behaviour.
        return PLAUSIBLE
    return RETURN_EXPR


def functional_report(records: list[dict]) -> dict:
    """Separate functional metrics for scored evaluation records.

    Records need ``parses``, ``exact``, ``structural``, ``binding``,
    ``functional`` (``{"passed": bool|None}``) and ``failure_category``.
    """
    records = list(records)
    n = len(records)
    parses = sum(bool(r["parses"]) for r in records)
    binding = sum(bool(r["binding"]) for r in records)
    functional_scored = [r for r in records if (r.get("functional") or {}).get("passed") is not None]
    functional_passed = sum(r["functional"]["passed"] is True for r in functional_scored)
    parsed = [r for r in records if r["parses"]]
    parsed_functional = [r for r in parsed if (r.get("functional") or {}).get("passed") is not None]
    bound = [r for r in records if r["binding"]]
    bound_functional = [r for r in bound if (r.get("functional") or {}).get("passed") is not None]
    categories = Counter(r.get("failure_category", "uncategorised") for r in records)
    return {
        "records": n,
        "parse_rate": round(parses / n, 4) if n else 0.0,
        "exact_rate": round(sum(bool(r["exact"]) for r in records) / n, 4) if n else 0.0,
        "structural_rate": round(sum(bool(r["structural"]) for r in records) / n, 4) if n else 0.0,
        "binding_rate": round(binding / n, 4) if n else 0.0,
        "functional_rate": round(functional_passed / len(functional_scored), 4) if functional_scored else None,
        "functional_scored": len(functional_scored),
        "functional_given_parse_count": len(parsed_functional),
        "functional_given_parse": (
            round(sum(r["functional"]["passed"] is True for r in parsed_functional) / len(parsed_functional), 4)
            if parsed_functional
            else None
        ),
        "functional_given_binding_count": len(bound_functional),
        "functional_given_binding": (
            round(sum(r["functional"]["passed"] is True for r in bound_functional) / len(bound_functional), 4)
            if bound_functional
            else None
        ),
        "failure_categories": dict(sorted(categories.items())),
    }


def safe_execution_available() -> bool:
    """True when an isolated interpreter can be started for the safe gate."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", "pass"],
            input="",
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
