"""Case-by-case evaluation runner (Taksh v1.4.0).

The runner is deliberately explicit: it calls a generator, extracts the code,
validates the syntax, optionally attempts functional execution under the
sandbox policy, applies the case's own validator, and records what happened
with its denominators.

It never executes generated code itself. Functional results are only produced
when the sandbox policy provides an external isolation mechanism; otherwise
the case is recorded as ``blocked`` and the report says so.
"""

from __future__ import annotations

import time
from pathlib import Path

from evaluation.python_eval import REPORT_SCHEMA_VERSION
from evaluation.python_eval.metrics import aggregate
from evaluation.python_eval.sandbox import default_policy, execute_python
from evaluation.python_eval.syntax import extract_code_block, validate_syntax


def _apply_validator(validator: dict, code: str, stdout: str) -> bool:
    kind = validator.get("kind")
    if kind == "syntax_only":
        return True
    if kind == "exact_match":
        return code.strip() == str(validator.get("expected", "")).strip()
    if kind == "contains_all":
        return all(str(item) in code for item in validator.get("contains", []))
    if kind == "regex":
        import re

        return bool(re.search(str(validator.get("pattern")), code))
    return False


def run_case(case: dict, generator, *, sandbox_policy=None) -> dict:
    """Execute one case and return its record."""
    policy = sandbox_policy or default_policy()
    budget = case.get("budget") or {}
    max_new_tokens = int(budget.get("max_new_tokens", 96))
    prompt = case["prompt"]
    started = time.time()
    refusal = None
    finish_reason = None
    tokens = 0
    raw = ""
    try:
        result = generator.generate(
            prompt, max_new_tokens=max_new_tokens, format=case.get("prompt_format")
        )
        raw = result.get("text", "")
        finish_reason = result.get("finish_reason")
        tokens = int(result.get("completion_tokens", 0))
        refusal = result.get("refusal")
    except Exception as exc:  # noqa: BLE001 - a failed case is data, not a crash
        return {
            "case_id": case.get("case_id"),
            "category": case.get("category"),
            "language": case.get("language"),
            "outcome": "fail",
            "failure_category": "generator_error",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": time.time() - started,
            "syntax_valid": None,
            "functional": None,
            "format_compliant": None,
            "tokens_generated": 0,
            "timed_out": False,
            "refused": False,
        }
    latency = time.time() - started

    code = extract_code_block(raw)
    syntax_ok, syntax_category, syntax_detail = validate_syntax(code, case["language"])
    functional = None
    failure_category = None
    if not syntax_ok:
        failure_category = "syntax_error"
    else:
        validator = case.get("validator") or {}
        needs_execution = validator.get("kind") not in ("syntax_only",)
        if needs_execution and case.get("execution_environment") != "none":
            execution = execute_python(code, policy)
            if execution.blocked:
                functional = None
                failure_category = "functional_blocked"
            elif execution.timed_out:
                functional = False
                failure_category = "timeout"
            else:
                functional = execution.ok and _apply_validator(
                    validator, code, execution.stdout
                )
                if not functional:
                    failure_category = "functional_fail"
        if failure_category is None:
            passed = _apply_validator(validator, code, "")
            if not passed:
                failure_category = "contract_fail"
    if failure_category is None:
        outcome = "pass"
    elif failure_category == "functional_blocked":
        outcome = "blocked"
    else:
        outcome = "fail"

    return {
        "case_id": case.get("case_id"),
        "category": case.get("category"),
        "language": case.get("language"),
        "prompt_format": case.get("prompt_format"),
        "outcome": outcome,
        "failure_category": failure_category,
        "syntax_valid": bool(syntax_ok),
        "syntax_detail": syntax_detail,
        "functional": functional,
        "format_compliant": case.get("category") == "format_compliance"
        and failure_category is None,
        "tokens_generated": tokens,
        "finish_reason": finish_reason,
        "refused": bool(refusal),
        "timed_out": finish_reason == "timeout",
        "latency_seconds": latency,
        "raw_output": raw,
    }


def evaluate(cases: list, generator, *, model_identity: dict, sandbox_policy=None,
             repeats: int = 1) -> dict:
    """Run a suite against one model and return the report."""
    policy = sandbox_policy or default_policy()
    all_records = []
    for _ in range(max(1, repeats)):
        for case in cases:
            all_records.append(run_case(case, generator, sandbox_policy=policy))
    primary = all_records[: len(cases)]
    metrics = aggregate(primary)
    reproducibility = None
    if repeats > 1 and all_records:
        keys = sorted(
            {(r["case_id"], r["outcome"]) for r in all_records}
        )
        per_case = defaultdict(set)
        for record in all_records:
            per_case[record["case_id"]].add(
                (record["outcome"], record.get("syntax_valid"))
            )
        reproducibility = {
            "repeats": repeats,
            "cases_with_identical_outcome": sum(
                1 for outcomes in per_case.values() if len(outcomes) == 1
            ),
            "cases_total": len(per_case),
            "deterministic": all(len(v) == 1 for v in per_case.values()),
        }
    return {
        "schema": REPORT_SCHEMA_VERSION,
        "model": model_identity,
        "sandbox": policy.to_dict(),
        "metrics": metrics,
        "reproducibility": reproducibility,
        "records": primary,
        "all_records": all_records,
    }
