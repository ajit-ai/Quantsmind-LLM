"""M5 scorers.

Each scorer is independently testable, reports its own numerator and
denominator, and never merges into a single opaque number. Where an example
cannot be scored automatically, it is **excluded** and the exclusion is counted
-- never scored as a failure.

Syntax scoring deliberately reuses the M4 fix: a comment-only or prose-only
answer is not executable Python, even though it parses. That defect produced a
spurious 72% baseline in M3 and must not return.

Functional scoring runs authored assertions in a subprocess with a timeout.
**This is process isolation, not a security sandbox**, and is labelled that way
in every report. Only test specifications this repository authored are run.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from evaluation.code_check import (
    EXTRACTION_AMBIGUOUS,
    NO_CODE,
    SAFE_TESTS,
    _FENCE_RE,
    _has_executable_content,
)
from evaluation.m5_schema import TASK_METRICS

SCORING_VERSION = "quantsmind.m5_scoring/1"
TEST_TIMEOUT_SECONDS = 15

SYNTAX_VALID = "SYNTAX_VALID"
SYNTAX_INVALID = "SYNTAX_INVALID"
NO_EXECUTABLE_CODE = "NO_EXECUTABLE_CODE"
TEST_PASS = "TEST_PASS"
TEST_FAIL = "TEST_FAIL"
TEST_TIMEOUT = "TEST_TIMEOUT"
NOT_TESTABLE = "NOT_TESTABLE"


def extract_code(text):
    """Return ``(code, status)``. Ambiguity is reported, never resolved."""
    if not isinstance(text, str) or not text.strip():
        return None, NO_CODE
    blocks = [b.strip() for b in _FENCE_RE.findall(text) if b.strip()]
    if len(blocks) > 1:
        return None, EXTRACTION_AMBIGUOUS
    if len(blocks) == 1:
        # A fence asserts "this is my code", so it is returned even when it
        # fails to parse: SYNTAX_INVALID must be distinguishable from NO_CODE.
        return blocks[0], "SINGLE_BLOCK"
    candidate = text.strip()
    try:
        tree = ast.parse(candidate)
    except (SyntaxError, ValueError):
        return None, NO_CODE
    if not _has_executable_content(tree):
        return None, NO_CODE
    return candidate, "BARE_CODE"


def score_syntax(output, language="python"):
    """Syntax validity of the *executable* content, not merely parseable text.

    Only Python is scored. Another language is reported ``not_checked`` rather
    than assumed valid.
    """
    if (language or "python") != "python":
        return {"metric": "syntax", "status": "not_checked", "score": None,
                "denominator": 0, "numerator": 0,
                "detail": f"no syntax checker available for {language!r}; "
                          "not claimed as valid"}
    code, extraction = extract_code(output)
    if code is None:
        status = (EXTRACTION_AMBIGUOUS if extraction == EXTRACTION_AMBIGUOUS
                  else NO_EXECUTABLE_CODE)
        return {"metric": "syntax", "status": status, "score": 0.0,
                "denominator": 1, "numerator": 0,
                "detail": f"extraction: {extraction}"}
    try:
        ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return {"metric": "syntax", "status": SYNTAX_INVALID, "score": 0.0,
                "denominator": 1, "numerator": 0, "detail": str(exc)}
    return {"metric": "syntax", "status": SYNTAX_VALID, "score": 1.0,
            "denominator": 1, "numerator": 1,
            "detail": "parses and contains executable content"}


def run_functional_test(code, assertions, timeout=TEST_TIMEOUT_SECONDS):
    """Execute authored assertions in a subprocess with a timeout.

    Process isolation with a timeout -- **not** a security sandbox. No network
    or filesystem isolation is claimed.
    """
    program = f"{code}\n\n{assertions}\nprint('__QM_TEST_OK__')\n"
    handle = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                        encoding="utf-8")
    handle.write(program)
    handle.close()
    try:
        completed = subprocess.run(
            [sys.executable, handle.name], capture_output=True, text=True,
            timeout=timeout, cwd=tempfile.gettempdir(),
        )
    except subprocess.TimeoutExpired:
        return {"status": TEST_TIMEOUT, "detail": f"timed out after {timeout}s"}
    finally:
        Path(handle.name).unlink(missing_ok=True)

    if completed.returncode == 0 and "__QM_TEST_OK__" in (completed.stdout or ""):
        return {"status": TEST_PASS, "detail": "all assertions passed"}
    tail = (completed.stderr or "").strip().splitlines()
    return {"status": TEST_FAIL,
            "detail": tail[-1] if tail else f"exit {completed.returncode}"}


def score_functional(output, test_specification, language="python"):
    """Functional correctness, or an explicit exclusion when untestable."""
    if not test_specification:
        return {"metric": "functional", "status": NOT_TESTABLE, "score": None,
                "denominator": 0, "numerator": 0,
                "detail": "no trustworthy oracle; excluded, not scored"}
    if (language or "python") != "python":
        return {"metric": "functional", "status": NOT_TESTABLE, "score": None,
                "denominator": 0, "numerator": 0,
                "detail": "functional execution implemented for Python only"}

    assertions = (SAFE_TESTS[test_specification]
                  if test_specification in SAFE_TESTS else test_specification)
    code, extraction = extract_code(output)
    if code is None:
        return {"metric": "functional", "status": NOT_TESTABLE, "score": None,
                "denominator": 0, "numerator": 0,
                "detail": f"not run: no executable code ({extraction})"}
    outcome = run_functional_test(code, assertions)
    passed = outcome["status"] == TEST_PASS
    return {"metric": "functional", "status": outcome["status"],
            "score": 1.0 if passed else 0.0, "denominator": 1,
            "numerator": 1 if passed else 0, "detail": outcome["detail"]}


def _check_required_keys(text, keys):
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"parse failure: {exc.msg}"
    if not isinstance(parsed, dict):
        return False, f"expected a JSON object, got {type(parsed).__name__}"
    missing = [k for k in keys if k not in parsed]
    if missing:
        return False, f"schema failure: missing keys {missing}"
    return True, "parsed and all required keys present"


def _check_text_rule(kind, text, spec):
    """Objective rules only. No subjective rubric is implemented here."""
    if kind == "non_empty":
        return bool(text), "non-empty" if text else "output was empty"
    if kind == "no_fences":
        ok = "```" not in text
        return ok, "no markdown fences" if ok else "output contained fences"
    if kind == "contains":
        needle = spec.get("value", "")
        return needle in text, f"contains {needle!r}: {needle in text}"
    if kind == "not_contains":
        needle = spec.get("value", "")
        ok = needle not in text
        return ok, f"excludes {needle!r}: {ok}"
    if kind == "max_lines":
        limit = int(spec.get("value", 1))
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return len(lines) <= limit, f"{len(lines)} non-empty line(s), limit {limit}"
    if kind == "line_starts_with":
        prefix = spec.get("value", "")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        ok = bool(lines) and all(ln.startswith(prefix) for ln in lines)
        return ok, f"{len(lines)} line(s) all start with {prefix!r}: {ok}"
    if kind == "equals":
        expected = str(spec.get("value", "")).strip()
        ok = text.strip() == expected
        return ok, f"equals {expected!r}: {ok}"
    return False, f"unknown rule {kind!r}"


CHECKLIST_TYPES = {
    "json_keys", "contains", "not_contains", "no_fences", "non_empty",
    "max_lines", "line_starts_with", "equals",
}


def score_structured(output, spec):
    """Structured-output validity.

    Parse failure and schema failure are reported separately, and neither is
    reported as semantic correctness.
    """
    if not isinstance(spec, dict) or "type" not in spec:
        return {"metric": "structured", "status": NOT_TESTABLE, "score": None,
                "denominator": 0, "numerator": 0,
                "detail": "no structured specification"}
    kind = spec["type"]
    text = (output or "").strip()
    if kind == "json_keys":
        ok, detail = _check_required_keys(text, spec.get("keys") or [])
        return {"metric": "structured",
                "status": "SCHEMA_VALID" if ok else "SCHEMA_INVALID",
                "score": 1.0 if ok else 0.0, "denominator": 1,
                "numerator": 1 if ok else 0, "detail": detail}
    if kind in CHECKLIST_TYPES:
        ok, detail = _check_text_rule(kind, text, spec)
        return {"metric": "structured",
                "status": "VALID" if ok else "INVALID",
                "score": 1.0 if ok else 0.0, "denominator": 1,
                "numerator": 1 if ok else 0, "detail": detail}
    return {"metric": "structured", "status": NOT_TESTABLE, "score": None,
            "denominator": 0, "numerator": 0,
            "detail": f"unsupported structured check {kind!r}"}


def score_compliance(output, checklist):
    """Instruction compliance against explicit, machine-checkable rules."""
    if not checklist:
        return {"metric": "compliance", "status": NOT_TESTABLE, "score": None,
                "denominator": 0, "numerator": 0,
                "detail": "no machine-checkable checklist"}
    text = (output or "").strip()
    results, passed = [], 0
    for item in checklist:
        if isinstance(item, str):
            item = {"type": item}
        if not isinstance(item, dict) or item.get("type") not in CHECKLIST_TYPES:
            results.append({"rule": item, "ok": False,
                            "detail": "unsupported or malformed rule"})
            continue
        ok, detail = _check_text_rule(item["type"], text, item)
        results.append({"rule": item["type"], "ok": ok, "detail": detail})
        passed += 1 if ok else 0
    return {
        "metric": "compliance",
        "status": "COMPLIANT" if passed == len(checklist) else "NON_COMPLIANT",
        "score": round(passed / len(checklist), 6),
        "denominator": len(checklist), "numerator": passed,
        "checks": results, "detail": f"{passed}/{len(checklist)} rules satisfied",
    }


def score_example(record, output):
    """Score one generation. Each applicable metric is reported separately."""
    task_type = record.get("task_type")
    language = record.get("language") or "python"
    applicable = TASK_METRICS.get(task_type, ())
    metrics = {}

    if "syntax" in applicable:
        metrics["syntax"] = score_syntax(output, language)
    if "functional" in applicable:
        metrics["functional"] = score_functional(
            output, record.get("test_specification"), language)
    if "structured" in applicable:
        metrics["structured"] = score_structured(output, record.get("checklist"))
    if "compliance" in applicable:
        metrics["compliance"] = score_compliance(output, record.get("checklist"))

    metrics["human_review"] = {
        "metric": "human_review", "status": "REQUIRES_HUMAN_REVIEW",
        "score": None, "denominator": 0, "numerator": 0,
        "detail": "no automatic oracle; excluded from every automated score",
    }
    return metrics


def aggregate(metrics_by_example):
    """Aggregate with explicit denominators. No composite score is produced.

    A blended number would hide that a model can score well on format
    compliance while being useless at code -- the distinction this milestone
    exists to preserve.
    """
    totals = {}
    for example in metrics_by_example:
        for name, metric in example.items():
            slot = totals.setdefault(name, {
                "numerator": 0, "denominator": 0, "excluded": 0,
                "status_counts": Counter(),
            })
            slot["status_counts"][metric["status"]] += 1
            slot["denominator"] += metric["denominator"]
            slot["numerator"] += metric["numerator"]
            if metric["score"] is None:
                slot["excluded"] += 1

    summary = {}
    for name, slot in sorted(totals.items()):
        denominator = slot["denominator"]
        rate = (slot["numerator"] / denominator) if denominator else None
        summary[name] = {
            "numerator": slot["numerator"],
            "denominator": denominator,
            "rate": round(rate, 6) if rate is not None else None,
            "excluded_unscorable": slot["excluded"],
            "status_counts": dict(slot["status_counts"]),
            "preliminary": 0 < denominator < 10,
            "note": (
                "denominator below 10: preliminary, not a strong result"
                if 0 < denominator < 10 else
                ("no eligible examples; no score is defined" if denominator == 0
                 else "")
            ),
        }
    return summary