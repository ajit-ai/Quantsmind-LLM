"""Versioned evaluation-case schema and validation (Taksh v1.4.0).

A case is data, not code. Every case declares what is asked, how a correct
answer is decided, and where the case came from, so a report can state its
own provenance and denominators instead of asserting a number.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.python_eval import (
    CATEGORIES,
    CONTAMINATION_CLASSES,
    EXECUTION_ENVIRONMENTS,
    SCHEMA_VERSION,
)

REQUIRED_FIELDS = (
    "case_id",
    "category",
    "language",
    "prompt",
    "prompt_format",
    "expected_contract",
    "validator",
    "budget",
    "provenance",
    "contamination",
    "execution_environment",
)

VALIDATOR_KINDS = ("syntax_only", "exact_match", "contains_all", "regex")

#: Prompt surfaces a case may use. A code-completion prompt is raw source; a
#: chat-instruction prompt is a natural-language request. Each is only
#: meaningful for a checkpoint qualified on that surface.
PROMPT_FORMATS = ("code_completion", "chat_instruction")


class CaseError(ValueError):
    """A case is malformed."""


def validate_case(case: dict, index: int = 0) -> list:
    """Return every problem with a case. An empty list means usable."""
    problems: list = []
    if not isinstance(case, dict):
        return [f"case {index} must be a mapping"]
    for field in REQUIRED_FIELDS:
        if field not in case:
            problems.append(f"case {index}: missing '{field}'")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        problems.append(f"case {index}: case_id must be a non-empty string")
    if case.get("category") not in CATEGORIES:
        problems.append(
            f"case {index}: category {case.get('category')!r} is not one of {list(CATEGORIES)}"
        )
    if not isinstance(case.get("language"), str) or not case.get("language"):
        problems.append(f"case {index}: language must be a non-empty string")
    if not isinstance(case.get("prompt"), str) or not case.get("prompt", "").strip():
        problems.append(f"case {index}: prompt must be a non-empty string")
    if case.get("prompt_format") not in PROMPT_FORMATS:
        problems.append(
            f"case {index}: prompt_format must be one of {list(PROMPT_FORMATS)}"
        )
    validator = case.get("validator")
    if not isinstance(validator, dict):
        problems.append(f"case {index}: validator must be a mapping")
    else:
        if validator.get("kind") not in VALIDATOR_KINDS:
            problems.append(
                f"case {index}: validator.kind must be one of {list(VALIDATOR_KINDS)}"
            )
        if validator.get("kind") == "exact_match" and not validator.get("expected"):
            problems.append(f"case {index}: exact_match validator needs 'expected'")
        if validator.get("kind") == "contains_all" and not validator.get("contains"):
            problems.append(f"case {index}: contains_all validator needs 'contains'")
        if validator.get("kind") == "regex" and not validator.get("pattern"):
            problems.append(f"case {index}: regex validator needs 'pattern'")
    budget = case.get("budget")
    if not isinstance(budget, dict):
        problems.append(f"case {index}: budget must be a mapping")
    else:
        if int(budget.get("timeout_seconds", 0) or 0) <= 0:
            problems.append(f"case {index}: budget.timeout_seconds must be positive")
        if int(budget.get("max_output_bytes", 0) or 0) <= 0:
            problems.append(f"case {index}: budget.max_output_bytes must be positive")
    provenance = case.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("source"):
        problems.append(f"case {index}: provenance.source is required")
    if case.get("contamination") not in CONTAMINATION_CLASSES:
        problems.append(
            f"case {index}: contamination must be one of {list(CONTAMINATION_CLASSES)}"
        )
    if case.get("execution_environment") not in EXECUTION_ENVIRONMENTS:
        problems.append(
            f"case {index}: execution_environment must be one of {list(EXECUTION_ENVIRONMENTS)}"
        )
    return problems


def validate_suite(cases: list) -> list:
    """Validate a whole suite, including case-id uniqueness."""
    problems: list = []
    seen = set()
    for index, case in enumerate(cases):
        problems.extend(validate_case(case, index))
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if case_id in seen:
            problems.append(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
    return problems


def load_cases(path) -> list:
    """Load and validate a case file.

    The file is a JSON object with ``schema`` and ``cases``; a bare list is
    also accepted for convenience.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        cases = data
        schema = SCHEMA_VERSION
    else:
        schema = data.get("schema")
        cases = data.get("cases") or []
    if schema != SCHEMA_VERSION:
        raise CaseError(f"{path}: schema must be {SCHEMA_VERSION!r}, got {schema!r}")
    problems = validate_suite(cases)
    if problems:
        raise CaseError(f"{path}: invalid suite: " + "; ".join(problems))
    return cases
