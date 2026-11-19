"""Instruction-tuning dataset contract (M4).

A canonical instruction record::

    {
      "id": "qm-code-000001",
      "instruction": "Write a Python function that calculates factorial.",
      "input": "",
      "response": "def factorial(n): ...",
      "category": "coding",
      "task": "code_generation",
      "language": "python",
      "source": "psf-requests",
      "license": "Apache-2.0",
      "source_sha": "...",
      "split": "train",
      "transform": "qm.instruction/1"
    }

Design rules that this module enforces rather than documents:

* ``id`` is stable and derived from content, so rebuilding the dataset does not
  renumber examples.
* Every record carries provenance and licence. A record without them is invalid,
  not "probably fine" — M4 will not train on data it cannot attribute.
* ``language`` is validated against a fixed identifier set, not free text.
* ``category``/``task`` come from closed vocabularies, so evaluation can group
  results without string guessing.

This module validates. It does not build; see
:mod:`data.build_instruction_dataset`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = "quantsmind.instruction/1"
TRANSFORM = "qm.instruction/1"

CATEGORIES = ("coding", "technical", "instruction_following", "structured_output")

TASKS = (
    # coding
    "code_generation",
    "code_completion",
    "code_explanation",
    "debugging",
    "refactoring",
    "test_generation",
    "documentation",
    # technical
    "sql",
    "data_engineering",
    "devops",
    "architecture",
    "api_explanation",
    # instruction following
    "format_following",
    "constrained_output",
    "multi_step_instruction",
    # structured output
    "json_output",
    "config_output",
)

LANGUAGES = ("python", "javascript", "typescript", "sql", "java", "yaml", "json")

SPLITS = ("train", "val", "test")

_ID_RE = re.compile(r"^qm-[a-z0-9]+-[0-9a-f]{12}$")


class InstructionValidationError(ValueError):
    """A record does not satisfy the instruction dataset contract."""


def stable_id(task: str, instruction: str, response: str) -> str:
    """Content-addressed id: stable across rebuilds, collisions across examples."""
    digest = hashlib.sha256(
        f"{task}\x00{instruction}\x00{response}".encode("utf-8")
    ).hexdigest()[:12]
    return f"qm-{_slug(task)}-{digest}"


def _slug(task: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", task.lower())[:16] or "task"


def validate_record(record, *, require_provenance=True):
    """Return a list of problems. Empty means the record is valid.

    Never raises: a caller validating 10,000 records wants every problem, not
    the first one.
    """
    problems = []

    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not an object"]

    for field in ("id", "instruction", "response", "category", "task"):
        if field not in record:
            problems.append(f"missing required field {field!r}")
    if problems:
        return problems

    if not isinstance(record["id"], str) or not _ID_RE.match(record["id"]):
        problems.append(
            f"id {record['id']!r} does not match the stable form "
            "qm-<task>-<12 hex chars>"
        )

    for field in ("instruction", "response"):
        value = record[field]
        if not isinstance(value, str):
            problems.append(f"{field} must be a string, got {type(value).__name__}")
        elif not value.strip():
            problems.append(f"{field} is empty; an empty response cannot be supervised")

    instruction = record.get("instruction")
    if isinstance(instruction, str) and len(instruction) > 4000:
        problems.append(
            f"instruction is {len(instruction)} chars; over the 4000-char limit"
        )

    value = record.get("input")
    if value is not None and not isinstance(value, str):
        problems.append(f"input must be a string when present, got {type(value).__name__}")

    if record.get("category") not in CATEGORIES:
        problems.append(
            f"category {record.get('category')!r} is not one of {list(CATEGORIES)}"
        )
    if record.get("task") not in TASKS:
        problems.append(f"task {record.get('task')!r} is not one of {list(TASKS)}")

    language = record.get("language")
    if language is not None:
        if language not in LANGUAGES:
            problems.append(
                f"language {language!r} is not one of {list(LANGUAGES)}"
            )
        elif record.get("category") == "coding" and language is None:
            problems.append("coding records must declare a language")

    if record.get("split") not in SPLITS:
        problems.append(f"split {record.get('split')!r} is not one of {list(SPLITS)}")

    if require_provenance:
        for field in ("source", "license"):
            if not record.get(field):
                problems.append(f"missing provenance field {field!r}")

    return problems


def is_valid_python(response: str) -> bool:
    """Syntax check used by the dataset quality pass, not a capability claim."""
    import ast

    try:
        ast.parse(response)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return False
    return True


def contains_control_characters(text: str) -> bool:
    """Reject control characters that are not ordinary whitespace.

    Stray control bytes survive JSON and survive the tokenizer, then produce
    training text that is neither readable nor reproducible.
    """
    for ch in text:
        code = ord(ch)
        if code < 32 and ch not in "\t\n\r":
            return True
        if code == 0x7F:
            return True
    return False


def load_instruction_jsonl(path):
    """Parse a JSONL instruction file, reporting malformed lines by number."""
    records, malformed = [], []
    text = Path(path).read_text(encoding="utf-8-sig")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            malformed.append({"line": number, "reason": str(exc)})
    return records, malformed


def summarise(records) -> dict:
    """Dataset statistics. Counts only; nothing here is a quality judgement."""
    from collections import Counter

    categories = Counter(r.get("category") for r in records)
    tasks = Counter(r.get("task") for r in records)
    languages = Counter(r.get("language") for r in records if r.get("language"))
    splits = Counter(r.get("split") for r in records)
    licences = Counter(r.get("license") for r in records if r.get("license"))
    sources = Counter(r.get("source") for r in records if r.get("source"))
    return {
        "total": len(records),
        "by_category": dict(categories),
        "by_task": dict(tasks),
        "by_language": dict(languages),
        "by_split": dict(splits),
        "by_license": dict(licences),
        "by_source": dict(sources),
        "distinct_sources": len(sources),
    }