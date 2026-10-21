"""Author the v1.5.0 surface-aligned evaluation suite.

The v1.4.0 twenty-case suite is left untouched; this is a *separate* suite
whose cases declare a task surface, so a base completion model is not scored
on instruction prompts it was never trained for.

Surfaces
--------
``base_completion``     a code prefix; the assembled program is judged
``instruction_to_code`` a natural-language instruction; needs an
                        instruction-qualified checkpoint
``chat_instruction``    the same, through a chat contract; needs a
                        chat-qualified checkpoint and the matching contract
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.python_eval import SCHEMA_VERSION

PROVENANCE = {
    "source": "authored:v1.5.0",
    "author": "taksh",
    "license": "project-internal",
}

BASE_CASES = [
    ("v15-base-001", "def add(a, b):\n"),
    ("v15-base-002", "def is_even(value):\n"),
    ("v15-base-003", "def total(values):\n"),
    ("v15-base-004", "def longest(words):\n"),
    ("v15-base-005", "class Counter:\n"),
    ("v15-base-006", "import os\n\n\ndef sep():\n"),
    ("v15-base-007", "def clamp(value, low, high):\n"),
    ("v15-base-008", "for index in range(10):\n"),
]

INSTRUCTION_CASES = [
    ("v15-instr-001", "Write a Python function that returns the maximum value in a list.",
     ["def "]),
    ("v15-instr-002", "Write a Python function that reverses a string.",
     ["def ", "return"]),
    ("v15-instr-003", "Write a Python function that counts the vowels in a word.",
     ["def ", "for"]),
    ("v15-instr-004", "Write a Python function that returns the n-th Fibonacci number.",
     ["def ", "return"]),
    ("v15-instr-005", "Write a Python function that checks whether a number is prime.",
     ["def ", "if"]),
]

CHAT_CASES = [
    ("v15-chat-001", "Write a Python function that sums the even numbers in a list.",
     ["def "]),
    ("v15-chat-002", "Write a Python function that returns the keys of a dictionary sorted by value.",
     ["def ", "return"]),
    ("v15-chat-003", "Write a Python function that parses a comma separated line into a list.",
     ["def ", "split"]),
]


def _case(case_id, category, prompt, surface, validator, budget, environment):
    return {
        "case_id": case_id,
        "category": category,
        "language": "python",
        "surface": surface,
        "prompt": prompt,
        "prompt_format": "code_completion" if surface == "base_completion" else "chat_instruction",
        "prompt_contract": "qm.chat_instruction/2" if surface == "chat_instruction" else None,
        "expected_contract": f"A {surface} answer satisfying the case validator.",
        "validator": validator,
        "budget": budget,
        "provenance": dict(PROVENANCE),
        "contamination": "authored",
        "execution_environment": environment,
    }


def build_cases() -> list:
    cases = []
    for case_id, prompt in BASE_CASES:
        cases.append(
            _case(case_id, "function_completion", prompt, "base_completion",
                  {"kind": "syntax_only"},
                  {"timeout_seconds": 10, "max_output_bytes": 65536, "max_new_tokens": 64},
                  "none")
        )
    for case_id, prompt, contains in INSTRUCTION_CASES:
        cases.append(
            _case(case_id, "small_algorithm", prompt, "instruction_to_code",
                  {"kind": "contains_all", "contains": contains},
                  {"timeout_seconds": 10, "max_output_bytes": 65536, "max_new_tokens": 96},
                  "none")
        )
    for case_id, prompt, contains in CHAT_CASES:
        cases.append(
            _case(case_id, "control_flow", prompt, "chat_instruction",
                  {"kind": "contains_all", "contains": contains},
                  {"timeout_seconds": 10, "max_output_bytes": 65536, "max_new_tokens": 96},
                  "none")
        )
    return cases


def main() -> int:
    path = Path(__file__).resolve().parent / "data" / "python_capability_v15.json"
    payload = {
        "schema": SCHEMA_VERSION,
        "suite": {
            "id": "python_capability_v15",
            "version": "v1",
            "description": (
                "Surface-aligned Python coding cases authored for v1.5.0. The "
                "v1.4.0 twenty-case suite is unchanged and remains the baseline."
            ),
            "provenance": PROVENANCE,
        },
        "cases": build_cases(),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} with {len(payload['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
