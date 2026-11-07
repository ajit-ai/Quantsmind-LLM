"""Author the v1.4.0 Python capability case file.

The cases are written here so the suite is reproducible and reviewable: the
file is generated from one table, every case is tagged as ``authored``
(not taken from a published suite), and the generator is deterministic.

Usage:
    python -m evaluation.python_eval.author_cases
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.python_eval import SCHEMA_VERSION

PROVENANCE = {
    "source": "authored:v1.4.0",
    "author": "taksh",
    "license": "project-internal",
}

SUITE = {
    "id": "python_capability",
    "version": "v1",
    "description": (
        "Authored in-repository Python coding cases. Not drawn from any "
        "published benchmark, so no published-suite licence or contamination "
        "applies."
    ),
    "provenance": (
        "authored for Taksh v1.4.0 in this repository; no external suite and "
        "no training-corpus overlap"
    ),
}

# (id, category, prompt, prompt_format, contains, environment, max_new_tokens)
ROWS = [
    ("py-syntax-001", "syntax_validity",
     "Write a Python function named `is_even` that returns True when a number is even.",
     "chat_instruction", [], "none", 96),
    ("py-syntax-002", "syntax_validity",
     "Write a Python class named `Counter` with an `increment` method.",
     "chat_instruction", [], "none", 96),
    ("py-syntax-003", "syntax_validity",
     "def scale(value):\n    return value * 2\n",
     "code_completion", [], "none", 96),
    ("py-complete-001", "function_completion",
     "def average(values):\n    total = 0\n",
     "code_completion", ["return"], "python3-isolated", 96),
    ("py-complete-002", "function_completion",
     "def is_palindrome(text):\n    cleaned = ''\n",
     "code_completion", ["return"], "python3-isolated", 96),
    ("py-complete-003", "function_completion",
     "def clamp(value, low, high):\n    \"\"\"Clamp value into the range [low, high].\"\"\"\n",
     "code_completion", ["if", "return"], "python3-isolated", 96),
    ("py-complete-004", "function_completion",
     "def count_vowels(text):\n    count = 0\n",
     "code_completion", ["return"], "python3-isolated", 96),
    ("py-algo-001", "small_algorithm",
     "Write a Python function `fibonacci(n)` that returns the n-th Fibonacci number using a loop.",
     "chat_instruction", ["def fibonacci", "for"], "python3-isolated", 128),
    ("py-algo-002", "small_algorithm",
     "Write a Python function `factorial(n)` that returns n! for a non-negative n.",
     "chat_instruction", ["def factorial", "return"], "python3-isolated", 128),
    ("py-algo-003", "small_algorithm",
     "Write a Python function `unique(values)` that returns the unique values of a list, preserving order.",
     "chat_instruction", ["def unique"], "python3-isolated", 128),
    ("py-str-001", "string_and_list",
     "Write a Python function `join_names(names)` that joins a list of names with a comma and a space.",
     "chat_instruction", ["def join_names", "return"], "python3-isolated", 128),
    ("py-str-002", "string_and_list",
     "Write a Python function `reverse_words(sentence)` that reverses the order of the words in a sentence.",
     "chat_instruction", ["def reverse_words"], "python3-isolated", 128),
    ("py-flow-001", "control_flow",
     "Write a Python function `classify(n)` that returns the string `positive`, `negative` or `zero`.",
     "chat_instruction", ["def classify", "if"], "python3-isolated", 128),
    ("py-flow-002", "control_flow",
     "Write a Python function `count_evens(numbers)` that counts how many numbers in the list are even.",
     "chat_instruction", ["def count_evens", "for"], "python3-isolated", 128),
    ("py-debug-001", "debugging",
     "This function returns the wrong result:\n\n"
     "def mean(values):\n    return sum(values) / len(values) + 1\n\n"
     "Rewrite it so it returns the mean.",
     "chat_instruction", ["def mean", "return"], "python3-isolated", 128),
    ("py-debug-002", "debugging",
     "Fix this loop so it counts down from ten to one:\n\n"
     "for i in range(10):\n    print(i)\n",
     "chat_instruction", ["for"], "python3-isolated", 128),
    ("py-test-001", "test_generation",
     "Write a pytest test function named `test_add` for a function `add(a, b)` that returns a + b.",
     "chat_instruction", ["def test_add", "assert"], "python3-isolated", 128),
    ("py-test-002", "test_generation",
     "Write a pytest test function named `test_len_of_empty` that checks that `len([]) == 0`.",
     "chat_instruction", ["def test_len_of_empty", "assert"], "python3-isolated", 128),
    ("py-fmt-001", "format_compliance",
     "Return the code for a function `greet(name)` inside a single fenced code block.",
     "chat_instruction", ["def greet"], "none", 128),
    ("py-fmt-002", "format_compliance",
     "Output only Python code, no prose, for a function `double(x)` that returns x * 2.",
     "chat_instruction", ["def double"], "none", 128),
]


def build_cases() -> list:
    cases = []
    for case_id, category, prompt, prompt_format, contains, environment, tokens in ROWS:
        validator = (
            {"kind": "syntax_only"}
            if not contains
            else {"kind": "contains_all", "contains": list(contains)}
        )
        cases.append(
            {
                "case_id": case_id,
                "category": category,
                "language": "python",
                "prompt": prompt,
                "prompt_format": prompt_format,
                "expected_contract": (
                    f"An answer to the {category} task that satisfies the case "
                    "contract."
                ),
                "validator": validator,
                "budget": {
                    "timeout_seconds": 10,
                    "max_output_bytes": 65536,
                    "max_new_tokens": tokens,
                },
                "provenance": dict(PROVENANCE),
                "contamination": "authored",
                "execution_environment": environment,
            }
        )
    return cases


def main() -> int:
    path = Path(__file__).resolve().parent / "data" / "python_capability_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA_VERSION,
        "suite": SUITE,
        "cases": build_cases(),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {path} with {len(payload['cases'])} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
