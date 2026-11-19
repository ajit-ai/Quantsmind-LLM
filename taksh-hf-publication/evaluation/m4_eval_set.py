"""The M4 evaluation set: fixed, small, and chosen before any tuning.

These prompts were authored for this milestone and are **not** derived from the
instruction training data, so a model cannot score well by having memorised its
own training set. They were written before the M3 baseline or the tuned model
produced a single token, and they are not edited afterwards. The file records
``frozen: true`` and a content hash, which tests assert, so "we did not tune on
the eval set" is checkable rather than promised.

Small by design: this is not a benchmark. It is a fixed instrument for one
question -- did instruction tuning change behaviour on held-out instructions?
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SET_ID = "m4-eval-v1"
FROZEN = True

PROMPTS = [
    {
        "id": "code-01-reverse",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called reverse_string that takes one "
                  "string argument and returns it reversed. Output only a single "
                  "fenced Python code block.",
        "test": "reverse_string",
        "expects": "a function reverse_string(s) returning s reversed",
    },
    {
        "id": "code-02-palindrome",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called is_palindrome that returns True "
                  "if the input string reads the same forwards and backwards, "
                  "ignoring case. Output only a single fenced Python code block.",
        "test": "is_palindrome",
        "expects": "a boolean palindrome predicate",
    },
    {
        "id": "code-03-fizzbuzz",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called fizzbuzz_n that returns a list "
                  "of strings for 1..n where multiples of 3 become 'Fizz', "
                  "multiples of 5 become 'Buzz', multiples of both become "
                  "'FizzBuzz', and other numbers become their decimal string. "
                  "Output only a single fenced Python code block.",
        "test": "fizzbuzz_n",
        "expects": "a fizzbuzz list generator",
    },
    {
        "id": "code-04-flatten",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called flatten that takes a list which "
                  "may contain nested lists of arbitrary depth and returns a flat "
                  "list preserving order. Output only a single fenced Python code "
                  "block.",
        "test": "flatten",
        "expects": "a recursive list flattener",
    },
    {
        "id": "code-05-second-largest",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called second_largest that returns the "
                  "second largest distinct value in a list of numbers, or None "
                  "if there is none. Output only a single fenced Python code "
                  "block.",
        "test": "second_largest",
        "expects": "second largest distinct value or None",
    },
    {
        "id": "code-06-explain",
        "category": "technical",
        "task": "code_explanation",
        "prompt": "In one short paragraph, explain what this function does and "
                  "why the closure variable is needed:\n\n```python\n"
                  "def make_counter():\n    count = 0\n"
                  "    def increment():\n        nonlocal count\n"
                  "        count += 1\n        return count\n"
                  "    return increment\n```",
        "test": None,
        "expects": "describes a closure holding mutable state",
    },
    {
        "id": "code-07-explain-api",
        "category": "technical",
        "task": "api_explanation",
        "prompt": "In one short paragraph, explain what the `re` module's "
                  "`finditer` returns and how it differs from `re.match`.",
        "test": None,
        "expects": "describes scanning all matches versus matching at the start",
    },
    {
        "id": "code-08-fix-bug",
        "category": "coding",
        "task": "debugging",
        "prompt": "This function should return the sum of an iterable but returns "
                  "0. Explain the bug in one sentence, then give the corrected "
                  "code:\n\n```python\ndef total(numbers):\n    sum = 0\n"
                  "    for i in range(len(numbers)):\n"
                  "        sum += numbers[len(numbers)]\n    return sum\n```",
        "test": None,
        "expects": "identifies numbers[len(numbers)] as out of range",
    },
    {
        "id": "if-01-exact-word",
        "category": "instruction_following",
        "task": "constrained_output",
        "prompt": "Respond with the single word ACKNOWLEDGED and no other "
                  "text of any kind.",
        "test": None,
        "expects": "ACKNOWLEDGED",
    },
    {
        "id": "if-02-bullets",
        "category": "instruction_following",
        "task": "format_following",
        "prompt": "Produce a bulleted list of exactly three prime numbers "
                  "greater than 10. Use Markdown bullets and add nothing else.",
        "test": None,
        "expects": "exactly three '- ' lines",
    },
    {
        "id": "if-03-numbered",
        "category": "instruction_following",
        "task": "format_following",
        "prompt": "List exactly four items as a numbered list and nothing else.",
        "test": None,
        "expects": "exactly four numbered lines",
    },
    {
        "id": "struct-01-json",
        "category": "structured_output",
        "task": "json_output",
        "prompt": "Emit a JSON object for a cache entry. It must contain "
                  "precisely: key (string), ttl_seconds (integer), warm (boolean). "
                  "Output the JSON alone, without backticks or commentary.",
        "test": None,
        "expects": "a JSON object with exactly those four keys",
    },
    {
        "id": "struct-02-yaml",
        "category": "structured_output",
        "task": "config_output",
        "prompt": "Write a YAML Kubernetes-style deployment named `web` with 3 "
                  "replicas and a CPU limit of \"2\". Output only YAML, no fences.",
        "test": None,
        "expects": "YAML without markdown fences",
    },
    {
        "id": "struct-03-sql",
        "category": "technical",
        "task": "sql",
        "prompt": "Write SQL that lists, for each department, the number of "
                  "employees hired in the last year, highest count first. "
                  "Provide the statement and nothing else.",
        "test": None,
        "expects": "a SELECT with WHERE and ORDER BY",
    },
    {
        "id": "if-04-multistep",
        "category": "instruction_following",
        "task": "multi_step_instruction",
        "prompt": "Do exactly three things, one per line, numbered 1. 2. 3., and "
                  "nothing else: (1) state how many items are in the list "
                  "[1,2,3,4,5]; (2) give their sum; (3) give their product.",
        "test": None,
        "expects": "three numbered lines with 5, 15, 120",
    },
    {
        "id": "code-09-factorial",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called factorial that returns n! for a "
                  "non-negative integer n, with factorial(0) == 1. Output only a "
                  "single fenced Python code block.",
        "test": "factorial",
        "expects": "a correct factorial implementation",
    },
    {
        "id": "code-10-sum",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called sum_list that returns the sum of "
                  "all numbers in a list, and 0 for an empty list. Output only a "
                  "single fenced Python code block.",
        "test": "sum_list",
        "expects": "sum_list(numbers) summing a list",
    },
    {
        "id": "code-11-vowels",
        "category": "coding",
        "task": "code_generation",
        "prompt": "Write a Python function called count_vowels that returns how "
                  "many vowels (a, e, i, o, u) appear in a string, counting case "
                  "insensitively. Output only a single fenced Python code block.",
        "test": "count_vowels",
        "expects": "count_vowels(s) counting aeiou",
    },
]


def prompt_hash() -> str:
    payload = json.dumps(PROMPTS, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_set(path="evaluation/m4_eval_set.jsonl"):
    record = {
        "set_id": SET_ID,
        "frozen": FROZEN,
        "prompt_hash": prompt_hash(),
        "prompt_count": len(PROMPTS),
        "note": (
            "Authored before any M4 training. Not derived from the instruction "
            "training data. Not edited after seeing model output."
        ),
    }
    out = Path(path)
    with out.open("w", encoding="utf-8") as handle:
        for prompt in PROMPTS:
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    out.with_suffix(".meta.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    return record


def load_set(path="evaluation/m4_eval_set.jsonl"):
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main(argv=None) -> int:
    from collections import Counter

    info = write_set()
    print(f"evaluation set: {info['set_id']} ({info['prompt_count']} prompts)")
    print(f"  frozen    : {info['frozen']}")
    print(f"  hash      : {info['prompt_hash']}")
    print(f"  categories: {dict(Counter(p['category'] for p in PROMPTS))}")
    print(f"  testable  : {sum(1 for p in PROMPTS if p['test'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())