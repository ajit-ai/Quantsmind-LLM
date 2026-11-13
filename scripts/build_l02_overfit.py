"""Build the L0.2 Phase-B overfitting dataset and its memorization benchmark.

24 original, unambiguous Python instruction->code examples covering: hello
world, arithmetic, conditionals, loops, strings, lists, factorial, fibonacci.
Responses are complete functions (L0.2 response-format rule).

This dataset is BOTH the training set and the memorization test: a model that
cannot reproduce these 24 examples after training has not learned the pipeline.
It is NOT a generalization benchmark (that is l02_python_bench_v1.json).

Executable verification: expected outputs are computed at build time by
executing the functions in this file (trusted, original code). Training renders
through the standard Taksh v1 chat template, exactly as production SFT does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Each entry: (id, instruction, response, tests[(name, args, expected)])
EXAMPLES = [
    ("hello_world",
     "Write a Python function named hello_world() that returns the string 'Hello, world!'.",
     "def hello_world():\n    return 'Hello, world!'",
     [("basic", [], "Hello, world!")]),
    ("add_two",
     "Write a Python function named add_two(a, b) that returns the sum of the two numbers a and b.",
     "def add_two(a, b):\n    return a + b",
     [("ints", [2, 3], 5), ("negatives", [-1, 1], 0)]),
    ("is_even",
     "Write a Python function named is_even(n) that returns True if n is even and False otherwise.",
     "def is_even(n):\n    return n % 2 == 0",
     [("even", [4], True), ("odd", [7], False)]),
    ("is_odd",
     "Write a Python function named is_odd(n) that returns True if n is odd and False otherwise.",
     "def is_odd(n):\n    return n % 2 != 0",
     [("odd", [3], True), ("even", [10], False)]),
    ("max_of_two",
     "Write a Python function named max_of_two(a, b) that returns the larger of the two numbers a and b.",
     "def max_of_two(a, b):\n    return a if a >= b else b",
     [("a_larger", [9, 4], 9), ("b_larger", [2, 8], 8)]),
    ("reverse_string",
     "Write a Python function named reverse_string(text) that returns the reversed string.",
     "def reverse_string(text):\n    return text[::-1]",
     [("basic", ["hello"], "olleh")]),
    ("sum_list",
     "Write a Python function named sum_list(numbers) that returns the sum of the list of numbers. Return 0 for an empty list.",
     "def sum_list(numbers):\n    total = 0\n    for value in numbers:\n        total += value\n    return total",
     [("basic", [[1, 2, 3]], 6), ("empty", [[]], 0)]),
    ("factorial",
     "Write a Python function named factorial(n) that returns n! (n factorial) using a loop. factorial(0) is 1.",
     "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
     [("zero", [0], 1), ("five", [5], 120)]),
    ("fibonacci",
     "Write a Python function named fibonacci(n) that returns the n-th Fibonacci number, with fibonacci(0) equal to 0 and fibonacci(1) equal to 1, using a loop.",
     "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     [("zero", [0], 0), ("seven", [7], 13)]),
    ("square_number",
     "Write a Python function named square_number(n) that returns n squared.",
     "def square_number(n):\n    return n * n",
     [("basic", [7], 49)]),
    ("cube_number",
     "Write a Python function named cube_number(n) that returns n cubed.",
     "def cube_number(n):\n    return n * n * n",
     [("basic", [3], 27)]),
    ("absolute_value",
     "Write a Python function named absolute_value(n) that returns the absolute value of n.",
     "def absolute_value(n):\n    return abs(n)",
     [("negative", [-5], 5), ("positive", [5], 5)]),
    ("largest_of_three",
     "Write a Python function named largest_of_three(a, b, c) that returns the largest of the three numbers a, b and c.",
     "def largest_of_three(a, b, c):\n    return max(a, b, c)",
     [("first", [7, 2, 3], 7), ("third", [1, 5, 9], 9)]),
    ("countdown_from",
     "Write a Python function named countdown_from(n) that returns a list with the integers from n down to 1.",
     "def countdown_from(n):\n    return list(range(n, 0, -1))",
     [("five", [5], [5, 4, 3, 2, 1]), ("one", [1], [1])]),
    ("is_prime",
     "Write a Python function named is_prime(n) that returns True if n is a prime number, otherwise False. n less than 2 is not prime.",
     "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
     [("prime", [7], True), ("composite", [9], False), ("tiny", [1], False)]),
    ("swap_values",
     "Write a Python function named swap_values(a, b) that returns a list with b first and a second.",
     "def swap_values(a, b):\n    return [b, a]",
     [("basic", [1, 'x'], ['x', 1])]),
    ("double_list",
     "Write a Python function named double_list(numbers) that returns a list with every number doubled.",
     "def double_list(numbers):\n    return [value * 2 for value in numbers]",
     [("basic", [[1, 2, 3]], [2, 4, 6])]),
    ("average_of_two",
     "Write a Python function named average_of_two(a, b) that returns the average of the two numbers a and b.",
     "def average_of_two(a, b):\n    return (a + b) / 2",
     [("basic", [4, 8], 6.0)]),
    ("char_count",
     "Write a Python function named char_count(text, ch) that returns how many times the single character ch occurs in text.",
     "def char_count(text, ch):\n    return text.count(ch)",
     [("basic", ['banana', 'a'], 3)]),
    ("repeat_text",
     "Write a Python function named repeat_text(text, times) that returns text repeated times times with no separator.",
     "def repeat_text(text, times):\n    return text * times",
     [("basic", ['ab', 3], 'ababab'), ("zero", ['ab', 0], '')]),
    ("greet",
     "Write a Python function named greet(name) that returns 'Hello, ' followed by name and a '!'.",
     "def greet(name):\n    return 'Hello, ' + name + '!'",
     [("basic", ['Taksh'], 'Hello, Taksh!')]),
    ("increment_by_two",
     "Write a Python function named increment_by_two(n) that returns n + 2.",
     "def increment_by_two(n):\n    return n + 2",
     [("basic", [5], 7)]),
    ("first_char",
     "Write a Python function named first_char(text) that returns the first character of text, or an empty string if text is empty.",
     "def first_char(text):\n    if not text:\n        return ''\n    return text[0]",
     [("basic", ['abc'], 'a'), ("empty", [''], '')]),
    ("square_of_sum",
     "Write a Python function named square_of_sum(a, b) that returns the square of (a + b).",
     "def square_of_sum(a, b):\n    return (a + b) * (a + b)",
     [("basic", [1, 2], 9)]),
]


def main():
    out_sft = Path("data/l02_overfit/sft.jsonl")
    out_sft.parent.mkdir(parents=True, exist_ok=True)
    with out_sft.open("w", encoding="utf-8") as fh:
        for i, (task, ins, resp, _) in enumerate(EXAMPLES):
            fh.write(json.dumps({"id": f"overfit-{i:02d}", "task": task,
                                 "instruction": ins, "input": "", "response": resp},
                                ensure_ascii=False) + "\n")

    problems = []
    for i, (task, ins, resp, tests) in enumerate(EXAMPLES):
        problems.append({
            "id": f"OF{i:02d}", "category": "OF", "name": task,
            "instruction": ins, "entry_point": resp.split("(")[0].split()[-1],
            "kind": "function",
            "tests": [{"name": t[0], "args": t[1], "expected": t[2]} for t in tests],
        })
    bench = {
        "schema": "taksh.l02_overfit_bench/1",
        "name": "Taksh L0.2 Phase-B overfitting benchmark",
        "version": "1.0.0",
        "provenance": "Original examples authored for Taksh L0.2 Phase B (2026-11-10). Train and evaluation share the same 24 examples ON PURPOSE: this is a memorization gate, not a generalization measurement.",
        "problems": problems,
    }
    out_bench = Path("evaluation/l02_overfit_bench_v1.json")
    out_bench.write_text(json.dumps(bench, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(EXAMPLES)} pairs to {out_sft} and {len(problems)} problems to {out_bench}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
