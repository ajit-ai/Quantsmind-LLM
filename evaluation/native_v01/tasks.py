"""Benchmark task inventory for TAKSH Native v0.1.

Tasks are deterministic specifications only. Expected behavior is expressed as
acceptance criteria and executable checks, not model-generated answers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    category: str
    difficulty: str
    prompt: str
    expected_behavior: str
    acceptance_criteria: tuple[str, ...]
    kind: str = "function"  # function | completion | instruction | debug
    entry_point: str | None = None
    tests: tuple[dict[str, Any], ...] = ()
    completion_prefix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["acceptance_criteria"] = list(self.acceptance_criteria)
        data["tests"] = list(self.tests)
        return data


def _case(name: str, args: list[Any], expected: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "expected": expected}


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        task_id="NV01-BASIC-001",
        category="basic_function_generation",
        difficulty="easy",
        prompt="Write a Python function `add(a, b)` that returns the sum of `a` and `b`.",
        expected_behavior="Defines add(a,b) and returns a+b for numeric inputs.",
        acceptance_criteria=("defines function add(a,b)", "returns correct sum"),
        entry_point="add",
        tests=(
            _case("ints", [2, 3], 5),
            _case("floats", [1.5, 2.5], 4.0),
            _case("negative", [-1, 4], 3),
        ),
    ),
    TaskSpec(
        task_id="NV01-BASIC-002",
        category="basic_function_generation",
        difficulty="easy",
        prompt="Write a Python function `is_even(n)` that returns True if `n` is even, otherwise False.",
        expected_behavior="Defines is_even(n) and returns a boolean parity result.",
        acceptance_criteria=("defines function is_even(n)", "returns True for even", "returns False for odd"),
        entry_point="is_even",
        tests=(
            _case("even", [4], True),
            _case("odd", [7], False),
            _case("zero", [0], True),
        ),
    ),
    TaskSpec(
        task_id="NV01-ARITH-003",
        category="arithmetic_string_manipulation",
        difficulty="easy",
        prompt="Write a Python function `square(n)` that returns `n` multiplied by itself.",
        expected_behavior="Defines square(n) and returns n*n.",
        acceptance_criteria=("defines function square(n)", "returns numeric square"),
        entry_point="square",
        tests=(
            _case("positive", [5], 25),
            _case("negative", [-3], 9),
            _case("zero", [0], 0),
        ),
    ),
    TaskSpec(
        task_id="NV01-STRING-004",
        category="arithmetic_string_manipulation",
        difficulty="easy",
        prompt="Write a Python function `reverse_string(s)` that returns the string `s` reversed.",
        expected_behavior="Defines reverse_string(s) and returns characters in reverse order.",
        acceptance_criteria=("defines function reverse_string(s)", "reverses non-empty string", "handles empty string"),
        entry_point="reverse_string",
        tests=(
            _case("word", ["taksh"], "hskat"),
            _case("space", ["a b"], "b a"),
            _case("empty", [""], ""),
        ),
    ),
    TaskSpec(
        task_id="NV01-STRING-005",
        category="arithmetic_string_manipulation",
        difficulty="medium",
        prompt="Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u) in string `s`, case-insensitively.",
        expected_behavior="Counts vowels in a string ignoring case.",
        acceptance_criteria=("defines function count_vowels(s)", "counts both cases", "returns integer count"),
        entry_point="count_vowels",
        tests=(
            _case("mixed", ["Education"], 5),
            _case("none", ["rhythm"], 0),
            _case("upper", ["AEIOU"], 5),
        ),
    ),
    TaskSpec(
        task_id="NV01-LIST-006",
        category="lists_dicts_iteration",
        difficulty="easy",
        prompt="Write a Python function `sum_list(nums)` that returns the sum of all numbers in the list `nums`.",
        expected_behavior="Iterates or aggregates a list of numbers and returns the total.",
        acceptance_criteria=("defines function sum_list(nums)", "sums values", "handles empty list"),
        entry_point="sum_list",
        tests=(
            _case("basic", [[1, 2, 3, 4]], 10),
            _case("empty", [[]], 0),
            _case("mixed", [[-1, 5, 2]], 6),
        ),
    ),
    TaskSpec(
        task_id="NV01-DICT-007",
        category="lists_dicts_iteration",
        difficulty="medium",
        prompt="Write a Python function `get_value(d, key, default)` that returns `d[key]` if present, otherwise returns `default`.",
        expected_behavior="Safely retrieves a dictionary value with a fallback default.",
        acceptance_criteria=("defines function get_value(d,key,default)", "returns existing value", "returns default when missing"),
        entry_point="get_value",
        tests=(
            _case("present", [{"a": 1}, "a", 0], 1),
            _case("missing", [{"a": 1}, "b", 7], 7),
            _case("none_value", [{"a": None}, "a", 7], None),
        ),
    ),
    TaskSpec(
        task_id="NV01-ITER-008",
        category="lists_dicts_iteration",
        difficulty="medium",
        prompt="Write a Python function `filter_positive(nums)` that returns a new list containing only the positive numbers from `nums` in original order.",
        expected_behavior="Filters positive numbers (> 0) preserving order.",
        acceptance_criteria=("defines function filter_positive(nums)", "keeps positives only", "preserves order"),
        entry_point="filter_positive",
        tests=(
            _case("mixed", [[-1, 0, 2, 5, -3]], [2, 5]),
            _case("all_negative", [[-2, -1]], []),
            _case("empty", [[]], []),
        ),
    ),
    TaskSpec(
        task_id="NV01-COND-009",
        category="conditional_logic_edge_cases",
        difficulty="easy",
        prompt="Write a Python function `max_of_two(a, b)` that returns the larger of `a` and `b`.",
        expected_behavior="Compares two values and returns the maximum.",
        acceptance_criteria=("defines function max_of_two(a,b)", "returns larger value", "handles equality"),
        entry_point="max_of_two",
        tests=(
            _case("first", [9, 3], 9),
            _case("second", [2, 8], 8),
            _case("equal", [4, 4], 4),
        ),
    ),
    TaskSpec(
        task_id="NV01-EDGE-010",
        category="conditional_logic_edge_cases",
        difficulty="medium",
        prompt="Write a Python function `safe_divide(a, b)` that returns `a / b`, or returns None if `b` is 0.",
        expected_behavior="Performs division and guards division-by-zero.",
        acceptance_criteria=("defines function safe_divide(a,b)", "divides normally", "returns None on zero divisor"),
        entry_point="safe_divide",
        tests=(
            _case("normal", [8, 2], 4.0),
            _case("zero", [5, 0], None),
            _case("float", [7, 2], 3.5),
        ),
    ),
    TaskSpec(
        task_id="NV01-EDGE-011",
        category="conditional_logic_edge_cases",
        difficulty="medium",
        prompt="Write a Python function `clamp(x, low, high)` that returns `low` if `x < low`, `high` if `x > high`, otherwise `x`.",
        expected_behavior="Clamps a value into an inclusive range.",
        acceptance_criteria=("defines function clamp(x,low,high)", "clamps below", "clamps above", "passes through in-range"),
        entry_point="clamp",
        tests=(
            _case("below", [-1, 0, 10], 0),
            _case("above", [15, 0, 10], 10),
            _case("inside", [5, 0, 10], 5),
        ),
    ),
    TaskSpec(
        task_id="NV01-ALGO-012",
        category="simple_algorithm",
        difficulty="medium",
        prompt="Write a Python function `factorial(n)` that returns `n!` for a non-negative integer `n` (define 0! as 1).",
        expected_behavior="Computes factorial iteratively or recursively for n >= 0.",
        acceptance_criteria=("defines function factorial(n)", "returns 1 for 0", "returns correct factorial"),
        entry_point="factorial",
        tests=(
            _case("zero", [0], 1),
            _case("one", [1], 1),
            _case("five", [5], 120),
        ),
    ),
    TaskSpec(
        task_id="NV01-ALGO-013",
        category="simple_algorithm",
        difficulty="medium",
        prompt="Write a Python function `fibonacci(n)` that returns the nth Fibonacci number with `fibonacci(0)=0` and `fibonacci(1)=1`.",
        expected_behavior="Returns the nth Fibonacci number for n >= 0.",
        acceptance_criteria=("defines function fibonacci(n)", "base cases correct", "computes sequence value"),
        entry_point="fibonacci",
        tests=(
            _case("zero", [0], 0),
            _case("one", [1], 1),
            _case("seven", [7], 13),
        ),
    ),
    TaskSpec(
        task_id="NV01-ALGO-014",
        category="simple_algorithm",
        difficulty="medium",
        prompt="Write a Python function `is_palindrome(s)` that returns True if string `s` reads the same forwards and backwards, otherwise False.",
        expected_behavior="Checks exact string palindrome equality.",
        acceptance_criteria=("defines function is_palindrome(s)", "detects palindrome", "rejects non-palindrome"),
        entry_point="is_palindrome",
        tests=(
            _case("yes", ["racecar"], True),
            _case("no", ["python"], False),
            _case("empty", [""], True),
        ),
    ),
    TaskSpec(
        task_id="NV01-COMP-015",
        category="function_completion",
        difficulty="easy",
        kind="completion",
        prompt="Complete the following Python function so that it returns the length of the list `items`:\n\n```python\ndef count_items(items):\n```",
        expected_behavior="Completes count_items(items) to return len(items).",
        acceptance_criteria=("defines/completes count_items(items)", "returns length"),
        entry_point="count_items",
        completion_prefix="def count_items(items):\n",
        tests=(
            _case("three", [[1, 2, 3]], 3),
            _case("empty", [[]], 0),
            _case("strings", [["a", "b"]], 2),
        ),
    ),
    TaskSpec(
        task_id="NV01-COMP-016",
        category="function_completion",
        difficulty="medium",
        kind="completion",
        prompt="Complete the following Python function so that it returns a list of squares of numbers in `nums`:\n\n```python\ndef squares(nums):\n```",
        expected_behavior="Completes squares(nums) returning [n*n for n in nums].",
        acceptance_criteria=("defines/completes squares(nums)", "returns squares in order"),
        entry_point="squares",
        completion_prefix="def squares(nums):\n",
        tests=(
            _case("basic", [[1, 2, 3]], [1, 4, 9]),
            _case("empty", [[]], []),
            _case("negative", [[-2, 0]], [4, 0]),
        ),
    ),
    TaskSpec(
        task_id="NV01-INST-017",
        category="instruction_following",
        difficulty="easy",
        kind="instruction",
        prompt="Write only a Python function named `greet(name)` that returns the string `Hello, <name>!`. Do not include explanations.",
        expected_behavior="Follows instruction to produce the named function with exact greeting format.",
        acceptance_criteria=("defines greet(name)", "returns formatted greeting", "minimal instruction-following"),
        entry_point="greet",
        tests=(
            _case("ada", ["Ada"], "Hello, Ada!"),
            _case("sam", ["Sam"], "Hello, Sam!"),
            _case("empty", [""], "Hello, !"),
        ),
    ),
    TaskSpec(
        task_id="NV01-INST-018",
        category="instruction_following",
        difficulty="medium",
        kind="instruction",
        prompt="Write a Python function `unique_sorted(nums)` that returns a sorted list of unique numbers from `nums`.",
        expected_behavior="Removes duplicates and returns values in ascending order.",
        acceptance_criteria=("defines unique_sorted(nums)", "deduplicates", "sorts ascending"),
        entry_point="unique_sorted",
        tests=(
            _case("mixed", [[3, 1, 2, 3, 2]], [1, 2, 3]),
            _case("empty", [[]], []),
            _case("negatives", [[0, -1, -1, 5]], [-1, 0, 5]),
        ),
    ),
    TaskSpec(
        task_id="NV01-DEBUG-019",
        category="code_correction_debugging",
        difficulty="medium",
        kind="debug",
        prompt=(
            "Fix the bug in this Python function so it correctly returns the sum of numbers from 1 to n inclusive.\n\n"
            "```python\n"
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(n):\n"
            "        total += i\n"
            "    return total\n"
            "```"
        ),
        expected_behavior="Corrects off-by-one/range bug to include n.",
        acceptance_criteria=("defines sum_to_n(n)", "includes n", "returns correct sums"),
        entry_point="sum_to_n",
        tests=(
            _case("one", [1], 1),
            _case("five", [5], 15),
            _case("zero", [0], 0),
        ),
    ),
    TaskSpec(
        task_id="NV01-DEBUG-020",
        category="code_correction_debugging",
        difficulty="medium",
        kind="debug",
        prompt=(
            "Fix the bug in this Python function so it returns True when `x` is found in `items`, otherwise False.\n\n"
            "```python\n"
            "def contains(items, x):\n"
            "    for item in items:\n"
            "        if item = x:\n"
            "            return True\n"
            "    return False\n"
            "```"
        ),
        expected_behavior="Fixes assignment/comparison bug and returns correct membership result.",
        acceptance_criteria=("defines contains(items,x)", "uses comparison", "returns correct boolean"),
        entry_point="contains",
        tests=(
            _case("found", [[1, 2, 3], 2], True),
            _case("missing", [[1, 2, 3], 4], False),
            _case("empty", [[], 1], False),
        ),
    ),
)


def get_tasks() -> list[TaskSpec]:
    return list(TASKS)
