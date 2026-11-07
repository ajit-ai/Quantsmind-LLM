"""Taksh v1.8.0 — Build expanded SFT dataset (100+ examples).

Creates instruction→code completion pairs covering diverse Python programming
concepts: arithmetic, logic, strings, lists, dicts, recursion, file I/O,
error handling, classes, algorithms, and more.

Output: data/v180_sft/sft.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

OUT_DIR = Path("data/v180_sft")
OUT_FILE = OUT_DIR / "sft.jsonl"

INSTRUCTION_TEMPLATE = (
    "Complete the following Python function. Return only the code that "
    "replaces the ellipsis or completes the return statement. "
    "Do not repeat the function signature.\n\n{prompt}"
)

# (id, category, prompt, expected_tail, test_args, expected_outputs)
PROBLEMS = [
    # Arithmetic
    ("p01_add", "arithmetic", "def add(a, b):\n    return ", "a + b", [(2, 3), (-5, 3), (0, 0)], [5, -2, 0]),
    ("p02_mul", "arithmetic", "def multiply(x, y):\n    return ", "x * y", [(3, 4), (-2, 5), (0, 100)], [12, -10, 0]),
    ("p03_sub", "arithmetic", "def subtract(a, b):\n    return ", "a - b", [(10, 3), (5, 10), (0, 0)], [7, -5, 0]),
    ("p04_div", "arithmetic", "def divide(a, b):\n    return ", "a / b", [(10, 2), (7, 3), (0, 5)], [5.0, 2.3333333333333335, 0.0]),
    ("p05_mod", "arithmetic", "def mod(a, b):\n    return ", "a % b", [(10, 3), (7, 2), (-10, 3)], [1, 1, 2]),
    ("p06_pow", "arithmetic", "def power(a, b):\n    return ", "a ** b", [(2, 3), (5, 0), (4, 0.5)], [8, 1, 2.0]),
    ("p07_abs", "arithmetic", "def absolute(x):\n    if x < 0:\n        return \n    else:\n        return x", "-x", [(5), (-5), (0)], [5, 5, 0]),
    ("p08_max", "arithmetic", "def maximum(a, b):\n    if a > b:\n        return a\n    else:\n        return ", "b", [(3, 7), (10, 2), (5, 5)], [7, 10, 5]),
    ("p09_min", "arithmetic", "def minimum(a, b):\n    if a < b:\n        return a\n    else:\n        return ", "b", [(3, 7), (10, 2), (5, 5)], [3, 2, 5]),
    ("p10_avg", "arithmetic", "def average(nums):\n    return ", "sum(nums) / len(nums)", [([1, 2, 3]), ([10]), ([])], [2.0, 10.0, None]),

    # Logic
    ("p11_is_even", "logic", "def is_even(n):\n    return ", "n % 2 == 0", [(4), (5), (0)], [True, False, True]),
    ("p12_is_odd", "logic", "def is_odd(n):\n    return ", "n % 2 != 0", [(4), (5), (0)], [False, True, False]),
    ("p13_is_pal", "logic", "def is_palindrome(s):\n    return ", "s == s[::-1]", [("aba"), ("abc"), ("")], [True, False, True]),
    ("p14_is_prime", "logic", "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return ", "False\n    return True", [(2), (4), (7)], [True, False, True]),
    ("p15_gt", "logic", "def greater_than(a, b):\n    return ", "a > b", [(5, 3), (3, 5), (5, 5)], [True, False, False]),
    ("p16_lt", "logic", "def less_than(a, b):\n    return ", "a < b", [(5, 3), (3, 5), (5, 5)], [False, True, False]),
    ("p17_and", "logic", "def both_true(a, b):\n    return ", "a and b", [(True, True), (True, False), (False, True)], [True, False, False]),
    ("p18_or", "logic", "def either_true(a, b):\n    return ", "a or b", [(True, True), (True, False), (False, False)], [True, True, False]),
    ("p19_not", "logic", "def negate(x):\n    return ", "not x", [(True), (False), (0)], [False, True, True]),
    ("p20_is_positive", "logic", "def is_positive(x):\n    return ", "x > 0", [(5), (-3), (0)], [True, False, False]),

    # Strings
    ("p21_reverse", "string", "def reverse_string(s):\n    return ", "s[::-1]", [("abc"), (""), ("aba")], ["cba", "", "aba"]),
    ("p22_len", "string", "def string_length(s):\n    return ", "len(s)", [("hello"), (""), ("a b")], [5, 0, 3]),
    ("p23_caps", "string", "def capitalize_words(s):\n    return ", "s.title()", [("hello world"), (""), ("PYTHON")], ["Hello World", "", "Python"]),
    ("p24_lower", "string", "def to_lower(s):\n    return ", "s.lower()", [("HELLO"), ("Hello"), ("")], ["hello", "hello", ""]),
    ("p25_upper", "string", "def to_upper(s):\n    return ", "s.upper()", [("hello"), ("Hello"), ("")], ["HELLO", "HELLO", ""]),
    ("p26_count_vowels", "string", "def count_vowels(s):\n    count = 0\n    for ch in s:\n        if ch in 'aeiou':\n            count += 1\n    return ", "count", [("hello"), ("aeiou"), ("xyz")], [2, 5, 0]),
    ("p27_strip", "string", "def strip_spaces(s):\n    return ", "s.strip()", [("  hello  "), ("hello"), ("   ")], ["hello", "hello", ""]),
    ("p28_replace", "string", "def replace_char(s, old, new):\n    return ", "s.replace(old, new)", [("hello", "l", "x"), ("", "a", "b")], ["hexxo", ""]),
    ("p29_contains", "string", "def contains(s, sub):\n    return ", "sub in s", [("hello", "ell"), ("hello", "xyz")], [True, False]),
    ("p30_join", "string", "def join_strings(words, sep):\n    return ", "sep.join(words)", [(["a", "b", "c"], "-"), ([], ",")], ["a-b-c", ""]),

    # Lists
    ("p31_sum_list", "list", "def sum_list(lst):\n    total = 0\n    for x in lst:\n        total += x\n    return ", "total", [([1, 2, 3]), ([5]), ([])], [6, 5, 0]),
    ("p32_filter_pos", "list", "def filter_positive(nums):\n    result = []\n    for n in nums:\n        if n > 0:\n            result.", "append(n)", [([-1, 2, -3, 4]), ([-1, -2]), ([1, 2, 3])], [[2, 4], [], [1, 2, 3]]),
    ("p33_map_double", "list", "def double_all(nums):\n    result = []\n    for n in nums:\n        result.", "append(n * 2)", [([1, 2, 3]), ([])], [[2, 4, 6], []]),
    ("p34_len_list", "list", "def list_length(lst):\n    return ", "len(lst)", [([1, 2, 3]), ([])], [3, 0]),
    ("p35_append", "list", "def append_item(lst, item):\n    lst.", "append(item)\n    return lst", [([1, 2], 3), ([], "a")], [[1, 2, 3], ["a"]]),
    ("p36_reverse_list", "list", "def reverse_list(lst):\n    return ", "lst[::-1]", [([1, 2, 3]), ([])], [[3, 2, 1], []]),
    ("p37_sort_list", "list", "def sort_list(lst):\n    return ", "sorted(lst)", [([3, 1, 2]), ([])], [[1, 2, 3], []]),
    ("p38_flatten", "list", "def flatten_list(nested):\n    flat = []\n    for sub in nested:\n        for item in sub:\n            flat.", "append(item)", [([[1, 2], [3, 4]]), ([])], [[1, 2, 3, 4], []]),
    ("p39_dedup", "list", "def remove_duplicates(items):\n    return ", "list(set(items))", [([1, 2, 2, 3]), ([])], [[1, 2, 3], []]),
    ("p40_index_of", "list", "def index_of(lst, item):\n    for i, x in enumerate(lst):\n        if x == item:\n            return ", "i\n    return -1", [([1, 2, 3], 2), ([1, 2, 3], 5)], [1, -1]),

    # Dicts
    ("p41_merge", "dict", "def merge_dicts(d1, d2):\n    result = {}\n    for k, v in d1.items():\n        result[k] = v\n    for k, v in d2.items():\n        result[k] = v\n    return ", "result", [({"a": 1}, {"b": 2}), ({"a": 1}, {"a": 2}), ({}, {})], [{"a": 1, "b": 2}, {"a": 2}, {}]),
    ("p42_get", "dict", "def get_value(d, key):\n    return ", "d.get(key)", [({"a": 1}, "a"), ({"a": 1}, "b")], [1, None]),
    ("p43_keys", "dict", "def dict_keys(d):\n    return ", "list(d.keys())", [({"a": 1, "b": 2}), ({})], [["a", "b"], []]),
    ("p44_values", "dict", "def dict_values(d):\n    return ", "list(d.values())", [({"a": 1, "b": 2}), ({})], [[1, 2], []]),
    ("p45_contains_key", "dict", "def has_key(d, key):\n    return ", "key in d", [({"a": 1}, "a"), ({"a": 1}, "b")], [True, False]),

    # Recursion
    ("p46_factorial", "recursion", "def factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return ", "n * factorial(n-1)", [(0), (1), (5)], [1, 1, 120]),
    ("p47_fib", "recursion", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return ", "fibonacci(n-1) + fibonacci(n-2)", [(0), (1), (6)], [0, 1, 8]),
    ("p48_sum_rec", "recursion", "def sum_recursive(lst):\n    if not lst:\n        return 0\n    return ", "lst[0] + sum_recursive(lst[1:])", [([1, 2, 3]), ([])], [6, 0]),

    # File I/O
    ("p49_read_file", "file_io", "def read_file(path):\n    with open(path, 'r') as f:\n        return ", "f.read()", [("nonexistent.txt"), ], [None]),
    ("p50_write_file", "file_io", "def write_file(path, content):\n    with open(path, 'w') as f:\n        f.", "write(content)\n    return True", [("test.txt", "hello")], [True]),

    # Classes
    ("p51_class_init", "class", "class Point:\n    def __init__(self, x, y):\n        self.", "x = x\n        self.y = y", [()], [None]),
    ("p52_class_method", "class", "class Calculator:\n    def add(self, a, b):\n        return ", "a + b", [(2, 3)], [5]),

    # Algorithms
    ("p53_bubble_sort", "algorithm", "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = ", "arr[j+1], arr[j]\n    return arr", [([3, 1, 2]), ([])], [[1, 2, 3], []]),
    ("p54_binary_search", "algorithm", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return ", "mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1", [([1, 2, 3, 4, 5], 3), ([1, 2, 3, 4, 5], 6)], [2, -1]),
    ("p55_count_occurrences", "algorithm", "def count_occurrences(lst, item):\n    count = 0\n    for x in lst:\n        if x == item:\n            count += 1\n    return ", "count", [([1, 2, 2, 3, 2], 2), ([], 1)], [3, 0]),

    # Error handling
    ("p56_safe_divide", "error", "def safe_divide(a, b):\n    try:\n        return ", "a / b\n    except ZeroDivisionError:\n        return None", [(10, 2), (10, 0)], [5.0, None]),
    ("p57_safe_int", "error", "def safe_int(s):\n    try:\n        return ", "int(s)\n    except ValueError:\n        return None", [("123"), ("abc")], [123, None]),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for pid, category, prompt, expected_tail, test_args, expected_outputs in PROBLEMS:
        instruction = INSTRUCTION_TEMPLATE.format(prompt=prompt)
        records.append({
            "id": pid,
            "task": category,
            "instruction": instruction,
            "input": "",
            "response": expected_tail.strip(),
            "test_args": test_args,
            "expected_outputs": expected_outputs,
        })

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} SFT examples to {OUT_FILE}")
    categories = sorted({r["task"] for r in records})
    print(f"Categories: {categories}")
    for cat in categories:
        count = sum(1 for r in records if r["task"] == cat)
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
