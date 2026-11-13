"""Build the Taksh L0.2 SFT dataset (original, license-cleared, self-authored).

Provenance: every (instruction, response) pair is produced by the templates
below, authored for the Taksh L0.2 milestone (2026-11-09). No external source,
no commercial LLM output, no copied code. Function and parameter names are drawn
from seeded pools; tasks are distinct from the L0.2 evaluation benchmark
(evaluation/l02_python_bench_v1.json) -- verified by
tests/test_l02_pipeline.py::test_l02_sft_has_no_benchmark_leakage.

Response style: a single complete function definition (never an expression-only
fragment). The v1.6/v1.8.0 SFT responses were bare expressions like "a + b",
which taught the model to emit fragments; L0.2 responses model full functions.

Determinism: seeded (seed=20261109). Output: data/l02_sft/sft.jsonl plus
data/l02_sft/PROVENANCE.md.
"""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

SEED = 20261109

LIST_POOLS = [["values", "numbers", "items", "data", "nums", "entries"]]
SINGLE_POOLS = [["text", "s", "word", "message", "payload"]]
INT_POOLS = [["n", "x", "count", "value", "amount", "total"]]
DICT_POOLS = [["mapping", "table", "registry", "record", "index"]]


def pick(rng, pool):
    return rng.choice(pool)


def _dedupe(pairs):
    seen = set()
    out = []
    for p in pairs:
        key = (p["instruction"].strip(), p["response"])
        if key in seen:
            continue
        seen.add(key)
        p["id"] = f"l02-{len(out):04d}"
        out.append(p)
    return out


def fn_style(name):
    templates = [
        "Write a Python function named {n}{sig} that {desc}.",
        "Define a Python function named {n}{sig} that {desc}.",
        "Implement {n}{sig}: {desc}.",
        "Create a Python function named {n}{sig} that {desc}.",
    ]
    return templates


def build_pairs():
    rng = random.Random(SEED)
    pairs = []

    def add(task, instruction, response):
        pid = f"l02-{len(pairs):04d}"
        pairs.append({"id": pid, "task": task, "instruction": instruction,
                      "input": "", "response": response})

    # Each family: (task, generator(rng) -> (instruction, response))
    # Response = complete function source, 2-5 lines, correct by construction
    # (reviewed against the stated spec; implementations kept trivially checkable).

    def fam_sum_evens(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["sum_even_numbers", "even_total", "sum_of_evens", "collect_even_sum"])
        ins = f"Write a Python function named {fn}({arg}) that returns the sum of the even integers in the list {arg}. Return 0 if there are none."
        body = (f"def {fn}({arg}):\n"
                f"    total = 0\n"
                f"    for value in {arg}:\n"
                f"        if value % 2 == 0:\n"
                f"            total += value\n"
                f"    return total")
        return ins, body

    def fam_sum_squares(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["sum_of_squares", "squared_total", "square_sum", "sum_squares"])
        ins = f"Write a Python function named {fn}({arg}) that returns the sum of the squares of the integers in the list {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(value * value for value in {arg})")
        return ins, body

    def fam_product(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["multiply_all", "product_of_list", "product_values", "times_together"])
        ins = f"Write a Python function named {fn}({arg}) that returns the product of all integers in the list {arg}. Return 1 for an empty list."
        body = (f"def {fn}({arg}):\n"
                f"    result = 1\n"
                f"    for value in {arg}:\n"
                f"        result *= value\n"
                f"    return result")
        return ins, body

    def fam_max_difference(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["max_difference", "range_of_list", "biggest_gap", "span_values"])
        ins = f"Write a Python function named {fn}({arg}) that returns the difference between the largest and the smallest value in the non-empty list {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return max({arg}) - min({arg})")
        return ins, body

    def fam_clamp(r):
        x = pick(r, ["x", "value", "n", "reading"])
        lo = pick(r, ["lo", "low", "floor"])
        hi = pick(r, ["hi", "high", "cap"])
        fn = pick(r, ["clamp_value", "bound_value", "limit_value", "constrain_value"])
        ins = f"Write a Python function named {fn}({x}, {lo}, {hi}) that returns {x} bounded to the inclusive range [{lo}, {hi}]: {lo} if {x} < {lo}, {hi} if {x} > {hi}, otherwise {x}."
        body = (f"def {fn}({x}, {lo}, {hi}):\n"
                f"    if {x} < {lo}:\n"
                f"        return {lo}\n"
                f"    if {x} > {hi}:\n"
                f"        return {hi}\n"
                f"    return {x}")
        return ins, body

    def fam_mean(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["average_of", "mean_value", "mean_of_list", "list_average"])
        ins = f"Write a Python function named {fn}({arg}) that returns the arithmetic mean of the non-empty list of numbers {arg} as a float."
        body = (f"def {fn}({arg}):\n"
                f"    return sum({arg}) / len({arg})")
        return ins, body

    def fam_count_vowels(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["count_vowels", "vowel_count", "tally_vowels", "number_of_vowels"])
        ins = f"Write a Python function named {fn}({arg}) that returns how many vowels (a, e, i, o, u, case-insensitive) are in the string {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(1 for ch in {arg} if ch.lower() in 'aeiou')")
        return ins, body

    def fam_uppercase_words(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["uppercase_words", "shout_words", "caps_words", "uppercase_each_word"])
        ins = f"Write a Python function named {fn}({arg}) that returns a list of the whitespace-separated words of {arg}, each converted to uppercase."
        body = (f"def {fn}({arg}):\n"
                f"    return [word.upper() for word in {arg}.split()]")
        return ins, body

    def fam_palindrome(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["is_palindrome", "check_palindrome", "reads_same_both_ways", "palindrome_word"])
        ins = f"Write a Python function named {fn}({arg}) that returns True if the string {arg} reads the same forwards and backwards, otherwise False."
        body = (f"def {fn}({arg}):\n"
                f"    return {arg} == {arg}[::-1]")
        return ins, body

    def fam_first_index(r):
        arg = pick(r, SINGLE_POOLS[0])
        target = pick(r, ["target", "needle", "wanted", "ch"])
        fn = pick(r, ["first_index_of", "locate_first", "find_first_index", "index_of_first"])
        ins = f"Write a Python function named {fn}({arg}, {target}) that returns the index of the first occurrence of the single-character string {target} in {arg}, or -1 if it is absent."
        body = (f"def {fn}({arg}, {target}):\n"
                f"    for index, ch in enumerate({arg}):\n"
                f"        if ch == {target}:\n"
                f"            return index\n"
                f"    return -1")
        return ins, body

    def fam_sum_digits(r):
        arg = pick(r, INT_POOLS[0])
        fn = pick(r, ["sum_of_digits", "digit_sum", "total_digits", "add_digits"])
        ins = f"Write a Python function named {fn}({arg}) that takes a non-negative integer and returns the sum of its decimal digits."
        body = (f"def {fn}({arg}):\n"
                f"    total = 0\n"
                f"    while {arg} > 0:\n"
                f"        total += {arg} % 10\n"
                f"        {arg} //= 10\n"
                f"    return total")
        return ins, body

    def fam_double_list(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["double_each", "double_values", "times_two_each", "twice_every"])
        ins = f"Write a Python function named {fn}({arg}) that returns a new list with every number in {arg} multiplied by 2."
        body = (f"def {fn}({arg}):\n"
                f"    return [value * 2 for value in {arg}]")
        return ins, body

    def fam_squares(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["squares_of", "square_each", "squared_values", "square_numbers"])
        ins = f"Write a Python function named {fn}({arg}) that returns a new list with the square of each number in {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return [value * value for value in {arg}]")
        return ins, body

    def fam_count_above(r):
        arg = pick(r, LIST_POOLS[0])
        limit = pick(r, ["limit", "threshold", "bound"])
        fn = pick(r, ["count_above", "above_threshold", "higher_than_count", "tally_above"])
        ins = f"Write a Python function named {fn}({arg}, {limit}) that returns how many numbers in the list {arg} are strictly greater than {limit}."
        body = (f"def {fn}({arg}, {limit}):\n"
                f"    return sum(1 for value in {arg} if value > {limit})")
        return ins, body

    def fam_word_lengths(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["word_lengths", "length_of_each_word", "count_word_lengths", "sizes_of_words"])
        ins = f"Write a Python function named {fn}({arg}) that returns a list with the length of each whitespace-separated word in the string {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return [len(word) for word in {arg}.split()]")
        return ins, body

    def fam_first_upper(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["capitalize_words", "title_each_word", "first_letter_upper", "capitalized_words"])
        ins = f"Write a Python function named {fn}({arg}) that returns the string {arg} with the first letter of each whitespace-separated word capitalized and the rest lowercased."
        body = (f"def {fn}({arg}):\n"
                f"    return ' '.join(word.capitalize() for word in {arg}.split())")
        return ins, body

    def fam_contains_digit(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["has_digit", "contains_a_digit", "includes_digit", "any_digit_present"])
        ins = f"Write a Python function named {fn}({arg}) that returns True if the string {arg} contains at least one digit character, otherwise False."
        body = (f"def {fn}({arg}):\n"
                f"    return any(ch.isdigit() for ch in {arg})")
        return ins, body

    def fam_total_length(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["total_string_length", "sum_string_sizes", "combined_length", "all_string_lengths"])
        ins = f"Write a Python function named {fn}({arg}) that returns the total number of characters across all strings in the list {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(len(item) for item in {arg})")
        return ins, body

    def fam_replace_vowels(r):
        arg = pick(r, SINGLE_POOLS[0])
        fill = pick(r, ["fill", "repl", "mark"])
        fn = pick(r, ["replace_vowels", "swap_vowels", "vowels_to_marks", "mask_vowels"])
        ins = f"Write a Python function named {fn}({arg}, {fill}) that returns the string {arg} with every vowel replaced by the single character {fill}."
        body = (f"def {fn}({arg}, {fill}):\n"
                f"    return ''.join({fill} if ch in 'aeiouAEIOU' else ch for ch in {arg})")
        return ins, body

    def fam_safe_divide(r):
        a = pick(r, ["a", "numerator", "top", "left"])
        b = pick(r, ["b", "denominator", "bottom", "right"])
        fb = pick(r, ["fallback", "default", "instead"])
        fn = pick(r, ["safe_divide", "guarded_divide", "divide_with_fallback", "protected_divide"])
        ins = f"Write a Python function named {fn}({a}, {b}, {fb}) that returns {a} / {b}, or {fb} when {b} is zero."
        body = (f"def {fn}({a}, {b}, {fb}):\n"
                f"    if {b} == 0:\n"
                f"        return {fb}\n"
                f"    return {a} / {b}")
        return ins, body

    def fam_grade_score(r):
        arg = pick(r, INT_POOLS[0])
        fn = pick(r, ["letter_grade", "grade_for_score", "score_to_grade", "mark_band"])
        ins = f"Write a Python function named {fn}({arg}) that takes an integer score 0-100 and returns 'A' for 90 or above, 'B' for 80-89, 'C' for 70-79, 'D' for 60-69, and 'F' otherwise."
        body = (f"def {fn}({arg}):\n"
                f"    if {arg} >= 90:\n"
                f"        return 'A'\n"
                f"    if {arg} >= 80:\n"
                f"        return 'B'\n"
                f"    if {arg} >= 70:\n"
                f"        return 'C'\n"
                f"    if {arg} >= 60:\n"
                f"        return 'D'\n"
                f"    return 'F'")
        return ins, body

    def fam_classify_sign(r):
        arg = pick(r, INT_POOLS[0])
        fn = pick(r, ["classify_sign", "sign_of", "sign_label", "sign_category"])
        ins = f"Write a Python function named {fn}({arg}) that returns 'positive' for a positive integer, 'negative' for a negative integer, and 'zero' for zero."
        body = (f"def {fn}({arg}):\n"
                f"    if {arg} > 0:\n"
                f"        return 'positive'\n"
                f"    if {arg} < 0:\n"
                f"        return 'negative'\n"
                f"    return 'zero'")
        return ins, body

    def fam_factorial(r):
        arg = pick(r, INT_POOLS[0])
        fn = pick(r, ["factorial", "product_factorial", "fact", "factorial_of"])
        ins = f"Write a Python function named {fn}({arg}) that returns the factorial of the non-negative integer {arg} using a loop."
        body = (f"def {fn}({arg}):\n"
                f"    result = 1\n"
                f"    for value in range(2, {arg} + 1):\n"
                f"        result *= value\n"
                f"    return result")
        return ins, body

    def fam_gcd(r):
        a = pick(r, ["a", "left", "first"])
        b = pick(r, ["b", "right", "second"])
        fn = pick(r, ["gcd", "greatest_common_divisor", "common_divisor", "highest_common_factor"])
        ins = f"Write a Python function named {fn}({a}, {b}) that returns the greatest common divisor of the positive integers {a} and {b} using a loop."
        body = (f"def {fn}({a}, {b}):\n"
                f"    while {b}:\n"
                f"        {a}, {b} = {b}, {a} % {b}\n"
                f"    return {a}")
        return ins, body

    def fam_unique_sorted(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["unique_sorted", "sorted_unique_values", "dedup_sorted", "distinct_sorted"])
        ins = f"Write a Python function named {fn}({arg}) that returns a sorted list of the distinct values in the list {arg}."
        body = (f"def {fn}({arg}):\n"
                f"    return sorted(set({arg}))")
        return ins, body

    def fam_invert_dict(r):
        arg = pick(r, DICT_POOLS[0])
        fn = pick(r, ["invert_mapping", "swap_mapping", "reverse_dict", "flip_mapping"])
        ins = f"Write a Python function named {fn}({arg}) that returns a new dict with the keys and values of the dict {arg} swapped. Values in {arg} are unique."
        body = (f"def {fn}({arg}):\n"
                f"    return {{value: key for key, value in {arg}.items()}}")
        return ins, body

    def fam_dict_value_sum(r):
        arg = pick(r, DICT_POOLS[0])
        fn = pick(r, ["sum_mapping_values", "total_mapped_values", "dict_sum_values", "add_all_values"])
        ins = f"Write a Python function named {fn}({arg}) that returns the sum of the integer values in the dict {arg}. Return 0 for an empty dict."
        body = (f"def {fn}({arg}):\n"
                f"    return sum({arg}.values())")
        return ins, body

    def fam_count_uppercase(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["count_uppercase", "tally_caps", "uppercase_count", "number_of_caps"])
        ins = f"Write a Python function named {fn}({arg}) that returns how many characters of the string {arg} are uppercase letters."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(1 for ch in {arg} if ch.isupper())")
        return ins, body

    def fam_read_first_line(r):
        fn = pick(r, ["read_first_line", "first_line_of_file", "peek_first_line", "head_line"])
        ins = f"Write a Python function named {fn}(path) that opens the text file at path and returns its first line with the trailing newline removed. Return an empty string if the file is empty."
        body = (f"def {fn}(path):\n"
                f"    with open(path, encoding='utf-8') as handle:\n"
                f"        first = handle.readline()\n"
                f"    return first.rstrip('\\n')")
        return ins, body

    def fam_line_count_file(r):
        fn = pick(r, ["count_words_in_file", "file_word_total", "words_in_file", "tally_file_words"])
        ins = f"Write a Python function named {fn}(path) that opens the text file at path and returns the total number of whitespace-separated words in it."
        body = (f"def {fn}(path):\n"
                f"    with open(path, encoding='utf-8') as handle:\n"
                f"        return len(handle.read().split())")
        return ins, body

    def fam_interleave(r):
        a = pick(r, ["first", "left", "primary"])
        b = pick(r, ["second", "right", "secondary"])
        fn = pick(r, ["interleave", "alternate_elements", "weave_lists", "zip_alternate"])
        ins = f"Write a Python function named {fn}({a}, {b}) that takes two lists of integers and returns a new list alternating elements from {a} and {b}. If one list is longer, append the remainder of the longer list at the end."
        body = (f"def {fn}({a}, {b}):\n"
                f"    out = []\n"
                f"    for x, y in zip({a}, {b}):\n"
                f"        out.append(x)\n"
                f"        out.append(y)\n"
                f"    longer = {a} if len({a}) > len({b}) else {b}\n"
                f"    out.extend(longer[min(len({a}), len({b})):])\n"
                f"    return out")
        return ins, body

    def fam_sliding_sum(r):
        arg = pick(r, LIST_POOLS[0])
        w = pick(r, ["width", "window", "span"])
        fn = pick(r, ["sliding_window_sums", "moving_sums", "windowed_sums", "rolling_totals"])
        ins = f"Write a Python function named {fn}({arg}, {w}) that returns the list of sums of every consecutive block of {w} numbers in {arg}. Return an empty list if {w} is larger than len({arg})."
        body = (f"def {fn}({arg}, {w}):\n"
                f"    return [sum({arg}[i:i + {w}]) for i in range(len({arg}) - {w} + 1)]")
        return ins, body

    def fam_rotate_left(r):
        arg = pick(r, LIST_POOLS[0])
        k = pick(r, ["k", "steps", "amount"])
        fn = pick(r, ["rotate_left", "cycle_left", "shift_left", "left_rotation"])
        ins = f"Write a Python function named {fn}({arg}, {k}) that returns the list {arg} rotated left by {k} positions (elements move to the front, preserving order). {k} is between 0 and len({arg}); handle k == 0 and empty lists."
        body = (f"def {fn}({arg}, {k}):\n"
                f"    if not {arg}:\n"
                f"        return []\n"
                f"    {k} = {k} % len({arg})\n"
                f"    return {arg}[{k}:] + {arg}[:{k}]")
        return ins, body

    def fam_first_distinct(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["first_distinct_letter", "first_unrepeated_char", "first_unique_character", "leading_singleton"])
        ins = f"Write a Python function named {fn}({arg}) that returns the first character in the string {arg} that occurs nowhere else in it, or an empty string if there is none."
        body = (f"def {fn}({arg}):\n"
                f"    for ch in {arg}:\n"
                f"        if {arg}.count(ch) == 1:\n"
                f"            return ch\n"
                f"    return ''")
        return ins, body

    def fam_repeat_join(r):
        arg = pick(r, SINGLE_POOLS[0])
        times = pick(r, ["times", "repeats", "count"])
        sep = pick(r, ["sep", "glue", "joint"])
        fn = pick(r, ["repeat_with_separator", "joined_repeats", "repeated_string", "n_copies_joined"])
        ins = f"Write a Python function named {fn}({arg}, {times}, {sep}) that returns {arg} repeated {times} times, with the copies joined by {sep}. Return an empty string when {times} is 0."
        body = (f"def {fn}({arg}, {times}, {sep}):\n"
                f"    return {sep}.join([{arg}] * {times})")
        return ins, body

    def fam_count_negative(r):
        arg = pick(r, LIST_POOLS[0])
        fn = pick(r, ["count_negative", "tally_negatives", "number_of_negatives", "below_zero_count"])
        ins = f"Write a Python function named {fn}({arg}) that returns how many numbers in the list {arg} are negative."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(1 for value in {arg} if value < 0)")
        return ins, body

    def fam_smallest_of_three(r):
        a, b, c = pick(r, [["a", "b", "c"], ["x", "y", "z"], ["p", "q", "r_"]])
        fn = pick(r, ["smallest_of_three", "minimum_of_three", "least_value", "min_triple"])
        ins = f"Write a Python function named {fn}({a}, {b}, {c}) that returns the smallest of the three numbers {a}, {b}, {c}."
        body = (f"def {fn}({a}, {b}, {c}):\n"
                f"    return min({a}, {b}, {c})")
        return ins, body

    def fam_total_characters_no_spaces(r):
        arg = pick(r, SINGLE_POOLS[0])
        fn = pick(r, ["count_non_spaces", "letters_and_digits", "non_space_length", "compact_length"])
        ins = f"Write a Python function named {fn}({arg}) that returns the number of characters in the string {arg} that are not spaces."
        body = (f"def {fn}({arg}):\n"
                f"    return sum(1 for ch in {arg} if ch != ' ')")
        return ins, body

    families = [
        fam_sum_evens, fam_sum_squares, fam_product, fam_max_difference,
        fam_clamp, fam_mean, fam_count_vowels, fam_uppercase_words,
        fam_palindrome, fam_first_index, fam_sum_digits, fam_double_list,
        fam_squares, fam_count_above, fam_word_lengths, fam_first_upper,
        fam_contains_digit, fam_total_length, fam_replace_vowels,
        fam_safe_divide, fam_grade_score, fam_classify_sign, fam_factorial,
        fam_gcd, fam_unique_sorted, fam_invert_dict, fam_dict_value_sum,
        fam_count_uppercase, fam_read_first_line, fam_line_count_file,
        fam_interleave, fam_sliding_sum, fam_rotate_left,
        fam_first_distinct, fam_repeat_join, fam_count_negative,
        fam_smallest_of_three, fam_total_characters_no_spaces,
    ]
    for fam in families:
        for _ in range(8):
            ins, body = fam(rng)
            add(fam.__name__[4:], ins, body)
    pairs = _dedupe(pairs)
    return pairs


def main():
    pairs = build_pairs()
    bench = json.loads(Path("evaluation/l02_python_bench_v1.json").read_text(encoding="utf-8"))
    bench_instructions = {p["instruction"].strip() for p in bench["problems"]}
    bench_entries = {p["entry_point"] for p in bench["problems"]}
    for p in pairs:
        assert p["instruction"].strip() not in bench_instructions, "instruction leaked: " + p["instruction"]
        assert p["response"].split("(")[0].removeprefix("def ") not in bench_entries, "name leaked: " + p["response"]
        ast.parse(p["response"])  # every response must parse

    out = Path("data/l02_sft")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "sft.jsonl"
    with target.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} pairs to {target}")


if __name__ == "__main__":
    main()
