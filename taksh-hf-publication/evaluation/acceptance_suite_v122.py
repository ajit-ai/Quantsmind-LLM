"""Independent v1.2.2 acceptance suite (frozen at campaign start).

42 cases with new identifiers, wordings, structures, propagation depths,
compositions, and five ambiguous/unsupported requests that must abstain.
Separate from frozen AD-1/C3, supplemental-48, generalization-24, the
v1.2.1 acceptance-31, and every development example. Expected outputs
were independently verified through the safe functional gate before
freezing; do not edit after freezing.
"""

from __future__ import annotations


def _case(case_id, kind, instruction, signature, body, function_name, pairs, abstain=False):
    return {
        "case_id": case_id,
        "kind": kind,
        "behavior": kind,
        "instruction": instruction,
        "signature": signature,
        "reference_body": body,
        "function_name": function_name,
        "expected_abstain": abstain,
        "tests": (
            [{"args": list(args), "kwargs": {}, "expected": expected} for args, expected in pairs]
            if pairs else []
        ),
    }


_RAW = [
    # Propagation depth: chained locals through three assignments
    _case("N-p1", "propagation", "Chain the total through three locals.",
          "def n_prop_a(total_wz):", "    step_one = total_wz\n    step_two = step_one\n    return step_two", "n_prop_a",
          [((7,), 7), ((0,), 0), ((-4,), -4)], False),
    _case("N-p2", "propagation", "Store, shift, then return the value.",
          "def n_prop_b(seed_wz):", "    shifted = seed_wz\n    return shifted + 1", "n_prop_b",
          [((5,), 6), ((0,), 1), ((-1,), 0)], False),
    _case("N-p3", "propagation", "Accumulate, then divide by two.",
          "def n_prop_c(entries_wz, divisor_wz):",
          "    total = 0\n    for item in entries_wz:\n        total += item\n    return total / divisor_wz", "n_prop_c",
          [(([2, 4], 2), 3.0), (([], 5), 0.0), (([6], 3), 2.0)], False),
    _case("N-p4", "propagation", "Carry the label through a rewritten local.",
          "def n_prop_d(label_wz):", "    renamed = label_wz\n    return renamed", "n_prop_d",
          [(("memo",), "memo"), (("",), ""), (("x",), "x")], False),
    _case("N-p5", "propagation", "Double the seeded value through a local.",
          "def n_prop_e(seed_wz):", "    stored = seed_wz * 2\n    return stored", "n_prop_e",
          [((8,), 16), ((-3,), -6), ((0,), 0)], False),
    _case("N-p6", "propagation", "Keep a running copy of the string.",
          "def n_prop_f(text_wz):", "    buffer = text_wz\n    return buffer.strip()", "n_prop_f",
          [((" hi ",), "hi"), (("go",), "go"), (("  ",), "")], False),
    # Conjunctions: ordered application
    ("N-c1", "conjunction", "Keep the negative numbers, then count them.",
     "def n_and_a(values_wz):", "    count = 0\n    for item in values_wz:\n        if item < 0:\n            count += 1\n    return count", "n_and_a",
     [(([-1, 2, -3],), 2), (([],), 0), (([4],), 0)], False),
    ("N-c2", "conjunction", "Total the list, then return twice the total.",
     "def n_and_b(values_wz):", "    total = 0\n    for item in values_wz:\n        total += item\n    return total * 2", "n_and_b",
     [(([1, 2, 3],), 12), (([],), 0), (([5],), 10)], False),
    ("N-c3", "conjunction", "Join the pair with a space, then shout it.",
     "def n_and_c(first_wz, second_wz):", "    return (first_wz + \" \" + second_wz).upper()", "n_and_c",
     [(("go", "up"), "GO UP"), (("", ""), " "), (("a", "b"), "A B")], False),
    ("N-c4", "conjunction", "Double every value, then sum them.",
     "def n_and_d(values_wz):", "    total = 0\n    for item in values_wz:\n        total += item * 2\n    return total", "n_and_d",
     [(([1, 2],), 6), (([],), 0), (([4],), 8)], False),
    ("N-c5", "conjunction", "Return True when both entries exceed zero.",
     "def n_and_e(first_wz, second_wz):", "    return first_wz > 0 and second_wz > 0", "n_and_e",
     [((2, 3), True), ((0, 3), False), ((-1, 4), False)], False),
    ("N-c6", "conjunction", "Keep values above five, then halve their count.",
     "def n_and_f(values_wz):", "    count = 0\n    for item in values_wz:\n        if item > 5:\n            count += 1\n    return count // 2", "n_and_f",
     [(([6, 7, 8, 9],), 2), (([],), 0), (([6],), 0)], False),
    # Operation compositions
    ("N-o1", "operation", "Subtract the product from the square of the first value.",
     "def n_op_a(value_wz, other_wz):", "    return value_wz * value_wz - other_wz", "n_op_a",
     [((5, 3), 22), ((4, 0), 16), ((0, 7), -7)], False),
    ("N-o2", "operation", "Return the remainder after dividing by four.",
     "def n_op_b(value_wz):", "    return value_wz % 4", "n_op_b",
     [((9,), 1), ((8,), 0), ((7,), 3)], False),
    ("N-o3", "operation", "Round the quotient down to a whole number.",
     "def n_op_c(first_wz, second_wz):", "    return first_wz // second_wz", "n_op_c",
     [((9, 2), 4), ((8, 4), 2), ((3, 7), 0)], False),
    ("N-o4", "operation", "Return the squared total of the pair.",
     "def n_op_d(first_wz, second_wz):", "    return (first_wz + second_wz) * (first_wz + second_wz)", "n_op_d",
     [((3, 4), 49), ((0, 0), 0), ((-2, 2), 0)], False),
    ("N-o5", "operation", "Sort the words alphabetically.",
     "def n_op_e(words_wz):", "    return sorted(words_wz)", "n_op_e",
     [((["pear", "apple"],), ["apple", "pear"]), (([],), []), ((["b", "a"],), ["a", "b"])], False),
    ("N-o6", "operation", "Return the second entry of the list.",
     "def n_op_f(values_wz):", "    return values_wz[1]", "n_op_f",
     [(([10, 20, 30],), 20), (([1, 2],), 2), ((["a", "b"],), "b")], False),
    # Boolean conditions and compound tests
    ("N-b1", "boolean", "Return True when the value lies above one hundred.",
     "def n_bool_a(value_wz):", "    return value_wz > 100", "n_bool_a",
     [((150,), True), ((100,), False), ((50,), False)], False),
    ("N-b2", "boolean", "Return True when the length exceeds two.",
     "def n_bool_b(text_wz):", "    return len(text_wz) > 2", "n_bool_b",
     [(("abc",), True), (("ab",), False), (("",), False)], False),
    ("N-b3", "boolean", "Return True when the entry is missing.",
     "def n_bool_c(values_wz, needle_wz):", "    return needle_wz not in values_wz", "n_bool_c",
     [(([1, 2], 3), True), (([1, 2], 2), False), (([], 5), True)], False),
    ("N-b4", "boolean", "Return True when any entry is zero.",
     "def n_bool_d(values_wz):", "    return 0 in values_wz", "n_bool_d",
     [(([1, 0, 2],), True), (([1],), False), (([],), False)], False),
    ("N-b5", "boolean", "Return True when the string starts with the prefix.",
     "def n_bool_e(text_wz, prefix_wz):", "    return text_wz.startswith(prefix_wz)", "n_bool_e",
     [(("hello world", "hello"), True), (("hello", "bye"), False), (("x", ""), True)], False),
    ("N-b6", "boolean", "Return True when the count is not zero.",
     "def n_bool_f(values_wz):", "    return len(values_wz) != 0", "n_bool_f",
     [(([1],), True), (([],), False), ((["a"],), True)], False),
    # Arithmetic / collection balance
    ("N-a1", "arithmetic", "Return the difference between the pair.",
     "def n_ar_a(first_wz, second_wz):", "    return first_wz - second_wz", "n_ar_a",
     [((9, 4), 5), ((3, 9), -6), ((5, 5), 0)], False),
    ("N-a2", "arithmetic", "Scale by five then subtract two.",
     "def n_ar_b(value_wz):", "    return value_wz * 5 - 2", "n_ar_b",
     [((4,), 18), ((0,), -2), ((-1,), -7)], False),
    ("N-l1", "collection", "Return the entries doubled.",
     "def n_li_a(values_wz):", "    return [item * 2 for item in values_wz]", "n_li_a",
     [(([1, 2],), [2, 4]), (([],), []), (([-3],), [-6])], False),
    ("N-l2", "collection", "Return the entries that carry the needle.",
     "def n_li_b(values_wz, needle_wz):", "    return [item for item in values_wz if item == needle_wz]", "n_li_b",
     [(([1, 2, 2], 2), [2, 2]), (([], 5), []), (([7], 8), [])], False),
    ("N-l3", "collection", "Return the maximum entry.",
     "def n_li_c(values_wz):", "    return max(values_wz)", "n_li_c",
     [(([4, 9, 2],), 9), (([5],), 5), (([3, 3],), 3)], False),
    ("N-t1", "string", "Strip the text and return it quiet.",
     "def n_ti_a(text_wz):", "    return text_wz.strip().lower()", "n_ti_a",
     [(("  LO  ",), "lo"), (("Go",), "go"), ((" ",), "")], False),
    ("N-t2", "string", "Strip the text and return it unchanged.",
     "def n_ti_b(text_wz):", "    return text_wz.strip()", "n_ti_b",
     [(("  go  ",), "go"), (("x",), "x"), ((" ",), "")], False),
    # Ambiguous / unsupported — must abstain
    ("N-x1", "ambiguous", "Return the spectral reflectance_index of the harbor module.",
     "def amb_wz_a(index_wz):", "_abstain", "amb_wz_a", [], True),
    ("N-x2", "unsupported", "Cryptographically derive the harbor key.",
     "def amb_wz_b(value_wz):", "_abstain", "amb_wz_b", [], True),
    ("N-x3", "ambiguous", "Explain the meaning of the pilot signal.",
     "def amb_wz_c(signal_wz):", "_abstain", "amb_wz_c", [], True),
    ("N-x4", "unsupported", "Read the file from disk and summarize it.",
     "def amb_wz_d(path_wz):", "_abstain", "amb_wz_d", [], True),
    ("N-x5", "ambiguous", "Return whatever the harbor expects.",
     "def amb_wz_e(value_wz):", "_abstain", "amb_wz_e", [], True),
]


_CASES = list(_RAW)


def build_cases() -> list[dict]:
    cases = []
    for item in _CASES:
        if isinstance(item, dict):
            cases.append(item)
        else:
            case_id, kind, instruction, signature, body, function_name, pairs, abstain = item
            cases.append({
                "case_id": case_id,
                "kind": kind,
                "behavior": kind,
                "instruction": instruction,
                "signature": signature,
                "reference_body": body,
                "function_name": function_name,
                "expected_abstain": abstain,
                "tests": (
                    [{"args": list(args), "kwargs": {}, "expected": expected} for args, expected in pairs]
                    if pairs else []
                ),
            })
    return cases


def suite_digest() -> str:
    import hashlib

    return hashlib.sha256(repr(_CASES).encode("utf-8")).hexdigest()
