"""Independent v1.2.3 qualification suite (frozen at campaign start).

46 cases: 40 functional + 6 abstention-expected. Novel identifiers,
wordings, propagation depths, compositions, and conditions. Separate from
every frozen suite (AD-1, C3, supplemental-48, generalization-24,
acceptance-31, plus copy probes). Reference answers verified through the
safe functional gate before freezing; do not edit after freezing.
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


_RAW_ORDERED = [
    # copy-family (identifier preservation under wording variation)
    ("Q-c1", "copy", "Return the value untouched.", "def q_copy_a(tag_qz):",
     "    return tag_qz", "q_copy_a", [(("hi",), "hi"), (("z",), "z"), (("",), "")], False),
    ("Q-c2", "copy", "Hand the label back exactly.", "def q_copy_b(label_qz):",
     "    return label_qz", "q_copy_b", [(("memo",), "memo"), (("q",), "q"), (("longer label",), "longer label")], False),
    ("Q-c3", "copy", "Duplicate the input into the output.", "def q_copy_c(item_qz):",
     "    result = item_qz\n    return result", "q_copy_c", [((42,), 42), ((0,), 0), ((-7,), -7)], False),
    # substitution (renamed local, same value)
    ("Q-s1", "substitution", "Rename the amount and return it.",
     "def q_sub_a(amount_qz):", "    renamed = amount_qz\n    return renamed", "q_sub_a",
     [((9,), 9), ((0,), 0), ((3,), 3)], False),
    ("Q-s2", "substitution", "Copy the label under a fresh local name.",
     "def q_sub_b(label_qz):", "    fresh = label_qz\n    return fresh", "q_sub_b",
     [(("tag",), "tag"), (("a",), "a"), (("",), "")], False),
    # propagation (multi-statement locals, depth matters)
    ("Q-p1", "propagation", "Store, normalize, then return the value.",
     "def q_prop_a(value_qz):", "    first = value_qz\n    second = first * 3\n    return second", "q_prop_a",
     [((4,), 12), ((0,), 0), ((-2,), -6)], False),
    ("Q-p2", "propagation", "Move the value across three locals.",
     "def q_prop_b(value_qz):", "    one = value_qz\n    two = one\n    three = two\n    return three", "q_prop_b",
     [((8,), 8), ((-1,), -1), ((0,), 0)], False),
    ("Q-p3", "propagation", "Accumulate, then report the accumulated result.",
     "def q_prop_c(values_qz):", "    acc = 0\n    for item in values_qz:\n        acc += item\n    return acc", "q_prop_c",
     [(([1, 2, 3],), 6), (([],), 0), (([5],), 5)], False),
    ("Q-p4", "propagation", "Double within a local, then return it.",
     "def q_prop_d(value_qz):", "    doubled = value_qz * 2\n    return doubled", "q_prop_d",
     [((6,), 12), ((-4,), -8), ((0,), 0)], False),
    ("Q-p5", "propagation", "Carry the text into a cleaned local.",
     "def q_prop_e(text_qz):", "    cleaned = text_qz.strip()\n    return cleaned", "q_prop_e",
     [((" hi ",), "hi"), (("go",), "go"), (("  ",), "")], False),
    # sequential composition (order matters)
    ("Q-q1", "conjunction", "Add the values, then divide by three.",
     "def q_seq_a(first_qz, second_qz):", "    return (first_qz + second_qz) / 3", "q_seq_a",
     [((3, 9), 4.0), ((1, 2), 1.0), ((0, 0), 0.0)], False),
    ("Q-q2", "conjunction", "Multiply the pair, then halve the result.",
     "def q_seq_b(first_qz, second_qz):", "    return first_qz * second_qz // 2", "q_seq_b",
     [((4, 5), 10), ((3, 3), 4), ((0, 9), 0)], False),
    ("Q-q3", "conjunction", "Keep the positives, then sum them.",
     "def q_seq_c(values_qz):", "    total = 0\n    for item in values_qz:\n        if item > 0:\n            total += item\n    return total", "q_seq_c",
     [(([-2, 3, 4],), 7), (([],), 0), (([-1],), 0)], False),
    ("Q-q4", "conjunction", "Uppercase each entry, then count the entries.",
     "def q_seq_d(words_qz):", "    count = 0\n    for item in words_qz:\n        if item.upper():\n            count += 1\n    return count", "q_seq_d",
     [((["a", "b"],), 2), (([],), 0), ((["x"],), 1)], False),
    ("Q-q5", "conjunction", "Join the pair, then report its length.",
     "def q_seq_e(first_qz, second_qz):", "    return len(first_qz + \" \" + second_qz)", "q_seq_e",
     [(("go", "up"), 5), (("", ""), 1), (("a", "bb"), 4)], False),
    ("Q-q6", "conjunction", "Filter the entries above ten and count them.",
     "def q_seq_f(values_qz):", "    count = 0\n    for item in values_qz:\n        if item > 10:\n            count += 1\n    return count", "q_seq_f",
     [(([9, 11, 12],), 2), (([],), 0), (([10],), 0)], False),
    # operation selection
    ("Q-op1", "operation", "Subtract the second value from the first.",
     "def q_op_a(first_qz, second_qz):", "    return first_qz - second_qz", "q_op_a",
     [((9, 4), 5), ((3, 9), -6), ((5, 5), 0)], False),
    ("Q-op2", "operation", "Return the square of the value.",
     "def q_op_b(value_qz):", "    return value_qz * value_qz", "q_op_b",
     [((7,), 49), ((0,), 0), ((-3,), 9)], False),
    ("Q-op3", "operation", "Return the value increased by twenty percent.",
     "def q_op_c(value_qz):", "    return value_qz + value_qz / 5", "q_op_c",
     [((10,), 12.0), ((0,), 0.0), ((-5,), -6.0)], False),
    ("Q-op4", "operation", "Return the second character of the code.",
     "def q_op_d(code_qz):", "    return code_qz[1]", "q_op_d",
     [(("abc",), "b"), (("xy",), "y"), (("zz",), "z")], False),
    ("Q-op5", "operation", "Return the text without outer whitespace.",
     "def q_op_e(text_qz):", "    return text_qz.strip()", "q_op_e",
     [((" hi ",), "hi"), (("go",), "go"), (("  x  ",), "x")], False),
    # filtering / aggregation
    ("Q-f1", "filtering", "Keep entries below zero.",
     "def q_fil_a(values_qz):", "    return [item for item in values_qz if item < 0]", "q_fil_a",
     [(([-1, 2, -3],), [-1, -3]), (([],), []), (([1],), [])], False),
    ("Q-f2", "aggregation", "Add up every entry.",
     "def q_fil_b(values_qz):", "    return sum(values_qz)", "q_fil_b",
     [(([1, 2, 3],), 6), (([],), 0), (([5],), 5)], False),
    ("Q-f3", "filtering", "Keep only the largest entry.",
     "def q_fil_c(values_qz):", "    if values_qz:\n        return max(values_qz)\n    return None", "q_fil_c",
     [(([2, 9],), 9), (([5],), 5), (([3, 3],), 3)], False),
    ("Q-f4", "aggregation", "Report how many entries the collection holds.",
     "def q_fil_d(values_qz):", "    return len(values_qz)", "q_fil_d",
     [(([1],), 1), (([],), 0), ((["a", "b", "c"],), 3)], False),
    # conditions and booleans
    ("Q-b1", "boolean", "Return True when the length is exactly three.",
     "def q_bool_a(text_qz):", "    return len(text_qz) == 3", "q_bool_a",
     [(("abc",), True), (("ab",), False), (("abcd",), False)], False),
    ("Q-b2", "boolean", "Return True when the first value exceeds the second by at least two.",
     "def q_bool_b(first_qz, second_qz):", "    return first_qz - second_qz >= 2", "q_bool_b",
     [((9, 3), True), ((4, 3), False), ((3, 3), False)], False),
    ("Q-b3", "boolean", "Return True when every entry is positive.",
     "def q_bool_c(values_qz):", "    return all(item > 0 for item in values_qz)", "q_bool_c",
     [(([1, 2],), True), (([1, -1],), False), (([],), True)], False),
    ("Q-b4", "boolean", "Return True when the needle is absent.",
     "def q_bool_d(values_qz, needle_qz):", "    return needle_qz not in values_qz", "q_bool_d",
     [(([1, 2], 3), True), (([1, 2], 2), False), (([], 5), True)], False),
    ("Q-b5", "boolean", "Return True when the upper-cased text equals the text.",
     "def q_bool_e(text_qz):", "    return text_qz.upper() == text_qz", "q_bool_e",
     [(("ABC",), True), (("abc",), False), (("",), True)], False),
    # arithmetic composition (3-param, nested)
    ("Q-n1", "composition", "Compute first plus second times third.",
     "def q_nest_a(first_qz, second_qz, third_qz):", "    return first_qz + second_qz * third_qz", "q_nest_a",
     [((2, 3, 4), 14), ((1, 1, 10), 11), ((0, 5, 2), 10)], False),
    ("Q-n2", "composition", "Square the total of the pair.",
     "def q_nest_b(first_qz, second_qz):", "    return (first_qz + second_qz) * (first_qz + second_qz)", "q_nest_b",
     [((3, 4), 49), ((0, 0), 0), ((-1, 1), 0)], False),
    ("Q-n3", "composition", "Halve the product of the pair.",
     "def q_nest_c(first_qz, second_qz):", "    return first_qz * second_qz // 2", "q_nest_c",
     [((4, 5), 10), ((3, 3), 4), ((0, 9), 0)], False),
    ("Q-m1", "arithmetic", "Return the square of the value.",
     "def q_m_a(value_qz):", "    return value_qz * value_qz", "q_m_a",
     [((9,), 81), ((0,), 0), ((-6,), 36)], False),
    ("Q-m2", "collection", "Return the collection without its last entry.",
     "def q_m_b(values_qz):", "    return values_qz[:-1]", "q_m_b",
     [(([1, 2, 3],), [1, 2]), (([],), []), (([7],), [])], False),
    ("Q-m3", "operation", "Halve the value, rounding down.",
     "def q_m_c(value_qz):", "    return value_qz // 2", "q_m_c",
     [((9,), 4), ((8,), 4), ((-3,), -2)], False),
    ("Q-m4", "boolean", "Return True when the count is even.",
     "def q_m_d(values_qz):", "    return len(values_qz) % 2 == 0", "q_m_d",
     [(([1, 2],), True), (([1],), False), (([],), True)], False),
    ("Q-m5", "string", "Duplicate the text.",
     "def q_m_e(text_qz):", "    return text_qz + text_qz", "q_m_e",
     [(("ab",), "abab"), (("",), ""), (("x",), "xx")], False),
    ("Q-m6", "aggregation", "Multiply all entries together.",
     "def q_m_f(values_qz):", "    total = 1\n    for item in values_qz:\n        total *= item\n    return total", "q_m_f",
     [(([2, 3, 4],), 24), (([],), 1), (([5],), 5)], False),
    ("Q-m7", "operation", "Swap the meaning: return the second of the two values.",
     "def q_m_g(first_qz, second_qz):", "    return second_qz", "q_m_g",
     [((2, 9), 9), ((9, 2), 2), ((0, 0), 0)], False),
    # ambiguous / unsupported — must abstain
    ("Q-x1", "ambiguous", "Return the harbor index of the pilot value.",
     "def q_amb_a(index_qz):", "_abstain", "q_amb_a", [], True),
    ("Q-x2", "unsupported", "Synchronously hash the harbor entry.",
     "def q_amb_b(value_qz):", "_abstain", "q_amb_b", [], True),
    ("Q-x3", "ambiguous", "Summarize the closeout notice.",
     "def q_amb_c(notice_qz):", "_abstain", "q_amb_c", [], True),
    ("Q-x4", "unsupported", "Delete the file entry from the harbor.",
     "def q_amb_d(path_qz):", "_abstain", "q_amb_d", [], True),
    ("Q-x5", "ambiguous", "Explain the fenestration.",
     "def q_amb_e(value_qz):", "_abstain", "q_amb_e", [], True),
    ("Q-x6", "unsupported", "Serve this entry over the network.",
     "def q_amb_f(entry_qz):", "_abstain", "q_amb_f", [], True),
]

# Freeze: 40 functional cases + 6 abstention = 46 total.
_CASES = []
for _case_id, _kind, _instruction, _signature, _body, _fname, _pairs, _abstain in _RAW_ORDERED:
    _CASES.append({
        "case_id": _case_id,
        "kind": _kind,
        "behavior": _kind,
        "instruction": _instruction,
        "signature": _signature,
        "reference_body": _body,
        "function_name": _fname,
        "expected_abstain": _abstain,
        "tests": (
            [{"args": list(args), "kwargs": {}, "expected": expected} for args, expected in _pairs]
            if _pairs else []
        ),
    })


def build_cases() -> list[dict]:
    return list(_CASES)


def suite_digest() -> str:
    import hashlib

    return hashlib.sha256(repr(_CASES).encode("utf-8")).hexdigest()
