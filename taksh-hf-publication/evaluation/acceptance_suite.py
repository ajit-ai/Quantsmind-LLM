"""New independent v1.2.1 acceptance suite (frozen for this campaign).

44 held-out cases with novel identifiers, wordings, structures, and
propagation depths, separate from all frozen fixtures and development
examples. Ambiguous/unsupported cases are marked ``expected_abstain`` and
scored on abstention, not execution. Hash recorded in the campaign report
at freeze time; cases must not be edited after freezing.
"""

from __future__ import annotations

_CASES_RAW = [
    # Propagation chains (multi-statement, intermediate locals)
    ("ACC-p1", "propagation", "Store the amount, then return it doubled.",
     "def prop_a(amount_kq):", "    stored = amount_kq\n    return stored * 2", "prop_a",
     [((6,), 12), ((0,), 0), ((-3,), -6)], False),
    ("ACC-p2", "propagation", "Carry the total through two locals unchanged.",
     "def prop_b(total_kq):", "    first = total_kq\n    second = first\n    return second", "prop_b",
     [((4,), 4), ((0,), 0), ((-5,), -5)], False),
    ("ACC-p3", "propagation", "Accumulate the entries, then subtract the tail value.",
     "def prop_c(entries_kq, tail_kq):", "    total = 0\n    for item in entries_kq:\n        total += item\n    return total - tail_kq", "prop_c",
     [(([1, 2, 3], 1), 5), (([], 4), -4), (([5], 5), 0)], False),
    ("ACC-p4", "propagation", "Keep the doubled value in a local and return it.",
     "def prop_d(value_kq):", "    doubled = value_kq * 2\n    return doubled", "prop_d",
     [((7,), 14), ((-1,), -2), ((0,), 0)], False),
    ("ACC-p5", "propagation", "Double the list, then total it.",
     "def prop_e(values_kq):", "    total = 0\n    for item in values_kq:\n        total += item * 2\n    return total", "prop_e",
     [(([1, 2, 3],), 12), (([],), 0), (([4],), 8)], False),
    ("ACC-p6", "propagation", "Copy the label into a local first.",
     "def prop_f(label_kq):", "    local_label = label_kq\n    return local_label", "prop_f",
     [(("memo",), "memo"), (("",), ""), (("x",), "x")], False),
    ("ACC-p7", "propagation", "Total the list, then double the total.",
     "def prop_g(items_kq):", "    total = 0\n    for item in items_kq:\n        total += item\n    return total * 2", "prop_g",
     [(([2, 3],), 10), (([],), 0), (([5],), 10)], False),
    # Conjunctions (ordered compositions)
    ("ACC-c1", "conjunction", "Keep the positive numbers, then count them.",
     "def and_a(values_kq):", "    count = 0\n    for item in values_kq:\n        if item > 0:\n            count += 1\n    return count", "and_a",
     [(([-1, 2, 3],), 2), (([],), 0), (([0, -5],), 0)], False),
    ("ACC-c2", "conjunction", "Uppercase the list entries, then join them with spaces.",
     "def and_b(words_kq):", "    return \" \".join([item.upper() for item in words_kq])", "and_b",
     [((["go", "up"],), "GO UP"), (([],), ""), ((["a"],), "A")], False),
    ("ACC-c3", "conjunction", "Add the values, and also return the count.",
     "def and_c(values_kq):", "    total = 0\n    for item in values_kq:\n        total += item\n    return total", "and_c",
     [(([1, 2],), 3), (([],), 0), (([7],), 7)], "partial"),
    ("ACC-c4", "conjunction", "Keep the negatives, then total them.",
     "def and_d(values_kq):", "    total = 0\n    for item in values_kq:\n        if item < 0:\n            total += item\n    return total", "and_d",
     [(([-1, -2, 3],), -3), (([],), 0), (([4],), 0)], False),
    ("ACC-c5", "conjunction", "Return True only when both values are positive.",
     "def and_e(first_kq, second_kq):", "    return first_kq > 0 and second_kq > 0", "and_e",
     [((2, 3), True), ((2, 0), False), ((-1, 4), False)], False),
    ("ACC-c6", "conjunction", "Double the list, then count the positives.",
     "def and_f(values_kq):", "    count = 0\n    for item in values_kq:\n        if item * 2 > 0:\n            count += 1\n    return count", "and_f",
     [(([-1, 2],), 1), (([],), 0), (([3],), 1)], False),
    ("ACC-c7", "conjunction", "Join the pair, then return it shouting.",
     "def and_g(first_kq, second_kq):", "    return (first_kq + \" \" + second_kq).upper()", "and_g",
     [(("go", "up"), "GO UP"), (("", ""), " "), (("a", "b"), "A B")], False),
    ("ACC-c8", "conjunction", "Filter the evens, then halve their total.",
     "def and_h(values_kq):", "    total = 0\n    for item in values_kq:\n        if item % 2 == 0:\n            total += item\n    return total // 2", "and_h",
     [(([2, 3, 4],), 3), (([],), 0), (([7],), 0)], False),
    # Operation compositions
    ("ACC-o1", "operation", "Scale the value by three, then add ten.",
     "def op_a(value_kq):", "    return value_kq * 3 + 10", "op_a",
     [((4,), 22), ((0,), 10), ((-2,), 4)], False),
    ("ACC-o2", "operation", "Subtract the second value from the first, then double it.",
     "def op_b(first_kq, second_kq):", "    return (first_kq - second_kq) * 2", "op_b",
     [((9, 3), 12), ((3, 9), -12), ((5, 5), 0)], False),
    ("ACC-o3", "operation", "Return the product modulo three.",
     "def op_c(first_kq, second_kq):", "    return first_kq * second_kq % 3", "op_c",
     [((4, 5), 2), ((6, 6), 0), ((7, 2), 2)], False),
    ("ACC-o4", "operation", "Return the average of the two values.",
     "def op_d(first_kq, second_kq):", "    return (first_kq + second_kq) / 2", "op_d",
     [((4, 6), 5.0), ((0, 0), 0.0), ((3, 7), 5.0)], False),
    ("ACC-o5", "operation", "Return True when the value is even.",
     "def op_e(value_kq):", "    return value_kq % 2 == 0", "op_e",
     [((4,), True), ((3,), False), ((0,), True)], False),
    ("ACC-o6", "operation", "Sort the list in ascending order.",
     "def op_f(values_kq):", "    return sorted(values_kq)", "op_f",
     [(([3, 1, 2],), [1, 2, 3]), (([],), []), (([5, 5],), [5, 5])], False),
    ("ACC-o7", "operation", "Return the first word of the message.",
     "def op_g(message_kq):", "    return message_kq.split()[0]", "op_g",
     [(("hi there",), "hi"), (("go",), "go"), (("a b c",), "a")], False),
    ("ACC-o8", "operation", "Return the last word of the message.",
     "def op_h(message_kq):", "    return message_kq.split()[-1]", "op_h",
     [(("hi there",), "there"), (("go",), "go"), (("a b c",), "c")], False),
    # Arithmetic / collection balance
    ("ACC-a1", "arithmetic", "Return the quotient rounded down.",
     "def ar_a(first_kq, second_kq):", "    return first_kq // second_kq", "ar_a",
     [((9, 2), 4), ((8, 4), 2), ((3, 7), 0)], False),
    ("ACC-a2", "arithmetic", "Return the sum of the three values.",
     "def ar_b(first_kq, second_kq, third_kq):", "    return first_kq + second_kq + third_kq", "ar_b",
     [((1, 2, 3), 6), ((0, 0, 0), 0), ((-1, 1, 5), 5)], False),
    ("ACC-l1", "collection", "Reverse the list.",
     "def li_a(values_kq):", "    return values_kq[::-1]", "li_a",
     [(([1, 2, 3],), [3, 2, 1]), (([],), []), (([7],), [7])], False),
    ("ACC-l2", "collection", "Return the largest entry.",
     "def li_b(values_kq):", "    return max(values_kq)", "li_b",
     [(([1, 7, 3],), 7), (([5],), 5), (([2, 2],), 2)], False),
    # Ambiguous / unsupported: must abstain (no confident mapping)
    ("ACC-x1", "ambiguous", "Compute the fractional part of the index value.",
     "def amb_a(index_kq):", "_abstain", "amb_a", [], True),
    ("ACC-x2", "ambiguous", "Return the color of the house.",
     "def amb_b(house_kq):", "_abstain", "amb_b", [], True),
    ("ACC-x3", "unsupported", "Cryptographically hash the input.",
     "def amb_c(value_kq):", "_abstain", "amb_c", [], True),
    ("ACC-x4", "ambiguous", "Return the previous file version.",
     "def amb_d(version_kq):", "_abstain", "amb_d", [], True),
]


_CASES = []
for _case_id, _kind, _instruction, _signature, _body, _fname, _pairs, _abstain in _CASES_RAW:
    if _body == "_unused":
        continue
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
