"""Additional held-out generalization cases (integrated campaign).

Novel identifier names, positions, structures, and propagation depths —
not used in implementation development examples. Separate from all frozen
fixtures. Every case carries an explicit expected-output relationship.
"""

from __future__ import annotations


def _case(case_id, kind, instruction, signature, reference_body, function_name, pairs):
    tests = [
        {"args": list(args), "kwargs": {}, "expected": expected}
        for args, expected in pairs
    ]
    return {
        "case_id": case_id,
        "kind": kind,
        "instruction": instruction,
        "signature": signature,
        "reference_body": reference_body,
        "function_name": function_name,
        "tests": tests,
    }


CASES = [
    _case("GEN-div-001", "div", "Divide the first value by the second.",
          "def div_probe(header_qty, unit_qty):", "    return header_qty / unit_qty", "div_probe",
          [((10, 2), 5.0), ((9, 3), 3.0), ((7, 2), 3.5)]),
    _case("GEN-div-002", "div", "Return the quotient.",
          "def quotient_fn(amount_tag, divisor_tag):", "    return amount_tag / divisor_tag", "quotient_fn",
          [((8, 4), 2.0), ((5, 2), 2.5), ((0, 7), 0.0)]),
    _case("GEN-mod-001", "mod", "Return the remainder.",
          "def rem_probe(total_qty, size_qty):", "    return total_qty % size_qty", "rem_probe",
          [((10, 3), 1), ((9, 3), 0), ((7, 4), 3)]),
    _case("GEN-nested-001", "nested", "Add, then multiply the sum by the third value.",
          "def nested_fn(first_part, second_part, third_part):", "    return (first_part + second_part) * third_part", "nested_fn",
          [((2, 3, 4), 20), ((1, 1, 10), 20), ((0, 5, 2), 10)]),
    _case("GEN-nested-002", "nested", "Multiply the pair, then add the third.",
          "def nested_two(alpha_hd, beta_hd, gamma_hd):", "    return alpha_hd * beta_hd + gamma_hd", "nested_two",
          [((2, 3, 4), 10), ((5, 1, 0), 5), ((3, 3, 3), 12)]),
    _case("GEN-str-001", "string", "Strip outer spaces and shout the text.",
          "def shout_probe(raw_entry):", "    return raw_entry.strip().upper()", "shout_probe",
          [((" hi ",), "HI"), (("  go  ",), "GO"), (("x",), "X")]),
    _case("GEN-str-002", "string", "Quiet the shouting text.",
          "def quiet_probe(noisy_entry):", "    return noisy_entry.lower()", "quiet_probe",
          [(("LOUD",), "loud"), (("AbC",), "abc"), (("",), "")]),
    _case("GEN-cond-001", "conditional", "Return the first when it exceeds the second, else the second.",
          "def pick_probe(high_hd, low_hd):", "    if high_hd > low_hd:\n        return high_hd\n    return low_hd", "pick_probe",
          [((9, 3), 9), ((2, 8), 8), ((5, 5), 5)]),
    _case("GEN-cond-002", "conditional", "Return the smaller value.",
          "def small_probe(upper_hd, lower_hd):", "    if upper_hd < lower_hd:\n        return upper_hd\n    return lower_hd", "small_probe",
          [((3, 9), 3), ((8, 2), 2), ((4, 4), 4)]),
    _case("GEN-list-001", "list", "Double every value in the list.",
          "def double_probe(seq_hd):", "    return [item * 2 for item in seq_hd]", "double_probe",
          [(([1, 2, 3],), [2, 4, 6]), (([],), []), (([-1, 0],), [-2, 0])]),
    _case("GEN-list-002", "list", "Keep only the even entries.",
          "def even_keep(seq_hd):", "    return [item for item in seq_hd if item % 2 == 0]", "even_keep",
          [(([1, 2, 3, 4],), [2, 4]), (([],), []), (([7],), [])]),
    _case("GEN-list-003", "list", "Count the long entries.",
          "def long_count(seq_hd):", "    count = 0\n    for item in seq_hd:\n        if len(item) > 3:\n            count += 1\n    return count", "long_count",
          [((["a", "bees", "cat"],), 1), (([],), 0), ((["four", "five"],), 2)]),
    _case("GEN-prop-001", "propagation", "Store the value, then double it.",
          "def carry_probe(value_hd):", "    stored = value_hd\n    return stored * 2", "carry_probe",
          [((6,), 12), ((0,), 0), ((-3,), -6)]),
    _case("GEN-prop-002", "propagation", "Chain the value through two locals.",
          "def chain_probe(seed_hd):", "    first = seed_hd\n    second = first + 1\n    return second", "chain_probe",
          [((4,), 5), ((0,), 1), ((-1,), 0)]),
    _case("GEN-prop-003", "propagation", "Accumulate the list into a local.",
          "def acc_probe(items_hd):", "    total = 0\n    for item in items_hd:\n        total += item\n    return total", "acc_probe",
          [(([1, 2, 3],), 6), (([],), 0), (([5],), 5)]),
    _case("GEN-bool-001", "boolean", "Return True when the value is positive.",
          "def pos_probe(num_hd):", "    return num_hd > 0", "pos_probe",
          [((3,), True), ((0,), False), ((-2,), False)]),
    _case("GEN-bool-002", "boolean", "Return True when both hold.",
          "def both_probe(flag_hd, other_hd):", "    return flag_hd > 0 and other_hd > 0", "both_probe",
          [((2, 3), True), ((2, 0), False), ((-1, 4), False)]),
    _case("GEN-join2-001", "join", "Join with a comma.",
          "def comma_probe(first_hd, second_hd):", "    return first_hd + \", \" + second_hd", "comma_probe",
          [(("a", "b"), "a, b"), (("", "x"), ", x"), (("x", ""), "x, ")]),
    _case("GEN-scale2-001", "scale", "Scale by three.",
          "def triple_probe(value_hd):", "    return value_hd * 3", "triple_probe",
          [((4,), 12), ((0,), 0), ((-2,), -6)]),
    _case("GEN-max2-001", "max", "Return the greater of the two.",
          "def greater_probe(alpha_hd, beta_hd):", "    if alpha_hd > beta_hd:\n        return alpha_hd\n    return beta_hd", "greater_probe",
          [((7, 2), 7), ((2, 7), 7), ((5, 5), 5)]),
    _case("GEN-min2-001", "min", "Return the lesser of the two.",
          "def lesser_probe(alpha_hd, beta_hd):", "    if alpha_hd < beta_hd:\n        return alpha_hd\n    return beta_hd", "lesser_probe",
          [((7, 2), 2), ((2, 7), 2), ((5, 5), 5)]),
    _case("GEN-sum2-001", "sum", "Add the three values.",
          "def three_sum(a_hd, b_hd, c_hd):", "    return a_hd + b_hd + c_hd", "three_sum",
          [((1, 2, 3), 6), ((0, 0, 0), 0), ((-1, 1, 5), 5)]),
    _case("GEN-label-001", "string", "Return the label unchanged.",
          "def label_probe(name_hd):", "    return name_hd", "label_probe",
          [(("memo",), "memo"), (("",), ""), (("x",), "x")]),
    _case("GEN-swap-001", "swap", "Return the second value.",
          "def second_probe(first_hd, second_hd):", "    return second_hd", "second_probe",
          [((3, 8), 8), ((8, 3), 3), ((5, 5), 5)]),
]


def build_cases() -> list[dict]:
    return list(CASES)


def suite_digest() -> str:
    import hashlib

    return hashlib.sha256(repr(CASES).encode("utf-8")).hexdigest()
