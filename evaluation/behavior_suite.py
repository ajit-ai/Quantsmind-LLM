"""Supplemental behavior suite (Taksh v1.2.0 Increment 6).

Deterministic, clearly-identified supplemental evaluation cases for the
eight established functional behaviors. This suite is **not** frozen AD-1
or C3 and must never be reported as such. Wording, function names,
parameter names, and sample values are novel relative to the training
builders (verified by test), and every case carries an explicit
expected-output relationship. Limited scope: 6 cases per behavior × 8 = 48
cases, so per-behavior rates have wide uncertainty; the suite supports
direction and coverage analysis, not precise rate claims.
"""

from __future__ import annotations

BEHAVIORS = (
    "scale",
    "join",
    "compose",
    "compose_second",
    "filter_count",
    "long_words",
    "is_even",
    "sum_positive",
)


def _case(behavior, instruction, signature, body, function_name, pairs):
    tests = [
        {"args": list(args), "kwargs": {}, "expected": expected}
        for args, expected in pairs
    ]
    return (behavior, instruction, signature, body, function_name, tests)


_CASES_RAW = [
    # Scale: output == amount * factor
    _case("scale", "Multiply the amount by the factor.", "def probe_scale(item_count, factor):",
          "    return item_count * factor", "probe_scale",
          [((9, 6), 54), ((0, 8), 0), ((-4, 5), -20)]),
    _case("scale", "Return the scaled value.", "def scaled_probe(base_amount, multiplier):",
          "    return base_amount * multiplier", "scaled_probe",
          [((7, 3), 21), ((11, 0), 0), ((2, -5), -10)]),
    _case("scale", "Times these together.", "def probe_times(qty_total, ratio):",
          "    return qty_total * ratio", "probe_times",
          [((8, 4), 32), ((3, 6), 18), ((-2, -9), 18)]),
    _case("scale", "Give the product of value and scale.", "def value_scaled(amount_in, scale_by):",
          "    return amount_in * scale_by", "value_scaled",
          [((12, 2), 24), ((6, 5), 30), ((0, 0), 0)]),
    _case("scale", "Scale the count by k.", "def probe_by_k(count_k, k_val):",
          "    return count_k * k_val", "probe_by_k",
          [((5, 9), 45), ((1, 1), 1), ((-7, 2), -14)]),
    _case("scale", "What is n times m?", "def probe_n_m(n_val, m_val):",
          "    return n_val * m_val", "probe_n_m",
          [((4, 7), 28), ((9, 9), 81), ((0, 3), 0)]),
    # Join: output == first + " " + second
    _case("join", "Combine the two words with a space.", "def probe_join(tag_one, tag_two):",
          '    return tag_one + " " + tag_two', "probe_join",
          [(("red", "fox"), "red fox"), (("", "go"), " go"), (("go", ""), "go ")]),
    _case("join", "Put both labels together.", "def labels_probe(left_label, right_label):",
          '    return left_label + " " + right_label', "labels_probe",
          [(("north", "gate"), "north gate"), (("a", "b"), "a b"), (("x", ""), "x ")]),
    _case("join", "Join the pair with one space.", "def pair_probe(head_item, tail_item):",
          '    return head_item + " " + tail_item', "pair_probe",
          [(("sun", "set"), "sun set"), (("", ""), " "), (("up", "down"), "up down")]),
    _case("join", "Return the words side by side.", "def words_probe(first_term, second_term):",
          '    return first_term + " " + second_term', "words_probe",
          [(("cat", "nap"), "cat nap"), (("hi", "there"), "hi there"), (("", "end"), " end")]),
    _case("join", "Glue these strings with a space.", "def glue_probe(token_a, token_b):",
          '    return token_a + " " + token_b', "glue_probe",
          [(("open", "shut"), "open shut"), (("yes", "no"), "yes no"), (("a", ""), "a ")]),
    _case("join", "Make one line from two parts.", "def line_probe(part_a, part_b):",
          '    return part_a + " " + part_b', "line_probe",
          [(("black", "box"), "black box"), (("in", "out"), "in out"), (("", "z"), " z")]),
    # Compose: output == (a + b) * 2
    _case("compose", "Add the pair then double the total.", "def probe_compose(a_probe, b_probe):",
          "    return (a_probe + b_probe) * 2", "probe_compose",
          [((6, 1), 14), ((0, 5), 10), ((-3, 3), 0)]),
    _case("compose", "Double the sum of the two values.", "def probe_double_sum(u_val, v_val):",
          "    return (u_val + v_val) * 2", "probe_double_sum",
          [((4, 4), 16), ((1, -1), 0), ((7, 2), 18)]),
    _case("compose", "First add, then multiply by two.", "def probe_add_two(x_probe, y_probe):",
          "    return (x_probe + y_probe) * 2", "probe_add_two",
          [((10, 2), 24), ((0, 0), 0), ((-5, 5), 0)]),
    _case("compose", "Sum the inputs and scale by two.", "def probe_sum_scale(p_val, q_val):",
          "    return (p_val + q_val) * 2", "probe_sum_scale",
          [((3, 7), 20), ((8, -2), 12), ((1, 1), 4)]),
    _case("compose", "Return two times the combined value.", "def probe_combined(r_val, s_val):",
          "    return (r_val + s_val) * 2", "probe_combined",
          [((9, 1), 20), ((-2, 2), 0), ((6, 6), 24)]),
    _case("compose", "Add them up and double it.", "def probe_up_double(m_val, n_val):",
          "    return (m_val + n_val) * 2", "probe_up_double",
          [((5, 5), 20), ((2, 3), 10), ((-1, -1), -4)]),
    # Compose second: output == a * b + 2
    _case("compose_second", "Multiply the pair then add two.", "def probe_compose_two(a_probe, b_probe):",
          "    return a_probe * b_probe + 2", "probe_compose_two",
          [((6, 1), 8), ((0, 5), 2), ((-3, 3), -7)]),
    _case("compose_second", "Add two to the product.", "def probe_product_two(u_val, v_val):",
          "    return u_val * v_val + 2", "probe_product_two",
          [((4, 4), 18), ((1, 7), 9), ((5, -2), -8)]),
    _case("compose_second", "First multiply, then increase by two.", "def probe_mul_two(x_probe, y_probe):",
          "    return x_probe * y_probe + 2", "probe_mul_two",
          [((10, 2), 22), ((3, 3), 11), ((-2, -2), 6)]),
    _case("compose_second", "Product of the inputs plus two.", "def probe_prod_two(p_val, q_val):",
          "    return p_val * q_val + 2", "probe_prod_two",
          [((7, 1), 9), ((0, 0), 2), ((-4, 2), -6)]),
    _case("compose_second", "Times the values, then plus two.", "def probe_times_two(r_val, s_val):",
          "    return r_val * s_val + 2", "probe_times_two",
          [((9, 2), 20), ((5, 5), 27), ((-1, 4), -2)]),
    _case("compose_second", "Scale and add two.", "def probe_scale_two(m_val, n_val):",
          "    return m_val * n_val + 2", "probe_scale_two",
          [((8, 3), 26), ((2, 2), 6), ((1, -6), -4)]),
    # Filter-count: output == number of elements > 0
    _case("filter_count", "How many entries are above zero?", "def probe_fcount(seq_probe):",
          "    count = 0\n    for item in seq_probe:\n        if item > 0:\n            count += 1\n    return count", "probe_fcount",
          [(([-4, 6, 0, 2],), 2), (([],), 0), (([3],), 1)]),
    _case("filter_count", "Count the positives in this list.", "def probe_count_pos(seq_a):",
          "    count = 0\n    for item in seq_a:\n        if item > 0:\n            count += 1\n    return count", "probe_count_pos",
          [(([0, -1, 8],), 1), (([5, 5],), 2), (([-9],), 0)]),
    _case("filter_count", "Number of values greater than zero.", "def probe_num_pos(seq_b):",
          "    count = 0\n    for item in seq_b:\n        if item > 0:\n            count += 1\n    return count", "probe_num_pos",
          [(([2, -2, 4, -4],), 2), (([1],), 1), (([],), 0)]),
    _case("filter_count", "Tally anything above zero.", "def probe_tally_pos(seq_c):",
          "    count = 0\n    for item in seq_c:\n        if item > 0:\n            count += 1\n    return count", "probe_tally_pos",
          [(([0, 0, 1],), 1), (([-3, -2],), 0), (([9],), 1)]),
    _case("filter_count", "How many positives does the sequence hold?", "def probe_pos_hold(seq_d):",
          "    count = 0\n    for item in seq_d:\n        if item > 0:\n            count += 1\n    return count", "probe_pos_hold",
          [(([3, -1, 0, 7, -7],), 2), (([],), 0), (([4, 4],), 2)]),
    _case("filter_count", "Count entries over zero.", "def probe_over_zero(seq_e):",
          "    count = 0\n    for item in seq_e:\n        if item > 0:\n            count += 1\n    return count", "probe_over_zero",
          [(([-1, 2, 3],), 2), (([0],), 0), (([-5],), 0)]),
    # Long-words: output == [word for word in values if len(word) > 3]
    _case("long_words", "Keep words longer than three letters.", "def probe_long(words_probe):",
          "    return [word for word in words_probe if len(word) > 3]", "probe_long",
          [((["go", "longer", "up", "text"],), ["longer", "text"]), (([],), []), ((["abcd"],), ["abcd"])]),
    _case("long_words", "Return the lengthy entries.", "def probe_entries(words_a):",
          "    return [word for word in words_a if len(word) > 3]", "probe_entries",
          [((["a", "bees", "cat", "dogs"],), ["bees", "dogs"]), ((["x"],), []), ((["four", "five", "six"],), ["four", "five"])]),
    _case("long_words", "Filter to words of length four or more.", "def probe_len_four(words_b):",
          "    return [word for word in words_b if len(word) > 3]", "probe_len_four",
          [((["one", "two", "three"],), ["three"]), ((["long", "word"],), ["long", "word"]), (([],), [])]),
    _case("long_words", "Collect the long words.", "def probe_collect(words_c):",
          "    return [word for word in words_c if len(word) > 3]", "probe_collect",
          [((["sun", "moon", "star"],), ["moon", "star"]), ((["abc"],), []), ((["abcd", "efgh"],), ["abcd", "efgh"])]),
    _case("long_words", "Longer than three characters, please.", "def probe_longer(words_d):",
          "    return [word for word in words_d if len(word) > 3]", "probe_longer",
          [((["hi", "hello"],), ["hello"]), ((["tree", "leaf"],), ["tree", "leaf"]), ((["ok"],), [])]),
    _case("long_words", "Which entries exceed three letters?", "def probe_exceed(words_e):",
          "    return [word for word in words_e if len(word) > 3]", "probe_exceed",
          [((["up", "down", "left"],), ["down", "left"]), ((["abcd"],), ["abcd"]), ((["abc", "ab"],), [])]),
    # Is-even: output == n % 2 == 0
    _case("is_even", "Is this number even?", "def probe_even(num_probe):",
          "    return num_probe % 2 == 0", "probe_even",
          [((8,), True), ((3,), False), ((0,), True)]),
    _case("is_even", "Return True for even inputs.", "def probe_true_even(num_a):",
          "    return num_a % 2 == 0", "probe_true_even",
          [((2,), True), ((5,), False), ((-6,), True)]),
    _case("is_even", "Check evenness of the value.", "def probe_even_check(num_b):",
          "    return num_b % 2 == 0", "probe_even_check",
          [((10,), True), ((7,), False), ((-1,), False)]),
    _case("is_even", "Even or odd?", "def probe_even_odd(num_c):",
          "    return num_c % 2 == 0", "probe_even_odd",
          [((1,), False), ((12,), True), ((100,), True)]),
    _case("is_even", "Tell me if n is even.", "def probe_n_even(n_probe):",
          "    return n_probe % 2 == 0", "probe_n_even",
          [((9,), False), ((22,), True), ((-4,), True)]),
    _case("is_even", "Does two divide this number?", "def probe_divides(num_d):",
          "    return num_d % 2 == 0", "probe_divides",
          [((14,), True), ((15,), False), ((0,), True)]),
    # Sum-positive: output == sum of elements > 0
    _case("sum_positive", "Add up the positives.", "def probe_sum_pos(seq_probe):",
          "    total = 0\n    for item in seq_probe:\n        if item > 0:\n            total += item\n    return total", "probe_sum_pos",
          [(([-4, 6, 0, 2],), 8), (([],), 0), (([3],), 3)]),
    _case("sum_positive", "Total the values above zero.", "def probe_pos_total(seq_a):",
          "    total = 0\n    for item in seq_a:\n        if item > 0:\n            total += item\n    return total", "probe_pos_total",
          [(([0, -1, 8],), 8), (([5, 5],), 10), (([-9],), 0)]),
    _case("sum_positive", "Sum of positive numbers only.", "def probe_only_pos(seq_b):",
          "    total = 0\n    for item in seq_b:\n        if item > 0:\n            total += item\n    return total", "probe_only_pos",
          [(([2, -2, 4],), 6), (([1],), 1), (([],), 0)]),
    _case("sum_positive", "Add the positives in the sequence.", "def probe_seq_pos_sum(seq_c):",
          "    total = 0\n    for item in seq_c:\n        if item > 0:\n            total += item\n    return total", "probe_seq_pos_sum",
          [(([0, 0, 1],), 1), (([-3, -2],), 0), (([9],), 9)]),
    _case("sum_positive", "What do the positives add to?", "def probe_pos_add(seq_d):",
          "    total = 0\n    for item in seq_d:\n        if item > 0:\n            total += item\n    return total", "probe_pos_add",
          [(([3, -1, 0, 7],), 10), (([],), 0), (([4, 4],), 8)]),
    _case("sum_positive", "Sum entries over zero.", "def probe_over_zero_sum(seq_e):",
          "    total = 0\n    for item in seq_e:\n        if item > 0:\n            total += item\n    return total", "probe_over_zero_sum",
          [(([-1, 2, 3],), 5), (([0],), 0), (([-5],), 0)]),
]


def build_cases() -> list[dict]:
    cases = []
    for index, (behavior, instruction, signature, reference_body, function_name, tests) in enumerate(_CASES_RAW):
        cases.append(
            {
                "case_id": f"SUP-{behavior}-{index:03d}",
                "behavior": behavior,
                "instruction": instruction,
                "signature": signature,
                "reference_body": reference_body,
                "function_name": function_name,
                "tests": tests,
            }
        )
    return cases


def suite_digest() -> str:
    import hashlib

    return hashlib.sha256(repr(build_cases()).encode("utf-8")).hexdigest()
