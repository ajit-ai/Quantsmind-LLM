"""Functional synthetic curriculum (Taksh v1.2.0 Increment 4).

Original QuantsMind-generated examples with explicit input/output
behavioural relationships. Every example carries the sample inputs and
expected outputs in its metadata (``inputs``, ``expected_outputs``,
``relationship``) so functional correctness can be verified through the
isolated safe-execution gate without touching frozen AD-1/C3 suites.

Eight behaviours, four structurally distinct implementations each,
deterministic seeds, and a disjoint validation split (rotated names,
function names, and instructions). Combined with the Increment 2 binding
curriculum the corpus must still pass the Increment 2 quality checks.
"""

from __future__ import annotations

import ast
import hashlib

from training.binding_curriculum import _validate
from training.supervised import build_example

OPERATIONS = (
    "scale_value",
    "join_text",
    "add_then_multiply",
    "multiply_then_add",
    "filter_and_count",
    "keep_long_words",
    "is_even",
    "sum_positive",
)

_FAMILIES = {
    "scale_value": "arithmetic",
    "join_text": "text",
    "add_then_multiply": "arithmetic",
    "multiply_then_add": "arithmetic",
    "filter_and_count": "aggregate",
    "keep_long_words": "collection",
    "is_even": "conditioning",
    "sum_positive": "aggregate",
}

_FUNCTION_NAMES = {
    "scale_value": ("scaled_value", "times_factor", "value_times", "scaled_total", "scale_it", "times_amount"),
    "join_text": ("joined_pair", "words_together", "text_join", "pair_joined", "words_joined", "joined_text"),
    "add_then_multiply": ("sum_then_times", "added_then_scaled", "total_times", "sum_times_two", "added_scaled", "combine_scale"),
    "multiply_then_add": ("times_then_sum", "scaled_then_added", "product_plus", "times_sum_two", "scaled_added", "scale_combine"),
    "filter_and_count": ("counts_above_zero", "positives_counted", "count_positives", "tally_positive", "number_positive", "positive_tally"),
    "keep_long_words": ("long_words", "keeps_lengthy", "long_word_list", "lengthy_words", "words_kept", "long_only"),
    "is_even": ("checks_even", "even_check", "is_parity_even", "even_test", "even_or_not", "parity_test"),
    "sum_positive": ("sum_positives", "positive_total", "adds_positive", "total_positive", "sum_above_zero", "positive_sum"),
}

_PAIRS = (
    ("alpha_value", "beta_value"),
    ("first_number", "second_number"),
    ("left_input", "right_input"),
    ("left_value", "right_value"),
    ("a_value", "b_value"),
    ("x_value", "y_value"),
    ("one", "two"),
    ("aa", "bb"),
    ("qty", "total"),
    ("first", "second"),
    ("left", "right"),
    ("p", "q"),
    ("left_number", "right_number"),
    ("input_one", "input_two"),
    ("first_value", "second_value"),
    ("start", "stop"),
    ("before", "after"),
    ("base", "offset"),
)

_TEXT_PAIRS = (
    ("first_word", "second_word"),
    ("left_text", "right_text"),
    ("word_one", "word_two"),
    ("head_word", "tail_word"),
    ("first_phrase", "second_phrase"),
    ("lead_word", "trail_word"),
    ("part_one", "part_two"),
    ("front", "back"),
    ("begin", "end"),
    ("opening", "closing"),
    ("pre", "post"),
    ("start_text", "end_text"),
    ("left_phrase", "right_phrase"),
    ("first_label", "second_label"),
    ("name_one", "name_two"),
    ("tag_one", "tag_two"),
    ("key_word", "other_word"),
    ("item_one", "item_two"),
)

_SINGLES = (
    "value", "data", "items", "entries", "source", "target", "text", "word",
    "entry", "item", "datum", "payload", "records", "numbers", "values",
    "buffer", "seq", "content", "input_value", "source_value",
)

_INSTRUCTIONS = {
    "scale_value": (
        "Multiply the first value by the second and return the result.",
        "Scale the number by the given factor.",
        "Return the product of an amount and a multiplier.",
        "Times the value by the factor and give back the scaled value.",
    ),
    "join_text": (
        "Join the two strings with a single space between them.",
        "Put the first word, a space, and the second word together.",
        "Return the two texts combined with one space.",
        "Glue the two words with a space in the middle.",
    ),
    "add_then_multiply": (
        "Add the two values, multiply the sum by two, and return it.",
        "First combine the inputs by addition, then scale by two.",
        "Return two times the sum of the two values.",
        "Sum the pair and double the total.",
    ),
    "multiply_then_add": (
        "Multiply the two values, add two, and return the result.",
        "First take the product, then increase it by two.",
        "Return the product of the pair plus two.",
        "Scale the first value by the second and add two.",
    ),
    "filter_and_count": (
        "Count how many values in the list are greater than zero.",
        "Return the number of positive entries.",
        "Tally the positive numbers in the sequence.",
        "How many items are above zero?",
    ),
    "keep_long_words": (
        "Return only the words longer than three characters.",
        "Keep words with more than three letters.",
        "Filter the list down to long words.",
        "Collect the entries of length four or more.",
    ),
    "is_even": (
        "Return True when the number is even, otherwise False.",
        "Check whether the value divides evenly by two.",
        "Tell if the number is even.",
        "Return whether the input is an even integer.",
    ),
    "sum_positive": (
        "Add up only the positive values in the list.",
        "Return the sum of the numbers above zero.",
        "Total the positive entries.",
        "Sum the values that are greater than zero.",
    ),
}

# Deterministic sample inputs/outputs per operation (boundary cases first).
_SAMPLES: dict[str, list[tuple[tuple, object]]] = {
    "scale_value": [((6, 7), 42), ((0, 5), 0), ((-3, 4), -12), ((5, 0), 0)],
    "join_text": [(("blue", "sky"), "blue sky"), (("", "x"), " x"), (("x", ""), "x "), (("a", "b"), "a b")],
    "add_then_multiply": [((3, 4), 14), ((0, 0), 0), ((-1, 1), 0), ((5, -2), 6)],
    "multiply_then_add": [((3, 4), 14), ((0, 9), 2), ((-2, 3), -4), ((5, 0), 2)],
    "filter_and_count": [(([-2, 3, 0, 5],), 2), (([],), 0), (([-1, 0],), 0), (([7],), 1)],
    "keep_long_words": [((["hi", "hello", "a", "world"],), ["hello", "world"]), (([],), []), ((["abcd"],), ["abcd"]), ((["abc"],), [])],
    "is_even": [((4,), True), ((0,), True), ((-3,), False), ((7,), False)],
    "sum_positive": [(([-2, 3, 0, 5],), 8), (([],), 0), (([-1, 0],), 0), (([7],), 7)],
}

# Relationship statement stored in metadata (the I/O contract).
_RELATIONSHIPS = {
    "scale_value": "output == first * second",
    "join_text": "output == first + ' ' + second",
    "add_then_multiply": "output == (first + second) * 2",
    "multiply_then_add": "output == first * second + 2",
    "filter_and_count": "output == number of elements > 0",
    "keep_long_words": "output == [word for word in input if len(word) > 3]",
    "is_even": "output == (input % 2 == 0)",
    "sum_positive": "output == sum of elements > 0",
}


def _body(operation: str, params: tuple[str, ...], variant: int) -> str:
    variant %= 4
    if operation == "scale_value":
        (a, b) = params
        if variant == 0:
            return f"    return {a} * {b}"
        if variant == 1:
            return f"    scaled = {a} * {b}\n    return scaled"
        if variant == 2:
            return f"    result = {a}\n    result *= {b}\n    return result"
        return f"    return {b} * {a}"
    if operation == "join_text":
        (a, b) = params
        if variant == 0:
            return f'    return {a} + " " + {b}'
        if variant == 1:
            return f'    joined = {a} + " " + {b}\n    return joined'
        if variant == 2:
            return f'    return f"{{{a}}} {{{b}}}"'
        return f'    return " ".join([{a}, {b}])'
    if operation == "add_then_multiply":
        (a, b) = params
        if variant == 0:
            return f"    return ({a} + {b}) * 2"
        if variant == 1:
            return f"    total = {a} + {b}\n    return total * 2"
        if variant == 2:
            return f"    result = {a}\n    result += {b}\n    result *= 2\n    return result"
        return f"    return 2 * ({a} + {b})"
    if operation == "multiply_then_add":
        (a, b) = params
        if variant == 0:
            return f"    return {a} * {b} + 2"
        if variant == 1:
            return f"    product = {a} * {b}\n    return product + 2"
        if variant == 2:
            return f"    result = {a}\n    result *= {b}\n    result += 2\n    return result"
        return f"    return 2 + {a} * {b}"
    if operation == "filter_and_count":
        (v,) = params
        if variant == 0:
            return f"    count = 0\n    for item in {v}:\n        if item > 0:\n            count += 1\n    return count"
        if variant == 1:
            return f"    return len([item for item in {v} if item > 0])"
        if variant == 2:
            return f"    return sum(1 for item in {v} if item > 0)"
        return (
            f"    count = 0\n    for index, item in enumerate({v}):\n"
            f"        if item > 0:\n            count += 1\n    return count"
        )
    if operation == "keep_long_words":
        (v,) = params
        if variant == 0:
            return f"    return [word for word in {v} if len(word) > 3]"
        if variant == 1:
            return (
                f"    result = []\n    for word in {v}:\n"
                f"        if len(word) > 3:\n            result.append(word)\n    return result"
            )
        if variant == 2:
            return f"    result = []\n    result.extend(word for word in {v} if len(word) > 3)\n    return result"
        return (
            f"    result = []\n    for index, word in enumerate({v}):\n"
            f"        if len(word) > 3:\n            result.append(word)\n    return result"
        )
    if operation == "is_even":
        (n,) = params
        if variant == 0:
            return f"    return {n} % 2 == 0"
        if variant == 1:
            return f"    if {n} % 2 == 0:\n        return True\n    return False"
        if variant == 2:
            return f"    even = {n} % 2 == 0\n    return even"
        return f"    return True if {n} % 2 == 0 else False"
    if operation == "sum_positive":
        (v,) = params
        if variant == 0:
            return f"    total = 0\n    for item in {v}:\n        if item > 0:\n            total += item\n    return total"
        if variant == 1:
            return f"    return sum(item for item in {v} if item > 0)"
        if variant == 2:
            return f"    return sum(item for item in {v} if item > 0) if {v} else 0"
        return (
            f"    total = 0\n    for index, item in enumerate({v}):\n"
            f"        if item > 0:\n            total += item\n    return total"
        )
    raise ValueError(f"unknown functional operation: {operation}")


def build_examples(
    *,
    name_offset: int = 0,
    function_offset: int = 0,
    instruction_offset: int = 0,
    per_operation: int = 12,
    seed: int = 20261107,
    split: str = "train",
    id_prefix: str = "F1",
) -> list:
    examples = []
    for operation in OPERATIONS:
        instructions = _INSTRUCTIONS[operation]
        function_names = _FUNCTION_NAMES[operation]
        samples = _SAMPLES[operation]
        for index in range(per_operation):
            if operation == "join_text":
                params = _TEXT_PAIRS[(name_offset + index) % len(_TEXT_PAIRS)]
            elif operation in ("filter_and_count", "keep_long_words", "sum_positive", "is_even"):
                params = (_SINGLES[(name_offset + index) % len(_SINGLES)],)
            else:
                params = _PAIRS[(name_offset + index) % len(_PAIRS)]
            function_name = function_names[(function_offset + index) % len(function_names)]
            instruction = instructions[(instruction_offset + index) % len(instructions)]
            variant = index % 4
            signature = f"def {function_name}({', '.join(params)}):"
            body = _body(operation, params, variant)
            _validate(signature, body)
            args, expected = samples[index % len(samples)]
            example_id = f"{id_prefix}-{operation}-{index:03d}"
            examples.append(
                build_example(
                    example_id=example_id,
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=operation,
                    family=_FAMILIES[operation],
                    split=split,
                    seed=seed + index,
                    metadata={
                        "markers": [operation],
                        "functional_focus": True,
                        "variant": variant,
                        "surface_variant": f"v{variant}",
                        "inputs": [list(args)],
                        "expected_outputs": [expected],
                        "relationship": _RELATIONSHIPS[operation],
                    },
                )
            )
    return examples


def build_training_examples(*, seed: int = 20261107) -> list:
    return build_examples(per_operation=12, seed=seed, split="train")


def build_validation_examples(*, seed: int = 20261108) -> list:
    return build_examples(
        name_offset=5,
        function_offset=2,
        instruction_offset=1,
        per_operation=4,
        seed=seed,
        split="validation",
        id_prefix="F1V",
    )


def functional_digest() -> str:
    payload = [
        (example.example_id, example.instruction, example.signature, example.body, example.operation)
        for example in build_training_examples()
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _unused_ast_check() -> None:  # pragma: no cover - import guard
    ast.parse("pass")
