"""Deterministic binding-focused synthetic curriculum for Taksh v1.2.0.

Increment 2 revision. The examples are original to this repository. They
teach the model to copy the requested parameter names from the signature
into the body and to associate the instruction with the correct return
expression. They are not copied from any external dataset.

Increment 1 used one surface template per operation. That single-template
shape caused mode collapse in the held-out evaluation: the candidate
emitted the list-comprehension template for unrelated prompts. This
revision gives every operation four semantically equivalent but
structurally distinct body variants (direct return, named intermediate,
augmented assignment, conditional/builtin/loop alternative, and similar),
plus type-appropriate parameter names and wider instruction paraphrases.

Training examples use ``split="train"``. Held-out evaluation data must
remain separate; this module also provides a distinct validation builder
for smoke checks, but frozen AD-1/C3 suites remain the evaluation
reference. Use :mod:`training.curriculum_quality` to check balance,
template diversity, duplicates, leakage, and binding before any training.
"""

from __future__ import annotations

import ast
import hashlib

from training.supervised import build_example

TRAINED_OPERATIONS = (
    "add",
    "subtract",
    "multiply",
    "maximum",
    "minimum",
    "keep_positive",
    "keep_negative",
    "upper",
    "lower",
    "count_values",
    "sum_values",
    "choose_first",
    "choose_second",
)

_FAMILIES = {
    "add": "arithmetic",
    "subtract": "arithmetic",
    "multiply": "arithmetic",
    "maximum": "conditioning",
    "minimum": "conditioning",
    "keep_positive": "collection",
    "keep_negative": "collection",
    "upper": "text",
    "lower": "text",
    "count_values": "aggregate",
    "sum_values": "aggregate",
    "choose_first": "binding",
    "choose_second": "binding",
}

_FUNCTION_NAMES = {
    "add": ("adds_pair", "sums_two", "total_pair", "pair_total", "adds_inputs", "sums_inputs"),
    "subtract": ("subtracts_pair", "differences_pair", "differs_pair", "subtracts_inputs", "differences_inputs", "diff_inputs"),
    "multiply": ("multiplies_pair", "products_pair", "times_pair", "multiplies_inputs", "products_inputs", "times_inputs"),
    "maximum": ("picks_larger", "selects_greater", "returns_larger", "chooses_greater", "greater_of_pair", "larger_of_values"),
    "minimum": ("picks_smaller", "selects_lesser", "returns_smaller", "chooses_smaller", "smaller_of_pair", "lesser_of_values"),
    "keep_positive": ("collect_positive", "positive_only", "keeps_positive", "select_positive", "filter_positive", "positive_items"),
    "keep_negative": ("collect_negative", "negative_only", "keeps_negative", "select_negative", "filter_negative", "negative_items"),
    "upper": ("shouts_text", "uppercases_value", "capitalises_value", "caps_value", "uppercases_text", "shouts_value"),
    "lower": ("quiets_text", "lowercases_value", "decapitalises_value", "smallcases_value", "lowercases_text", "quiets_value"),
    "count_values": ("counts_items", "tallies_items", "numbers_items", "counts_entries", "tallies_entries", "how_many_items"),
    "sum_values": ("totals_items", "adds_items", "sums_items", "totals_entries", "adds_entries", "sum_items"),
    "choose_first": ("returns_first", "picks_first", "chooses_first", "first_value", "head_value", "front_value"),
    "choose_second": ("returns_second", "picks_second", "chooses_second", "second_value", "tail_value", "next_value"),
}

# Numeric parameter pairs for binary arithmetic / conditioning / binding.
_NUMERIC_PAIRS = (
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
    ("head", "tail"),
    ("lead", "trail"),
)

# String parameter names for text operations.
_STRING_NAMES = (
    "text",
    "word",
    "phrase",
    "title",
    "label",
    "name",
    "message",
    "sentence",
    "caption",
    "header",
    "code",
    "token",
    "note",
    "detail",
    "summary",
    "heading",
    "field",
    "remark",
    "comment",
    "prompt",
)

# Sequence parameter names for collection / aggregate operations.
_SEQUENCE_NAMES = (
    "values",
    "items",
    "numbers",
    "entries",
    "records",
    "data",
    "buffer",
    "sequence",
    "collection",
    "batch",
    "readings",
    "samples",
    "scores",
    "amounts",
    "elements",
    "list_values",
    "entries_list",
    "item_list",
    "value_list",
    "num_list",
)

_BINARY_OPERATIONS = ("add", "subtract", "multiply", "maximum", "minimum", "choose_first", "choose_second")
_TEXT_OPERATIONS = ("upper", "lower")

_INSTRUCTIONS = {
    "add": (
        "Return the sum of two numbers.",
        "Compute a plus b.",
        "Combine two inputs by addition.",
        "Add the two values together and return the total.",
    ),
    "subtract": (
        "Return the difference between two numbers.",
        "Compute a minus b.",
        "Take the second input away from the first.",
        "Subtract the second value from the first and return it.",
    ),
    "multiply": (
        "Return the product of two numbers.",
        "Compute a times b.",
        "Multiply two inputs together.",
        "Return the two values multiplied together.",
    ),
    "maximum": (
        "Return the larger of two values.",
        "Choose the greater input.",
        "Pick the maximum of two numbers.",
        "Return whichever of the two values is bigger.",
    ),
    "minimum": (
        "Return the smaller of two values.",
        "Choose the lesser input.",
        "Pick the minimum of two numbers.",
        "Return whichever of the two values is smaller.",
    ),
    "keep_positive": (
        "Return only positive numbers.",
        "Keep values greater than zero.",
        "Collect positive inputs.",
        "Filter the list down to its positive entries.",
    ),
    "keep_negative": (
        "Return only negative numbers.",
        "Keep values less than zero.",
        "Collect negative inputs.",
        "Filter the list down to its negative entries.",
    ),
    "upper": (
        "Return the text in upper case.",
        "Convert the string to capital letters.",
        "Shout the input text.",
        "Make every letter of the string a capital.",
    ),
    "lower": (
        "Return the text in lower case.",
        "Convert the string to small letters.",
        "Quiet the input text.",
        "Make every letter of the string small.",
    ),
    "count_values": (
        "Count the items in a list.",
        "Return how many entries the list holds.",
        "Tally the items.",
        "Tell me the number of values in the sequence.",
    ),
    "sum_values": (
        "Sum the items in a list.",
        "Add up the list values.",
        "Total the numeric items.",
        "Return the sum of every value in the sequence.",
    ),
    "choose_first": (
        "Return the first parameter.",
        "Pick the first input and ignore the second.",
        "Return only the leading value.",
        "Hand back the first of the two inputs.",
    ),
    "choose_second": (
        "Return the second parameter.",
        "Pick the second input and ignore the first.",
        "Return only the trailing value.",
        "Hand back the second of the two inputs.",
    ),
}

# Number of semantically equivalent surface variants per operation.
VARIANT_COUNT = 4


def _body(operation: str, params: tuple[str, ...], variant: int) -> str:
    """Return one of four equivalent surface forms for ``operation``."""
    variant %= VARIANT_COUNT
    if operation == "add":
        a, b = params
        if variant == 0:
            return f"    return {a} + {b}"
        if variant == 1:
            return f"    total = {a} + {b}\n    return total"
        if variant == 2:
            return f"    result = {a}\n    result += {b}\n    return result"
        return f"    return {b} + {a}"
    if operation == "subtract":
        a, b = params
        if variant == 0:
            return f"    return {a} - {b}"
        if variant == 1:
            return f"    difference = {a} - {b}\n    return difference"
        if variant == 2:
            return f"    result = {a}\n    result -= {b}\n    return result"
        return f"    return {a} + (-{b})"
    if operation == "multiply":
        a, b = params
        if variant == 0:
            return f"    return {a} * {b}"
        if variant == 1:
            return f"    product = {a} * {b}\n    return product"
        if variant == 2:
            return f"    result = {a}\n    result *= {b}\n    return result"
        return f"    return {b} * {a}"
    if operation == "maximum":
        a, b = params
        if variant == 0:
            return f"    if {a} > {b}:\n        return {a}\n    return {b}"
        if variant == 1:
            return f"    return {a} if {a} > {b} else {b}"
        if variant == 2:
            return f"    chosen = {a}\n    if {b} > chosen:\n        chosen = {b}\n    return chosen"
        return f"    return {a} if {a} >= {b} else {b}"
    if operation == "minimum":
        a, b = params
        if variant == 0:
            return f"    if {a} < {b}:\n        return {a}\n    return {b}"
        if variant == 1:
            return f"    return {a} if {a} < {b} else {b}"
        if variant == 2:
            return f"    chosen = {a}\n    if {b} < chosen:\n        chosen = {b}\n    return chosen"
        return f"    return {a} if {a} <= {b} else {b}"
    if operation in ("keep_positive", "keep_negative"):
        (value,) = params
        symbol = ">" if operation == "keep_positive" else "<"
        if variant == 0:
            return f"    return [item for item in {value} if item {symbol} 0]"
        if variant == 1:
            return (
                f"    result = []\n"
                f"    for item in {value}:\n"
                f"        if item {symbol} 0:\n"
                f"            result.append(item)\n"
                f"    return result"
            )
        if variant == 2:
            return f"    result = []\n    result.extend(item for item in {value} if item {symbol} 0)\n    return result"
        return (
            f"    result = []\n"
            f"    index = 0\n"
            f"    while index < len({value}):\n"
            f"        item = {value}[index]\n"
            f"        if item {symbol} 0:\n"
            f"            result.append(item)\n"
            f"        index += 1\n"
            f"    return result"
        )
    if operation in ("upper", "lower"):
        (text,) = params
        method = "upper" if operation == "upper" else "lower"
        if variant == 0:
            return f"    return {text}.{method}()"
        if variant == 1:
            local = "shouted" if operation == "upper" else "quieted"
            return f"    {local} = {text}.{method}()\n    return {local}"
        if variant == 2:
            return f"    if {text}:\n        return {text}.{method}()\n    return {text}"
        return f'    return f"{{{text}}}".{method}()'
    if operation == "count_values":
        (items,) = params
        if variant == 0:
            return f"    count = 0\n    for item in {items}:\n        count += 1\n    return count"
        if variant == 1:
            return f"    return len({items})"
        if variant == 2:
            return f"    return sum(1 for item in {items})"
        return (
            f"    count = 0\n"
            f"    index = 0\n"
            f"    while index < len({items}):\n"
            f"        count += 1\n"
            f"        index += 1\n"
            f"    return count"
        )
    if operation == "sum_values":
        (items,) = params
        if variant == 0:
            return f"    total = 0\n    for item in {items}:\n        total += item\n    return total"
        if variant == 1:
            return f"    return sum({items})"
        if variant == 2:
            return f"    return sum(item for item in {items})"
        return (
            f"    total = 0\n"
            f"    index = 0\n"
            f"    while index < len({items}):\n"
            f"        total += {items}[index]\n"
            f"        index += 1\n"
            f"    return total"
        )
    if operation == "choose_first":
        first, _second = params
        if variant == 0:
            return f"    return {first}"
        if variant == 1:
            return f"    chosen = {first}\n    return chosen"
        if variant == 2:
            return f'    """Return the first value."""\n    return {first}'
        return f"    chosen = {first}\n    if True:\n        return chosen\n    return {first}"
    if operation == "choose_second":
        _first, second = params
        if variant == 0:
            return f"    return {second}"
        if variant == 1:
            return f"    chosen = {second}\n    return chosen"
        if variant == 2:
            return f'    """Return the second value."""\n    return {second}'
        return f"    chosen = {second}\n    if True:\n        return chosen\n    return {second}"
    raise ValueError(f"unknown binding operation: {operation}")


def _signature(operation: str, function_name: str, params: tuple[str, ...]) -> str:
    return f"def {function_name}({', '.join(params)}):"


def _validate(signature: str, body: str) -> None:
    source = signature + "\n" + body
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("example did not parse as one function")
    params = [arg.arg for arg in tree.body[0].args.args]
    assigned: set[str] = set()
    for node in ast.walk(tree.body[0]):
        if isinstance(node, ast.arg):
            assigned.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            assigned.add(node.id)
        elif isinstance(node, (ast.For, ast.comprehension)):
            for target_node in ast.walk(node.target):
                if isinstance(target_node, ast.Name):
                    assigned.add(target_node.id)
    used = {node.id for node in ast.walk(tree.body[0]) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    undefined = used - set(params) - assigned - {"len", "sum", "list", "str", "filter", "enumerate", "range"}
    if undefined:
        raise ValueError(f"example has undefined names: {sorted(undefined)}")


def build_examples(
    *,
    name_offset: int = 0,
    function_offset: int = 0,
    instruction_offset: int = 0,
    per_operation: int = 12,
    seed: int = 20261024,
    split: str = "train",
    id_prefix: str = "B1",
) -> list:
    examples = []
    for operation in TRAINED_OPERATIONS:
        instructions = _INSTRUCTIONS[operation]
        function_names = _FUNCTION_NAMES[operation]
        for index in range(per_operation):
            if operation in _TEXT_OPERATIONS:
                params = (_STRING_NAMES[(name_offset + index) % len(_STRING_NAMES)],)
            elif operation in ("keep_positive", "keep_negative", "count_values", "sum_values"):
                params = (_SEQUENCE_NAMES[(name_offset + index) % len(_SEQUENCE_NAMES)],)
            else:
                params = _NUMERIC_PAIRS[(name_offset + index) % len(_NUMERIC_PAIRS)]
            function_name = function_names[(function_offset + index) % len(function_names)]
            instruction = instructions[(instruction_offset + index) % len(instructions)]
            variant = index % VARIANT_COUNT
            signature = _signature(operation, function_name, params)
            body = _body(operation, params, variant)
            _validate(signature, body)
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
                        "binding_focus": True,
                        "variant": variant,
                        "surface_variant": f"v{variant}",
                    },
                )
            )
    return examples


def build_training_examples(*, seed: int = 20261024) -> list:
    return build_examples(name_offset=0, per_operation=12, seed=seed, split="train")


def build_validation_examples(*, seed: int = 20261025) -> list:
    # Rotated function names and a different name offset guarantee that no
    # (instruction, signature) pair can coincide with a training example.
    return build_examples(
        name_offset=5,
        function_offset=2,
        instruction_offset=1,
        per_operation=4,
        seed=seed,
        split="validation",
        id_prefix="V1",
    )


def curriculum_digest() -> str:
    payload = [
        (example.example_id, example.instruction, example.signature, example.body, example.operation)
        for example in build_training_examples()
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
