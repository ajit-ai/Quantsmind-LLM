"""Operation-aligned synthetic curriculum (Taksh v2.1.0).

Every example pairs a **natural-language instruction** with a Python
**signature** and a **body** that performs exactly one operation. The
instruction is what the model is trained to condition on, and the body is
what it is trained to produce.

Thirteen operations across five families, each with a marker that makes the
operation machine-checkable in a generated program:

arithmetic  add, subtract, multiply
conditioning maximum, minimum
collection  keep_positive, keep_negative
text        upper, lower
aggregate   count_values, sum_values, lookup, count_keys

Partitions
----------
``T``   training examples (operation balanced, small)
``V``   same operations, unseen function and parameter names
``H1``  unseen natural-language paraphrases of the trained operations
``H2``  unseen compositions of trained operations
``H3``  operation families excluded from training
``C``   matched instruction pairs differing in exactly one requirement
"""

from __future__ import annotations

import hashlib
import random

from training.supervised import build_example

#: operation -> (family, instruction, signature, body, markers)
OPERATIONS = {
    "add": (
        "arithmetic", "Write a function that returns the sum of two integers.",
        "def add_values(left, right):", "    return left + right", [" + "]),
    "subtract": (
        "arithmetic",
        "Write a function that returns the difference between two integers.",
        "def subtract_values(left, right):", "    return left - right", [" - "]),
    "multiply": (
        "arithmetic", "Write a function that returns the product of two integers.",
        "def multiply_values(left, right):", "    return left * right", [" * "]),
    "maximum": (
        "conditioning", "Write a function that returns the larger of two values.",
        "def larger_value(left, right):",
        "    if left > right:\n        return left\n    return right", [">"]),
    "minimum": (
        "conditioning", "Write a function that returns the smaller of two values.",
        "def smaller_value(left, right):",
        "    if left < right:\n        return left\n    return right", ["<"]),
    "keep_positive": (
        "collection", "Write a function that returns only the positive numbers.",
        "def positive_values(values):",
        "    return [value for value in values if value > 0]", ["> 0"]),
    "keep_negative": (
        "collection", "Write a function that returns only the negative numbers.",
        "def negative_values(values):",
        "    return [value for value in values if value < 0]", ["< 0"]),
    "upper": (
        "text", "Write a function that converts a string to upper case.",
        "def upper_text(text):", "    return text.upper()", [".upper("]),
    "lower": (
        "text", "Write a function that converts a string to lower case.",
        "def lower_text(text):", "    return text.lower()", [".lower("]),
    "count_values": (
        "aggregate", "Write a function that counts how many values a list holds.",
        "def count_values_list(values):",
        "    count = 0\n    for value in values:\n        count += 1\n    return count",
        ["count += 1"]),
    "sum_values": (
        "aggregate", "Write a function that sums the values in a list.",
        "def sum_values_list(values):",
        "    total = 0\n    for value in values:\n        total += value\n    return total",
        ["total += value"]),
    "lookup": (
        "mapping", "Write a function that returns the value stored under a key.",
        "def lookup_value(mapping, key):", "    return mapping.get(key, 0)", [".get("]),
    "count_keys": (
        "mapping", "Write a function that returns how many keys a mapping holds.",
        "def count_keys_mapping(mapping, key):", "    return len(mapping)", ["len("]),
}

#: Unseen paraphrases used by the H1 partition.
PARAPHRASES = {
    "add": "Implement a routine that adds two numbers together.",
    "subtract": "Implement a routine that takes one number away from another.",
    "multiply": "Implement a routine that multiplies two numbers together.",
    "maximum": "Implement a routine that picks the greater of two numbers.",
    "minimum": "Implement a routine that picks the lesser of two numbers.",
    "keep_positive": "Implement a routine that keeps only numbers above zero.",
    "keep_negative": "Implement a routine that keeps only numbers below zero.",
    "upper": "Implement a routine that shouts a string in capitals.",
    "lower": "Implement a routine that quiets a string in lower case.",
    "count_values": "Implement a routine that tallies the items in a list.",
    "sum_values": "Implement a routine that totals the numbers in a list.",
    "lookup": "Implement a routine that reads a value out of a mapping.",
    "count_keys": "Implement a routine that tallies the entries of a mapping.",
}

#: Families withheld from training and evaluated in H3.
HELD_OUT_FAMILIES = ("mapping",)

#: Families trained and evaluated elsewhere.
TRAINED_FAMILIES = ("arithmetic", "conditioning", "collection", "text", "aggregate")

#: Unseen names used by the V partition.
ALTERNATE_NAMES = {
    "add_values": "sum_two", "subtract_values": "diff_two", "multiply_values": "times_two",
    "larger_value": "max_of_two", "smaller_value": "min_of_two",
    "positive_values": "keep_above_zero", "negative_values": "keep_below_zero",
    "upper_text": "capitalise", "lower_text": "small_case",
    "count_values_list": "how_many", "sum_values_list": "total_of",
    "lookup_value": "read_key", "count_keys_mapping": "how_many_keys",
}

#: A third naming set, used by the contrastive partition so that its text is
#: distinct from both the training and the validation partitions.
CONTRASTIVE_NAMES = {
    "add_values": "sum_pair", "subtract_values": "diff_pair", "multiply_values": "times_pair",
    "larger_value": "greater_pair", "smaller_value": "lesser_pair",
    "positive_values": "above_zero", "negative_values": "below_zero",
    "upper_text": "caps", "lower_text": "small",
    "count_values_list": "number_of", "sum_values_list": "sum_of",
    "lookup_value": "fetch_key", "count_keys_mapping": "key_count",
}

NAME_SETS = {"base": {}, "alt": ALTERNATE_NAMES, "contrast": CONTRASTIVE_NAMES}


def operation_markers(operation: str) -> list:
    return list(OPERATIONS[operation][4])


def _example(
    case_id, operation, split, seed, *, instruction=None, name_style="base",
    suffix="",
):
    family, text, signature, body, markers = OPERATIONS[operation]
    if instruction is None:
        instruction = text
    for old, new in NAME_SETS[name_style].items():
        signature = signature.replace(old, new)
    if suffix:
        signature = signature.replace("def ", f"def {suffix}_", 1)
    example = build_example(
        example_id=case_id,
        instruction=instruction,
        signature=signature,
        body=body,
        operation=operation,
        family=family,
        split=split,
        seed=seed,
        metadata={"markers": list(markers)},
    )
    return example


def build_operation_curriculum(seed: int = 20261021) -> dict:
    """Build the frozen operation curriculum (T, V, H1, H2, H3, C)."""
    partitions = {"T": [], "V": [], "H1": [], "H2": [], "H3": [], "C": []}
    trained_ops = [
        operation
        for operation, (family, *_rest) in OPERATIONS.items()
        if family in TRAINED_FAMILIES
    ]
    held_out_ops = [
        operation
        for operation, (family, *_rest) in OPERATIONS.items()
        if family in HELD_OUT_FAMILIES
    ]

    # T: a small, operation-balanced training set.
    for index, operation in enumerate(trained_ops * 2):
        partitions["T"].append(
            _example(
                f"T-{operation}-{index:03d}", operation, "T", seed + index,
                suffix="b" if index >= len(trained_ops) else "",
            )
        )

    # V: the same operations with unseen function names.
    for index, operation in enumerate(trained_ops * 2):
        partitions["V"].append(
            _example(
                f"V-{operation}-{index:03d}", operation, "V", seed + index,
                name_style="alt"
            )
        )

    # H1: unseen paraphrases of the trained operations.
    for index, operation in enumerate(trained_ops * 2):
        partitions["H1"].append(
            _example(
                f"H1-{operation}-{index:03d}",
                operation,
                "H1",
                seed + index,
                instruction=PARAPHRASES[operation],
            )
        )

    # H2: unseen compositions of trained operations.
    for index in range(12):
        first, second = trained_ops[index % len(trained_ops)], trained_ops[
            (index + 1) % len(trained_ops)
        ]
        parts = []
        for operation in (first, second):
            family, text, signature, body, markers = OPERATIONS[operation]
            parts.append(body)
        body = "\n".join(parts)
        instruction = (
            f"Write a function that first applies {first} and then {second}."
        )
        signature = f"def composed_{index:02d}(left, right):"
        from training.supervised import build_example as _build

        partitions["H2"].append(
            _build(
                example_id=f"H2-composed-{index:03d}",
                instruction=instruction,
                signature=signature,
                body=body,
                operation="composition",
                family="composition",
                split="H2",
                seed=seed + index,
                metadata={"markers": [], "parts": [first, second]},
            )
        )

    # H3: operation families that are never trained.
    for index, operation in enumerate(held_out_ops * 4):
        partitions["H3"].append(
            _example(f"H3-{operation}-{index:03d}", operation, "H3", seed + index)
        )

    # C: matched instruction pairs differing in one requirement.
    pairs = [
        ("add", "subtract"), ("maximum", "minimum"),
        ("keep_positive", "keep_negative"), ("upper", "lower"),
        ("count_values", "sum_values"), ("lookup", "count_keys"),
    ]
    for index, (first, second) in enumerate(pairs):
        for position, operation in enumerate((first, second)):
            example = _example(
                f"C-{index:02d}-{operation}",
                operation,
                "C",
                seed + index,
                name_style="contrast",
            )
            example.metadata["pair_id"] = f"pair-{index:02d}"
            example.metadata["pair_role"] = "a" if position == 0 else "b"
            partitions["C"].append(example)
    return partitions


def curriculum_digest(partitions: dict) -> str:
    payload = "\n".join(
        f"{split}:{example.example_id}:{example.content_sha256}"
        for split in sorted(partitions)
        for example in partitions[split]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def contamination_report(partitions: dict) -> dict:
    """Cross-partition contamination for the operation curriculum."""
    violations = []
    seen = {}
    for split in sorted(partitions):
        for example in partitions[split]:
            key = example.content_sha256
            if key in seen and seen[key] != split:
                violations.append(
                    f"identical training text in {seen[key]} and {split}: "
                    f"{example.example_id}"
                )
            seen.setdefault(key, split)
    train_operations = {example.operation for example in partitions["T"]}
    for example in partitions["H3"]:
        if example.operation in train_operations:
            violations.append(
                f"{example.example_id}: withheld operation {example.operation} "
                "appears in the training partition"
            )
    for example in partitions["C"]:
        for other in partitions["C"]:
            if other is example:
                continue
            if (
                other.metadata.get("pair_id") == example.metadata.get("pair_id")
                and other.instruction == example.instruction
            ):
                violations.append(
                    f"{example.example_id}: contrastive pair shares an instruction"
                )
    return {
        "schema": "quantsmind.synthetic_contamination/1",
        "violations": violations,
        "violation_count": len(violations),
        "contaminated": bool(violations),
        "verdict": "SYNTHETIC_EVALUATION_CONTAMINATED" if violations else "CLEAN",
        "partition_sizes": {
            split: len(examples) for split, examples in sorted(partitions.items())
        },
        "note": (
            "The T and V partitions deliberately share the same operations and "
            "the same bodies with different names; the H1 partition shares the "
            "bodies with different instruction paraphrases. That is the design "
            "of the generalisation test, and it is reported here rather than "
            "counted as a leak."
        ),
    }
