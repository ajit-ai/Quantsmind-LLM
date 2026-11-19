"""Targeted generalization curriculum (Taksh v1.2.0 Increment 7).

Addresses the Increment 6 failure evidence: both checkpoints scored 0/48
on novel-identifier variants of the eight functional behaviours because
training names were fixed pools (`text`, `left`, `right`, `values`,
`items`, ...). This module reuses the Increment 4 behaviours and surface
variants but with:

* **novel identifier pools** — 48 generated names per arity, disjoint from
  every Increment 2/4 pool and from the fixed names above;
* **wider prompt variation** — six paraphrases per behaviour, split so
  training and validation wording sets are disjoint;
* **semantic separation** — validation uses a disjoint identifier half and
  a disjoint paraphrase half (not a shuffle of training examples);
* **retention** — the Increment 2 binding curriculum (arithmetic,
  maximum/minimum, binding probes) is composed alongside this module at
  the experiment level, so AD-1-style operations keep coverage.

All examples are original synthetic data. Every example carries verifiable
input/output metadata and must pass the safe functional gate.
"""

from __future__ import annotations

import hashlib

from training.binding_curriculum import _validate
from training.functional_curriculum import (
    OPERATIONS,
    _FAMILIES,
    _INSTRUCTIONS,
    _RELATIONSHIPS,
    _SAMPLES,
    _body,
)
from training.supervised import build_example

_STEMS = (
    "probe", "node", "case", "entry", "vertex", "segment",
    "unit", "slot", "cell", "point", "vertex", "node",
)

# Two extra paraphrases per behaviour (six total with the Increment 4 set).
_EXTRA_INSTRUCTIONS = {
    "scale_value": ("Multiply these two quantities and report the product.", "Return value times factor."),
    "join_text": ("Combine the strings with one space between them.", "Return the joined pair of words."),
    "add_then_multiply": ("Double the combined total of the pair.", "Add first, then double the sum."),
    "multiply_then_add": ("Increase the product by two.", "Take the product and add two to it."),
    "filter_and_count": ("How many items sit above zero?", "Give the count of positive entries."),
    "keep_long_words": ("Keep entries whose length exceeds three.", "Return words longer than three characters."),
    "is_even": ("Determine whether the integer is even.", "Even parity check, please."),
    "sum_positive": ("What is the total of the positive entries?", "Add only the values above zero."),
}

_BINARY_OPERATIONS = ("scale_value", "join_text", "add_then_multiply", "multiply_then_add")
_SINGLE_OPERATIONS = ("filter_and_count", "keep_long_words", "is_even", "sum_positive")


def novel_pairs(count: int = 48) -> list[tuple[str, str]]:
    """48 unique novel parameter-name pairs, disjoint from all prior pools."""
    pairs = []
    for index in range(count):
        first = f"{_STEMS[index % len(_STEMS)]}_{index:02d}"
        second = f"{_STEMS[(index + 5) % len(_STEMS)]}_{index:02d}"
        pairs.append((first, second))
    return pairs


def novel_singles(count: int = 48) -> list[str]:
    return [f"{_STEMS[index % len(_STEMS)]}_{index:02d}" for index in range(count)]


def build_examples(
    *,
    per_operation: int = 16,
    seed: int = 20261108,
    split: str = "train",
    id_prefix: str = "T1",
) -> list:
    is_train = split == "train"
    # Semantic separation: train uses identifier half 0 (indices 0..23) and
    # paraphrases 0..3; validation uses half 1 (24..47) and paraphrases 4..5.
    pair_pool = novel_pairs(48)
    single_pool = novel_singles(48)
    name_start = 0 if is_train else 24
    examples = []
    for operation in OPERATIONS:
        instructions = list(_INSTRUCTIONS[operation]) + list(_EXTRA_INSTRUCTIONS[operation])
        active_instructions = instructions[0:4] if is_train else instructions[4:6]
        samples = _SAMPLES[operation]
        for index in range(per_operation):
            pool_index = name_start + index
            if operation in _BINARY_OPERATIONS:
                params = pair_pool[pool_index]
            else:
                params = (single_pool[pool_index],)
            function_name = f"op_{operation}_{index:02d}"
            instruction = active_instructions[index % len(active_instructions)]
            variant = index % 4
            signature = f"def {function_name}({', '.join(params)}):"
            body = _body(operation, params, variant)
            _validate(signature, body)
            sample_index = index % len(samples) if is_train else (index + 2) % len(samples)
            args, expected = samples[sample_index]
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
                        "targeted": True,
                        "variant": variant,
                        "surface_variant": f"v{variant}",
                        "inputs": [list(args)],
                        "expected_outputs": [expected],
                        "relationship": _RELATIONSHIPS[operation],
                    },
                )
            )
    return examples


def build_training_examples(*, seed: int = 20261108) -> list:
    return build_examples(per_operation=16, seed=seed, split="train")


def build_validation_examples(*, seed: int = 20261109) -> list:
    return build_examples(per_operation=4, seed=seed, split="validation", id_prefix="T1V")


def targeted_digest() -> str:
    payload = [
        (example.example_id, example.instruction, example.signature, example.body, example.operation)
        for example in build_training_examples()
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
