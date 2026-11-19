"""Staged identifier-copying curriculum (Taksh v1.2.0 Increment 9, WP3+WP4).

Seven stages from exact copying to functional generation, all original
synthetic data with novel identifier pools. Stage families use the
Increment 2/4 surface-variant machinery where practical so template
diversity and quality gates carry over. Validation is disjoint in
identifiers and wording (never a shuffle of training examples).
"""

from __future__ import annotations

import hashlib

from training.binding_curriculum import _validate
from training.functional_curriculum import _INSTRUCTIONS, _SAMPLES, _body
from training.supervised import build_example
from training.targeted_curriculum import novel_pairs, novel_singles

STAGES = (
    "exact_copy",
    "unfamiliar_copy",
    "substitution",
    "propagation",
    "multi_binding",
    "binding_structure",
    "functional",
)

_DIVERSE_INSTRUCTIONS = {
    "exact_copy": ("Return the named value.", "Give back the input.", "Yield the value as-is."),
    "unfamiliar_copy": ("Copy this identifier into the result.", "Pass the named input through."),
    "substitution": ("Rename the input and return it.", "Copy with a new local name."),
    "propagation": ("Keep the value in a local and return it.", "Store then return the input."),
    "multi_binding": ("Combine both values.", "Return the joint result."),
    "binding_structure": ("Choose between the values.", "Pick the larger value."),
    "functional": ("Carry out the described operation.", "Perform the named task."),
}


def _stage_examples(stage: str, *, split: str, id_prefix: str, per_stage: int) -> list:
    is_train = split == "train"
    pool_start = 0 if is_train else 24
    pairs = novel_pairs(48)
    singles = novel_singles(48)
    examples = []
    if stage in ("exact_copy", "unfamiliar_copy"):
        instruction_pool = _DIVERSE_INSTRUCTIONS[stage]
        for index in range(per_stage):
            name = singles[pool_start + (index % 24)]
            signature = f"def copy_{index:02d}({name}):"
            body = f"    return {name}"
            if stage == "unfamiliar_copy":
                body = f"    result = {name}\n    return result"
            instruction = instruction_pool[index % len(instruction_pool)]
            _validate(signature, body)
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=stage,
                    family="copy",
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [stage], "stage": stage,
                              "inputs": [[0]], "expected_outputs": [0],
                              "relationship": "output == the named input"},
                )
            )
        return examples
    if stage == "substitution":
        for index in range(per_stage):
            source, target_name = pair_pool = (pairs[pool_start + (index % 24)][0], singles[pool_start + (index % 24)])
            signature = f"def rename_{index:02d}({source}):"
            body = f"    {target_name} = {source}\n    return {target_name}"
            instruction = _DIVERSE_INSTRUCTIONS[stage][index % len(_DIVERSE_INSTRUCTIONS[stage])]
            _validate(signature, body)
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=stage,
                    family="copy",
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [stage], "stage": stage,
                              "inputs": [[0]], "expected_outputs": [0],
                              "relationship": "output == input under a new local name"},
                )
            )
        return examples
    if stage == "propagation":
        for index in range(per_stage):
            a = pairs[pool_start + (index % 24)][0]
            local = singles[pool_start + (index % 24)]
            signature = f"def carry_{index:02d}({a}):"
            body = f"    {local} = {a}\n    {local} = {local}\n    return {local}"
            instruction = _DIVERSE_INSTRUCTIONS[stage][index % len(_DIVERSE_INSTRUCTIONS[stage])]
            _validate(signature, body)
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=stage,
                    family="copy",
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [stage], "stage": stage,
                              "inputs": [[0]], "expected_outputs": [0],
                              "relationship": "output == input propagated through locals"},
                )
            )
        return examples
    if stage == "multi_binding":
        for index in range(per_stage):
            a, b = pairs[pool_start + (index % 24)]
            signature = f"def both_{index:02d}({a}, {b}):"
            body = f"    return {a} + {b}"
            instruction = _DIVERSE_INSTRUCTIONS[stage][index % len(_DIVERSE_INSTRUCTIONS[stage])]
            _validate(signature, body)
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=stage,
                    family="copy",
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [stage], "stage": stage,
                              "inputs": [[1, 2]], "expected_outputs": [3],
                              "relationship": "output == first + second"},
                )
            )
        return examples
    if stage == "binding_structure":
        for index in range(per_stage):
            a, b = pairs[pool_start + (index % 24)]
            signature = f"def choose_{index:02d}({a}, {b}):"
            body = f"    if {a} > {b}:\n        return {a}\n    return {b}"
            instruction = _DIVERSE_INSTRUCTIONS[stage][index % len(_DIVERSE_INSTRUCTIONS[stage])]
            _validate(signature, body)
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=stage,
                    family="copy",
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [stage], "stage": stage,
                              "inputs": [[3, 2]], "expected_outputs": [3],
                              "relationship": "output == the larger input"},
                )
            )
        return examples
    # functional stage: the eight established behaviours with novel pools.
    from training.functional_curriculum import OPERATIONS, _FAMILIES, _RELATIONSHIPS

    singles_pool = novel_singles(48)
    pairs_pool = novel_pairs(48)
    for operation in OPERATIONS:
        instructions = list(_INSTRUCTIONS[operation])
        samples = _SAMPLES[operation]
        for index in range(per_stage // len(OPERATIONS)):
            if operation in ("scale_value", "join_text", "add_then_multiply", "multiply_then_add"):
                params = pairs_pool[pool_start + (index % 24)]
            else:
                params = (singles_pool[pool_start + (index % 24)],)
            function_name = f"fn_{operation}_{index:02d}"
            instruction = instructions[(index + (0 if is_train else 2)) % len(instructions)]
            variant = index % 4
            signature = f"def {function_name}({', '.join(params)}):"
            body = _body(operation, params, variant)
            _validate(signature, body)
            args, expected = samples[index % len(samples)]
            examples.append(
                build_example(
                    example_id=f"{id_prefix}-{stage}-{operation}-{index:03d}",
                    instruction=instruction,
                    signature=signature,
                    body=body,
                    operation=operation,
                    family=_FAMILIES[operation],
                    split=split,
                    seed=20261109 + index,
                    metadata={"markers": [operation], "stage": stage,
                              "variant": variant,
                              "inputs": [list(args)], "expected_outputs": [expected],
                              "relationship": _RELATIONSHIPS[operation]},
                )
            )
    return examples


def build_training_examples(*, per_stage: int = 16) -> list:
    examples = []
    for stage in STAGES:
        examples.extend(_stage_examples(stage, split="train", id_prefix="C1", per_stage=per_stage))
    return examples


def build_validation_examples(*, per_stage: int = 4) -> list:
    examples = []
    for stage in STAGES:
        examples.extend(_stage_examples(stage, split="validation", id_prefix="C1V", per_stage=per_stage))
    return examples


def copy_curriculum_digest() -> str:
    payload = [
        (example.example_id, example.instruction, example.signature, example.body, example.operation)
        for example in build_training_examples()
    ]
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
