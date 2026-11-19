"""Unit tests for the deterministic v1.2.0 binding curriculum."""

from __future__ import annotations

import ast

from training.binding_curriculum import (
    TRAINED_OPERATIONS,
    build_training_examples,
    build_validation_examples,
    curriculum_digest,
)
from training.program_analysis import analyse


def _valid(example) -> bool:
    source = example.signature + "\n" + example.body
    tree = ast.parse(source)
    return isinstance(tree.body[0], ast.FunctionDef)


def test_training_curriculum_is_valid_and_deterministic():
    first = build_training_examples()
    second = build_training_examples()
    assert [example.example_id for example in first] == [example.example_id for example in second]
    assert curriculum_digest() == curriculum_digest()
    assert all(_valid(example) for example in first)
    assert {example.operation for example in first} == set(TRAINED_OPERATIONS)


def test_binding_examples_use_requested_parameter_names():
    examples = build_training_examples()
    bad = []
    for example in examples:
        result = analyse(example.signature, example.body)
        if not result["parses"] or result["undefined_names"]:
            bad.append(example.example_id)
    assert bad == []


def test_validation_uses_different_examples():
    training = {(example.instruction, example.signature) for example in build_training_examples()}
    validation = {(example.instruction, example.signature) for example in build_validation_examples()}
    assert training.isdisjoint(validation)
    assert len(build_training_examples()) == 13 * 12
    assert len(build_validation_examples()) == 13 * 4
