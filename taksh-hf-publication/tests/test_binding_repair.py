"""Compatibility tests for the conservative binding-repair API."""

from __future__ import annotations

from training.binding_repair import extract_signature, repair_completion, repair_from_prompt


def test_single_unique_binding_is_repaired():
    assert repair_completion("def convert(first):", "    return value + 1\n") == "    return first + 1\n"


def test_requested_bindings_are_preserved():
    body = "    return first + second\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_comprehension_target_is_not_renamed():
    body = "    return [item for item in values if item > 0]\n"
    assert repair_completion("def collect(data):", body) == "    return [item for item in data if item > 0]\n"


def test_loop_variable_is_not_renamed():
    body = "    for item in values:\n        return item\n"
    assert repair_completion("def pick(data):", body) == "    for item in data:\n        return item\n"


def test_ambiguous_mapping_is_unchanged():
    body = "    return left + right\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_prefix_noise_and_prompt_signature_are_handled():
    prompt = "Write a function.\ndef convert(first):\n"
    assert extract_signature(prompt) == "def convert(first):"
    assert repair_from_prompt(prompt, "    return value + 1\n") == "    return first + 1\n"


def test_invalid_body_is_returned_unchanged():
    body = "    return (value +\n"
    assert repair_completion("def broken(first):", body) == body


def test_whole_function_is_unchanged():
    body = "def solve(left, right):\n    return left + right\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_external_call_is_not_repaired():
    body = "    return external_value(value)\n"
    assert repair_completion("def run(first):", body) == body
