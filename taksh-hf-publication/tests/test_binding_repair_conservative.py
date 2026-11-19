"""Development tests for the conservative v2.5.0 binding repair."""

from __future__ import annotations

from training.binding_repair import (
    extract_signature,
    repair_completion,
    repair_completion_info,
)


def test_single_unique_unbound_name_is_repaired():
    assert repair_completion("def convert(first):", "    return value + 1\n") == "    return first + 1\n"


def test_existing_requested_binding_is_unchanged():
    body = "    return first + second\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_multiple_plausible_mappings_are_unchanged():
    body = "    return left + right\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_one_unbound_with_multiple_unused_candidates_is_unchanged():
    body = "    return fallback\n"
    assert repair_completion("def choose(first, second, third):", body) == body


def test_multiple_unbound_with_one_unused_parameter_is_unchanged():
    body = "    return left + right\n"
    assert repair_completion("def only_first(first):", body) == body


def test_nested_function_is_not_repaired():
    body = "    def helper(values):\n        return len(values)\n    return value\n"
    assert repair_completion("def outer(data):", body) == body


def test_lambda_is_not_repaired():
    body = "    return (lambda item: item)(value)\n"
    assert repair_completion("def apply(data):", body) == body


def test_comprehension_shadowing_is_safe_but_restricted():
    body = "    return [item for item in values if item > 0]\n"
    assert repair_completion("def collect(data):", body) == "    return [item for item in data if item > 0]\n"


def test_loop_variable_is_not_renamed():
    body = "    for value in values:\n        return value\n"
    assert repair_completion("def pick(data):", body) == "    for value in data:\n        return value\n"


def test_malformed_source_is_unchanged():
    body = "    return (value +\n"
    assert repair_completion("def broken(data):", body) == body


def test_negative_control_with_external_call_is_unchanged():
    body = "    return external_value(value)\n"
    assert repair_completion("def use_api(data):", body) == body


def test_unknown_dunder_attribute_is_unchanged():
    body = "    return value.__class__\n"
    assert repair_completion("def inspect(data):", body) == body


def test_whole_function_is_not_repaired():
    body = "def solve(left, right):\n    return left + right\n"
    assert repair_completion("def compute_total(first, second):", body) == body


def test_diagnostics_explain_ambiguity():
    info = repair_completion_info("def pair(first, second):", "    return left + right\n")
    assert info.status == "abstained_ambiguous"
    assert info.mapping == {}
    assert info.unbound == ["left", "right"]
    assert info.unused_params == ["first", "second"]


def test_diagnostics_explain_unique_mapping():
    info = repair_completion_info("def convert(first):", "    return value + 1\n")
    assert info.status == "repaired"
    assert info.mapping == {"value": "first"}


def test_already_bound_output_with_unused_parameters_is_unchanged():
    body = "    return first + 1\n"
    assert repair_completion("def choose(first, second, third):", body) == body
