"""Tests for M7 task-aware syntax scoring."""

from __future__ import annotations

from evaluation.m7_evaluate import (
    score_completion_syntax,
    score_generation_syntax,
)


def test_generation_accepts_a_real_function():
    result = score_generation_syntax(
        "def add_one(x):\n    return x + 1\n", "add_one")
    assert result["status"] == "SYNTAX_VALID"
    assert result["numerator"] == 1


def test_generation_rejects_prose_even_if_it_is_parseable():
    result = score_generation_syntax("\n".join(["Text"] * 40))
    assert result["status"] != "SYNTAX_VALID"
    assert result["numerator"] == 0


def test_generation_rejects_the_wrong_function_name():
    result = score_generation_syntax(
        "def other(x):\n    return x\n", "expected_name")
    assert result["status"] == "WRONG_FUNCTION_NAME"


def test_generation_rejects_a_docstring_only_body():
    result = score_generation_syntax(
        'def add_one(x):\n    """Add one."""\n', "add_one")
    assert result["status"] == "EMPTY_FUNCTION"


def test_completion_accepts_code_shape():
    result = score_completion_syntax("value = value + 1\n")
    assert result["status"] == "SYNTAX_VALID"


def test_completion_rejects_repeated_prose():
    result = score_completion_syntax("\n".join(["Text"] * 40))
    assert result["status"] == "NO_CODE_SHAPE"
    assert result["numerator"] == 0
