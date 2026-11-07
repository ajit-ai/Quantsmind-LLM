"""TAKSH v1.3.2-D: general longer-of-two (binary length comparison) rule.

A "longer of two" description compares the lengths of two arguments; it is
not a collection filter over entries (the previous long-word threshold rule
misapplied to it). Tests use novel identifiers and phrasings.
"""

from __future__ import annotations

from evaluation.functional_eval import evaluate_functional
from evaluation.operation_planner import compile_plan, plan, validate
from evaluation.operation_synth import synthesize


def _run(instruction, signature, fn, tests):
    result = synthesize(instruction, signature)
    assert result["action"] == "synthesized", (instruction, result)
    assert result["operation"] == "longer_string", result
    outcome = evaluate_functional(signature + "\n" + result["body"] + "\n", fn, tests)
    assert outcome.passed, (instruction, result, outcome.reason)


def test_longer_of_two_strings_novel():
    _run("Return the longer of two strings.", "def pick_zx(word_a, word_b):", "pick_zx",
         [{"args": ["hello", "hi"], "kwargs": {}, "expected": "hello"},
          {"args": ["hi", "hello"], "kwargs": {}, "expected": "hello"},
          {"args": ["same", "same2"], "kwargs": {}, "expected": "same2"}])


def test_longer_of_two_alternate_phrasing():
    _run("Give the longer text of the pair.", "def pick_zx(text_x, text_y):", "pick_zx",
         [{"args": ["ab", "abcd"], "kwargs": {}, "expected": "abcd"},
          {"args": ["abcd", "ab"], "kwargs": {}, "expected": "abcd"}])


def test_collection_long_word_filter_still_works():
    # Regression protection: single-collection "long" filter remains a filter.
    result = synthesize("Return the words that are longer than three characters.", "def keep_zx(words_z):")
    assert result["action"] == "synthesized", result


def test_unsupported_still_abstains():
    result = synthesize("Synchronously hash the harbor entry.", "def calc_zx(value_q):")
    assert result["action"] == "abstain"


def test_planner_longer_string_valid_and_compiles():
    result = plan("Return the longer of two strings.", "def pick_zx(word_a, word_b):")
    assert result.status == "planned"
    assert validate(result, "def pick_zx(word_a, word_b):")
    compiled = compile_plan(result, "def pick_zx(word_a, word_b):")
    assert compiled["action"] == "planned" and compiled.get("provenance") == "planner"
