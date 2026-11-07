"""Tests for M4 code-quality checks and the fixed evaluation set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.code_check import (
    EXTRACTION_AMBIGUOUS,
    NO_CODE,
    SAFE_TESTS,
    SYNTAX_INVALID,
    SYNTAX_VALID,
    TEST_FAIL,
    TEST_PASS,
    check_format_compliance,
    evaluate_generation,
    extract_code,
    summarise,
)
from evaluation.m4_eval_set import FROZEN, PROMPTS, SET_ID, prompt_hash


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_single_fenced_block_is_extracted():
    code, status = extract_code("```python\ndef f():\n    return 1\n```")
    assert "def f" in code
    assert status == "SINGLE_BLOCK"


def test_multiple_blocks_are_ambiguous_not_guessed():
    code, status = extract_code("```python\na=1\n```\n```python\nb=2\n```")
    assert code is None
    assert status == EXTRACTION_AMBIGUOUS


def test_empty_output_is_no_code():
    assert extract_code("")[1] == NO_CODE
    assert extract_code("   ")[1] == NO_CODE


def test_comments_only_output_is_not_counted_as_code():
    """A block of '#' lines parses as Python but contains no solution.

    Scoring it SYNTAX_VALID would let noise look like a working model.
    """
    noise = "# The default is used to the same value.\n# more\n#\n#"
    code, status = extract_code(noise)
    assert code is None
    assert status == NO_CODE


def test_docstring_only_output_is_not_counted_as_code():
    code, status = extract_code('"""just prose"""')
    assert code is None
    assert status == NO_CODE


def test_bare_real_code_is_accepted():
    code, status = extract_code("def f():\n    return 1")
    assert status == "BARE_CODE"
    assert "def f" in code


# --------------------------------------------------------------------------- #
# Syntax / test separation
# --------------------------------------------------------------------------- #
def test_valid_code_reports_syntax_valid():
    result = evaluate_generation(
        "id", "prompt", "```python\ndef reverse_string(s):\n    return s[::-1]\n```",
        SAFE_TESTS["reverse_string"])
    assert result["syntax_status"] == SYNTAX_VALID
    assert result["test_status"] == TEST_PASS


def test_syntax_valid_but_wrong_logic_is_a_test_failure():
    """The two dimensions must not be collapsed into one score."""
    result = evaluate_generation(
        "id", "prompt", "```python\ndef reverse_string(s):\n    return s\n```",
        SAFE_TESTS["reverse_string"])
    assert result["syntax_status"] == SYNTAX_VALID
    assert result["test_status"] == TEST_FAIL


def test_unparsable_code_is_not_tested():
    result = evaluate_generation("id", "prompt", "```python\ndef f(:\n```",
                                 SAFE_TESTS["reverse_string"])
    assert result["syntax_status"] == SYNTAX_INVALID
    assert result["test_status"] == "NOT_TESTABLE"


def test_authored_tests_actually_pass_on_reference_solutions():
    """A broken test would make every future comparison meaningless."""
    reference = {
        "reverse_string": "def reverse_string(s):\n    return s[::-1]",
        "is_palindrome": "def is_palindrome(s):\n    return s.lower() == s.lower()[::-1]",
        "factorial": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
        "sum_list": "def sum_list(xs):\n    return sum(xs)",
        "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "flatten": "def flatten(xs):\n    out = []\n    for x in xs:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out",
        "second_largest": "def second_largest(xs):\n    u = sorted(set(xs), reverse=True)\n    return u[1] if len(u) > 1 else None",
        "fizzbuzz_n": ("def fizzbuzz_n(n):\n    out = []\n"
                       "    for i in range(1, n + 1):\n"
                       "        out.append(\n"
                       "            'FizzBuzz' if i % 15 == 0 else\n"
                       "            'Fizz' if i % 3 == 0 else\n"
                       "            'Buzz' if i % 5 == 0 else str(i))\n"
                       "    return out"),
    }
    for name, test_source in SAFE_TESTS.items():
        result = evaluate_generation(
            name, "prompt", f"```python\n{reference[name]}\n```", test_source)
        assert result["test_status"] == TEST_PASS, f"{name}: {result['test_detail']}"


# --------------------------------------------------------------------------- #
# Instruction compliance
# --------------------------------------------------------------------------- #
def test_exact_word_compliance():
    ok, _ = check_format_compliance("Reply with exactly the word READY and nothing else.", "READY")
    assert ok
    ok, _ = check_format_compliance("Reply with exactly the word READY and nothing else.", "maybe")
    assert not ok


def test_bullet_count_compliance():
    ok, _ = check_format_compliance(
        "List exactly three items as a Markdown bullet list and nothing else.",
        "- a\n- b\n- c")
    assert ok
    ok, _ = check_format_compliance(
        "List exactly three items as a Markdown bullet list and nothing else.",
        "- a\n- b")
    assert not ok


def test_json_compliance():
    ok, _ = check_format_compliance(
        "Return a JSON object. Return only the JSON, with no explanation and no "
        "markdown fences.", '{"a": 1}')
    assert ok
    ok, _ = check_format_compliance(
        "Return a JSON object. Return only the JSON, with no explanation and no "
        "markdown fences.", "not json")
    assert not ok


def test_prompt_asking_for_a_fence_is_not_penalised_for_one():
    ok, detail = check_format_compliance(
        "Write a function. Output only a single fenced Python code block.",
        "```python\ndef f():\n    return 1\n```")
    assert ok, detail


def test_prompt_without_a_machine_rule_is_not_scored_as_failure():
    ok, detail = check_format_compliance("Explain closures.", "anything at all")
    assert ok
    assert "no machine-checkable" in detail


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summary_keeps_dimensions_separate():
    results = [
        evaluate_generation("a", "p", "```python\ndef f():\n    return 1\n```"),
        evaluate_generation("b", "p", "# nothing"),
    ]
    summary = summarise(results)
    assert summary["items"] == 2
    assert summary["syntax_valid"] == 1
    assert summary["test_pass_rate"] is None  # nothing was testable
    assert "composite" not in summary


# --------------------------------------------------------------------------- #
# The frozen evaluation set
# --------------------------------------------------------------------------- #
def test_eval_set_is_declared_frozen():
    assert FROZEN is True
    assert SET_ID == "m4-eval-v1"


def test_shipped_eval_set_matches_the_authored_prompts():
    """The file on disk must equal the prompts in the module."""
    path = Path("evaluation/m4_eval_set.jsonl")
    if not path.exists():
        pytest.skip("M4 evaluation data is not part of the source release")
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == len(PROMPTS)
    assert [r["id"] for r in rows] == [p["id"] for p in PROMPTS]


def test_eval_set_hash_is_recorded_and_stable():
    meta = json.loads(
        Path("evaluation/m4_eval_set.meta.json").read_text(encoding="utf-8"))
    assert meta["prompt_hash"] == prompt_hash()
    assert meta["frozen"] is True


def test_eval_set_prompt_ids_are_unique():
    ids = [p["id"] for p in PROMPTS]
    assert len(ids) == len(set(ids))


def test_every_testable_prompt_names_a_real_test():
    for prompt in PROMPTS:
        if prompt["test"]:
            assert prompt["test"] in SAFE_TESTS, prompt["id"]


def test_eval_set_does_not_overlap_the_training_data():
    """A prompt copied from training data would not be a held-out measurement."""
    train_path = Path("data/instruct_v1.train.jsonl")
    if not train_path.exists():
        pytest.skip("M4 instruction data is not part of the source release")
    train = [json.loads(l) for l in
             train_path.read_text(
                 encoding="utf-8").splitlines() if l.strip()]
    train_instructions = {r["instruction"].strip().lower() for r in train}
    for prompt in PROMPTS:
        assert prompt["prompt"].strip().lower() not in train_instructions, \
            prompt["id"]