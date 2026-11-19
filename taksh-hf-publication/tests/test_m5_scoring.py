"""Tests for M5 functional scoring and aggregation."""

from __future__ import annotations

from evaluation.m5_scorers import (
    TEST_FAIL,
    TEST_PASS,
    TEST_TIMEOUT,
    aggregate,
    run_functional_test,
    score_compliance,
    score_example,
    score_functional,
    score_structured,
)

FENCE = chr(96) * 3

REFERENCE = {
    "reverse_string": "def reverse_string(s):\n    return s[::-1]",
    "is_palindrome": "def is_palindrome(s):\n    return s.lower() == s.lower()[::-1]",
    "factorial": "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
    "sum_list": "def sum_list(xs):\n    return sum(xs)",
    "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
    "flatten": "def flatten(xs):\n    out = []\n    for x in xs:\n        out.extend(flatten(x) if isinstance(x, list) else [x])\n    return out",
    "second_largest": "def second_largest(xs):\n    u = sorted(set(xs), reverse=True)\n    return u[1] if len(u) > 1 else None",
    "fizzbuzz_n": "def fizzbuzz_n(n):\n    out = []\n    for i in range(1, n + 1):\n        out.append('FizzBuzz' if i % 15 == 0 else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else str(i))\n    return out",
}


def fenced(code):
    return f"{FENCE}python\n{code}\n{FENCE}"


def test_functional_pass_and_fail():
    assert score_functional(fenced(REFERENCE["reverse_string"]),
                            "reverse_string")["status"] == TEST_PASS
    wrong = "def reverse_string(s):\n    return s"
    assert score_functional(fenced(wrong), "reverse_string")["status"] == TEST_FAIL


def test_syntax_valid_but_wrong_is_not_a_pass():
    wrong = "def reverse_string(s):\n    return s"
    result = score_functional(fenced(wrong), "reverse_string")
    assert result["score"] == 0.0
    assert result["numerator"] == 0


def test_task_without_an_oracle_is_excluded_not_failed():
    result = score_functional("anything at all", None)
    assert result["score"] is None
    assert result["denominator"] == 0
    assert "excluded" in result["detail"]


def test_every_authored_oracle_passes_on_its_reference_solution():
    for name, code in REFERENCE.items():
        result = score_functional(fenced(code), name)
        assert result["status"] == TEST_PASS, f"{name}: {result['detail']}"


def test_functional_timeout_is_reported():
    result = run_functional_test("import time\ntime.sleep(30)", "pass",
                                 timeout=2)
    assert result["status"] == TEST_TIMEOUT


def test_functional_not_run_when_there_is_no_code():
    result = score_functional("# only a comment", "reverse_string")
    assert result["denominator"] == 0
    assert "not run" in result["detail"]


def test_compliance_counts_each_rule():
    result = score_compliance(
        "- a\n- b\n- c",
        [{"type": "max_lines", "value": 3},
         {"type": "line_starts_with", "value": "-"}])
    assert result["numerator"] == 2
    assert result["denominator"] == 2
    assert len(result["checks"]) == 2


def test_compliance_without_a_checklist_is_unscorable():
    result = score_compliance("anything", [])
    assert result["score"] is None
    assert result["denominator"] == 0


def test_structured_parse_and_schema_failures_are_distinct():
    good = score_structured('{"a": 1}', {"type": "json_keys", "keys": ["a"]})
    missing = score_structured('{"b": 2}', {"type": "json_keys", "keys": ["a"]})
    unparsable = score_structured("not json", {"type": "json_keys",
                                              "keys": ["a"]})
    assert good["status"] == "SCHEMA_VALID"
    assert missing["status"] == "SCHEMA_INVALID"
    assert "missing keys" in missing["detail"]
    assert "parse failure" in unparsable["detail"]


def test_aggregate_reports_explicit_denominators():
    metrics = [score_example(
        {"task_type": "code_generation", "test_specification": "reverse_string"},
        fenced(REFERENCE["reverse_string"]))]
    summary = aggregate(metrics)
    assert summary["functional"]["numerator"] == 1
    assert summary["functional"]["denominator"] == 1
    assert summary["functional"]["rate"] == 1.0


def test_human_review_is_never_counted_as_scored():
    summary = aggregate([score_example(
        {"task_type": "software_architecture"}, "anything")])
    assert summary["human_review"]["denominator"] == 0
    assert summary["human_review"]["rate"] is None


def test_zero_denominator_yields_no_score():
    summary = aggregate([{"syntax": {"metric": "syntax", "status": "not_checked",
                                     "score": None, "denominator": 0,
                                     "numerator": 0, "detail": ""}}])
    assert summary["syntax"]["rate"] is None
    assert "no score is defined" in summary["syntax"]["note"]


def test_small_denominator_is_marked_preliminary():
    summary = aggregate([score_example(
        {"task_type": "code_generation", "test_specification": "reverse_string"},
        fenced(REFERENCE["reverse_string"]))])
    assert summary["functional"]["preliminary"] is True
    assert "preliminary" in summary["functional"]["note"]


def test_aggregate_matches_the_raw_records():
    metrics = [
        score_example({"task_type": "code_generation",
                       "test_specification": "reverse_string"},
                      fenced(REFERENCE["reverse_string"])),
        score_example({"task_type": "code_generation",
                       "test_specification": "reverse_string"},
                      fenced("def reverse_string(s):\n    return s")),
    ]
    summary = aggregate(metrics)
    assert summary["functional"]["denominator"] == len(metrics)
    assert summary["functional"]["numerator"] == sum(
        m["functional"]["numerator"] for m in metrics)