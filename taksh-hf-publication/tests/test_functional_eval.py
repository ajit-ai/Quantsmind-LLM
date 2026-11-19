"""Focused tests for Increment 4 functional evaluation and curriculum."""

from __future__ import annotations

import pytest

from evaluation.functional_eval import (
    evaluate_functional,
    functional_report,
    safe_execution_available,
)


def _tests(args, expected):
    return [{"args": list(args), "kwargs": {}, "expected": expected}]


def test_safe_execution_boundary_is_available():
    assert safe_execution_available() is True


def test_correct_implementation_passes():
    result = evaluate_functional(
        "def add_pair(a, b):\n    return a + b\n",
        "add_pair",
        _tests((6, 7), 13),
    )
    assert result.passed is True
    assert result.category == "pass"


def test_incorrect_arithmetic_fails_functionally():
    result = evaluate_functional(
        "def add_pair(a, b):\n    return a - b\n",
        "add_pair",
        _tests((6, 7), 13),
    )
    assert result.passed is False
    assert result.category in ("incorrect_input_output_relationship", "incorrect_return_expression", "incorrect_operation")


def test_incorrect_binding_fails():
    result = evaluate_functional(
        "def add_pair(x, y):\n    return left + right\n",
        "add_pair",
        _tests((6, 7), 13),
    )
    assert result.passed is False
    assert result.category == "incorrect_parameter_binding"


def test_incorrect_string_transform_fails():
    result = evaluate_functional(
        "def shout(text):\n    return text.lower()\n",
        "shout",
        _tests(("hello",), "HELLO"),
    )
    assert result.passed is False


def test_incorrect_list_transform_fails():
    result = evaluate_functional(
        "def keep_positive(values):\n    return [i for i in values if i < 0]\n",
        "keep_positive",
        _tests(([-2, 3, 0, 5],), [3, 5]),
    )
    assert result.passed is False


def test_boundary_behaviour_is_checked():
    source = "def count_positive(values):\n    count = 0\n    for item in values:\n        if item > 0:\n            count += 1\n    return count\n"
    assert evaluate_functional(source, "count_positive", _tests(([],), 0)).passed is True
    assert evaluate_functional(source, "count_positive", _tests(([-1, 0],), 0)).passed is True
    assert evaluate_functional(source, "count_positive", _tests(([-1, 0],), 1)).passed is False


def test_plausible_but_wrong_code_is_rejected():
    # Parses, binds correctly, plausible shape — wrong constant.
    source = "def triple(n):\n    return n * 2\n"
    result = evaluate_functional(source, "triple", _tests((5,), 15))
    assert result.passed is False
    assert result.category in (
        "incorrect_input_output_relationship",
        "incorrect_return_expression",
        "superficially_plausible_but_functionally_incorrect",
    )


def test_timeout_is_a_safe_failure():
    source = (
        "def spin(n):\n    total = 0\n    for i in range(100000000):\n        total += i\n    return total\n"
    )
    result = evaluate_functional(source, "spin", _tests((5,), 0))
    assert result.passed is False
    assert result.category == "execution_timeout"


def test_forbidden_constructs_fail_closed():
    for source in (
        "import os\n\ndef f(n):\n    return n\n",
        "def f(n):\n    return open('x').read()\n",
        "def f(n):\n    return eval('n')\n",
        "def f(n):\n    return __import__('os')\n",
    ):
        result = evaluate_functional(source, "f", _tests((1,), 1))
        assert result.passed is False
        # Never executed: rejected by the gate or never reached.
        assert result.category in (
            "rejected_by_safe_execution_gate",
            "syntax_structural_failure",
            "execution_error",
        )


def test_functional_report_separates_conditional_metrics():
    records = [
        {"parses": True, "exact": False, "structural": True, "binding": True,
         "functional": {"passed": True}, "failure_category": "pass"},
        {"parses": True, "exact": False, "structural": True, "binding": True,
         "functional": {"passed": False}, "failure_category": "superficially_plausible_but_functionally_incorrect"},
        {"parses": False, "exact": False, "structural": False, "binding": False,
         "functional": {"passed": None}, "failure_category": "syntax_structural_failure"},
    ]
    report = functional_report(records)
    assert report["parse_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert report["binding_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert report["functional_rate"] == 0.5
    assert report["functional_given_parse_count"] == 2
    assert report["functional_given_parse"] == 0.5
    assert report["functional_given_binding"] == 0.5
    assert report["failure_categories"]["syntax_structural_failure"] == 1


def test_increment3_binding_without_functional_pattern_is_exposed():
    # Increment 3 pattern: 10 binding passes, 1 functional pass among them.
    records = [
        {"parses": True, "exact": False, "structural": True, "binding": True,
         "functional": {"passed": (index == 0)},
         "failure_category": "pass" if index == 0 else "superficially_plausible_but_functionally_incorrect"}
        for index in range(10)
    ]
    report = functional_report(records)
    assert report["binding_rate"] == 1.0
    assert report["functional_given_binding"] == 0.1
    assert report["functional_rate"] == 0.1


# --- Functional curriculum ---------------------------------------------------

def test_functional_curriculum_reproducible_and_valid():
    from training.functional_curriculum import (
        build_training_examples,
        build_validation_examples,
        functional_digest,
    )

    first = build_training_examples()
    second = build_training_examples()
    assert [e.example_id for e in first] == [e.example_id for e in second]
    assert functional_digest() == functional_digest()
    assert len(first) == 8 * 12
    assert len(build_validation_examples()) == 8 * 4
    training_pairs = {(e.instruction, e.signature) for e in first}
    validation_pairs = {(e.instruction, e.signature) for e in build_validation_examples()}
    assert training_pairs.isdisjoint(validation_pairs)
    for example in first + build_validation_examples():
        assert example.metadata["inputs"] and example.metadata["expected_outputs"]
        assert example.metadata["relationship"]


def test_functional_curriculum_examples_execute_correctly():
    from training.functional_curriculum import build_training_examples, build_validation_examples

    for example in build_training_examples() + build_validation_examples():
        tests = [
            {"args": list(args), "kwargs": {}, "expected": expected}
            for args, expected in zip(example.metadata["inputs"], example.metadata["expected_outputs"])
        ]
        source = example.signature + "\n" + example.body + "\n"
        name = example.signature.split("(")[0].removeprefix("def ").strip()
        result = evaluate_functional(source, name, tests)
        assert result.passed is True, (example.example_id, result)


def test_functional_curriculum_passes_quality_and_leakage_checks():
    import json
    from pathlib import Path

    from training.binding_curriculum import (
        build_training_examples as build_binding,
        build_validation_examples as build_binding_val,
    )
    from training.curriculum_quality import check_curriculum
    from training.functional_curriculum import (
        build_training_examples as build_functional,
        build_validation_examples as build_functional_val,
    )

    combined = build_binding() + build_functional()
    combined_val = build_binding_val() + build_functional_val()
    ad1 = json.loads(Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json").read_text(encoding="utf-8"))
    c3 = json.loads(Path("evaluation/python_benchmark_heldout_v1.json").read_text(encoding="utf-8"))
    heldout = {(c["instruction"], c["signature"]) for c in ad1["cases"]} | {
        (c["instruction"], c["signature"]) for c in c3["cases"]
    }
    report = check_curriculum(combined, combined_val, heldout_pairs=heldout)
    assert report["train_examples"] == 156 + 96
    assert report["validation_examples"] == 52 + 32
    assert report["exact_duplicates"] == 0
    assert report["train_validation_leakage"] == 0
    assert report["heldout_overlap"] == 0
