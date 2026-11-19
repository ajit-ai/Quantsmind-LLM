"""Focused tests for Increment 6 stabilization analysis. No training."""

from __future__ import annotations

import pytest

from evaluation.behavior_suite import BEHAVIORS, build_cases, suite_digest
from evaluation.stabilization import (
    compare_behaviors,
    decide_stabilization,
    per_behavior_summary,
)
from training.binding_curriculum import build_training_examples as build_binding
from training.functional_curriculum import build_training_examples as build_functional


def test_supplemental_suite_is_deterministic_and_complete():
    assert build_cases() == build_cases()
    assert suite_digest() == suite_digest()
    cases = build_cases()
    assert len(cases) == 48
    assert {case["behavior"] for case in cases} == set(BEHAVIORS)
    for behavior in BEHAVIORS:
        assert sum(case["behavior"] == behavior for case in cases) == 6
    assert all(case["tests"] for case in cases)


def test_supplemental_suite_does_not_overlap_frozen_or_training():
    import json
    from pathlib import Path

    ad1 = json.loads(Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json").read_text(encoding="utf-8"))
    c3 = json.loads(Path("evaluation/python_benchmark_heldout_v1.json").read_text(encoding="utf-8"))
    frozen = {(c["instruction"], c["signature"]) for c in ad1["cases"]} | {
        (c["instruction"], c["signature"]) for c in c3["cases"]
    }
    trained = {
        (e.instruction, e.signature)
        for e in build_binding() + build_functional()
    }
    supplemental = {(c["instruction"], c["signature"]) for c in build_cases()}
    assert supplemental.isdisjoint(frozen)
    assert supplemental.isdisjoint(trained)


def test_supplemental_reference_answers_are_correct():
    from evaluation.functional_eval import evaluate_functional

    for case in build_cases():
        result = evaluate_functional(
            case["signature"] + "\n" + case["reference_body"] + "\n",
            case["function_name"],
            case["tests"],
        )
        assert result.passed is True, (case["case_id"], result)


def _record(behavior, parses, exact, structural, binding, passed):
    return {
        "behavior": behavior,
        "parses": parses,
        "exact": exact,
        "structural": structural,
        "binding": binding,
        "functional": {"passed": passed},
    }


def test_per_behavior_summary_and_comparison():
    baseline = [_record("scale", True, False, True, True, True), _record("scale", True, False, True, False, False)]
    candidate = [_record("scale", True, True, True, True, True), _record("scale", True, False, True, True, True)]
    base_summary = per_behavior_summary(baseline)
    cand_summary = per_behavior_summary(candidate)
    assert base_summary["scale"]["functional_rate"] == 0.5
    assert cand_summary["scale"]["functional_rate"] == 1.0
    comparison = compare_behaviors(base_summary, cand_summary)
    assert comparison["improved"] == ["scale"]
    assert comparison["regression_count"] == 0


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        (
            {
                "baseline_functional_rate": 0.0,
                "candidate_functional_rate": 0.02,
                "baseline_conditional": 0.0,
                "candidate_conditional": 0.1,
                "behavior_regressions": 0,
                "ad1_binding_loss": 1,
                "ad1_parse_fall": False,
                "candidate_collapsed": False,
            },
            "A",
        ),
        (
            {
                "baseline_functional_rate": 0.0,
                "candidate_functional_rate": 0.02,
                "baseline_conditional": 0.0,
                "candidate_conditional": 0.1,
                "behavior_regressions": 0,
                "ad1_binding_loss": 4,
                "ad1_parse_fall": False,
                "candidate_collapsed": False,
            },
            "B",
        ),
        (
            {
                "baseline_functional_rate": 0.0,
                "candidate_functional_rate": 0.0,
                "baseline_conditional": 0.0,
                "candidate_conditional": 0.0,
                "behavior_regressions": 0,
                "ad1_binding_loss": 0,
                "ad1_parse_fall": False,
                "candidate_collapsed": False,
            },
            "C",
        ),
        (
            {
                "baseline_functional_rate": 0.02,
                "candidate_functional_rate": 0.0,
                "baseline_conditional": 0.0,
                "candidate_conditional": 0.0,
                "behavior_regressions": 3,
                "ad1_binding_loss": 2,
                "ad1_parse_fall": False,
                "candidate_collapsed": False,
            },
            "D",
        ),
        (
            {
                "baseline_functional_rate": 0.0,
                "candidate_functional_rate": 0.02,
                "baseline_conditional": 0.0,
                "candidate_conditional": 0.1,
                "behavior_regressions": 0,
                "ad1_binding_loss": 1,
                "ad1_parse_fall": True,
                "candidate_collapsed": False,
            },
            "D",
        ),
    ],
)
def test_decision_matrix(kwargs, expected):
    assert decide_stabilization(**kwargs)["decision"] == expected
