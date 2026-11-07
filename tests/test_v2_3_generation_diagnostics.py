"""Taksh v2.3.0 — Regression and reproducibility tests."""

from __future__ import annotations

import sys

from taksh.v2_3_detection import detect_degeneration
from taksh.v2_3_parameter_binding import analyse_parameter_binding, check_renamed_params
from taksh.v2_3_composition import analyse_composition
from taksh.v2_3_holdout_mapping import build_mapping_case, evaluate_mapping_case
from taksh.v2_3_canary import run_canary, compare_to_baseline


def test_repeated_tokens_detection():
    result = detect_degeneration("x = 1 + 1 + 1 + 1 + 1", "test_001")
    assert result.classification in ("warning", "fail")


def test_simple_identity_is_not_degenerate():
    result = detect_degeneration("def f(x): return x", "test_002")
    assert result.classification == "pass"
    assert not result.degenerate_semantics


def test_empty_function_body_is_fail():
    result = detect_degeneration("def f(x): pass", "test_003")
    assert result.classification == "fail"
    assert result.degenerate_semantics


def test_premature_eos_detection():
    result = detect_degeneration("def f(x): return (1 +", "test_004")
    assert result.premature_eos or result.classification in ("warning", "fail")


def test_excessive_repetition_is_fail():
    result = detect_degeneration("x = " + "foo " * 50, "test_005")
    assert result.excessive_repetition or result.classification in ("warning", "fail")


def test_parameter_binding_basic():
    result = analyse_parameter_binding("def add(a, b):", "    return a + b")
    assert result["parses"]
    assert result["consistent_references"]
    assert "a" in result["params_used"]
    assert "b" in result["params_used"]


def test_parameter_binding_undefined_name():
    result = analyse_parameter_binding("def f(x):", "    return y")
    assert result["parses"]
    assert "y" in result["undefined_names"]
    assert not result["consistent_references"]


def test_renamed_params_preserve_binding():
    result = check_renamed_params("def add(a, b):", "    return a + b", ["x", "y"])
    assert result["binding_preserved"]


def test_renamed_params_mismatch_count():
    result = check_renamed_params("def add(a, b):", "    return a + b", ["x"])
    assert not result["binding_preserved"]


def test_all_params_used():
    result = analyse_parameter_binding("def add(a, b, c):", "    return a + b + c")
    assert result["all_params_used"]


def test_composition_analysis():
    result = analyse_composition("def add_mul(a, b):", "    return a + b * 2")
    assert result["parses"]
    assert "add" in result["operations"]
    assert "multiply" in result["operations"]


def test_composition_depth():
    result = analyse_composition("def f(x):", "    if x > 0: return x")
    assert result["parses"]
    assert result["depth"] >= 1


def test_composition_novel_ops():
    result = analyse_composition("def f(x):", "    return foo(x)")
    assert "foo" in result["novel_operations"]


def test_mapping_case_build():
    case = build_mapping_case(
        "case_001",
        "Write a function to add two numbers",
        {"operations": ["add"], "params": ["a", "b"]},
    )
    assert case["case_id"] == "case_001"
    assert case["instruction_fingerprint"] is not None
    assert not case["is_negative"]


def test_mapping_case_evaluate():
    case = build_mapping_case(
        "case_002",
        "Write a function to add two numbers",
        {"operations": ["add"], "params": ["a", "b"]},
    )
    result = evaluate_mapping_case(case, "def add(a, b):\n    return a + b")
    assert result["parses"]
    assert result["structure_match"]
    assert "add" in result["found_ops"]
    assert "a" in result["found_params"]
    assert "b" in result["found_params"]


def test_mapping_negative_control():
    case = build_mapping_case(
        "case_neg",
        "Do not write a function",
        {"operations": [], "params": []},
        is_negative=True,
    )
    result = evaluate_mapping_case(case, "print('hello')")
    assert result["is_negative"]


def test_canary_pass():
    gen1 = detect_degeneration("def add(a, b): return a + b", "gen_001")
    gen2 = detect_degeneration("def sub(a, b): return a - b", "gen_002")
    canary = run_canary(
        checkpoint_path="test/checkpoint.pt",
        checkpoint_hash="abc123",
        generations=[gen1, gen2],
        config={"seed": 42},
    )
    outcome = canary.compute_outcome()
    assert outcome in ("PASS", "WARNING")
    assert canary.summary["total_generations"] == 2


def test_canary_does_not_flag_simple_identity():
    gen = detect_degeneration("def f(x): return x", "gen_fail")
    canary = run_canary(
        checkpoint_path="test/checkpoint.pt",
        checkpoint_hash="abc123",
        generations=[gen],
    )
    outcome = canary.compute_outcome()
    assert outcome == "PASS"
    assert not canary.summary["has_identity_failure"]


def test_compare_to_baseline():
    baseline = [detect_degeneration("def add(a, b): return a + b", "base_001")]
    candidate = [detect_degeneration("def f(x): return x", "cand_001")]
    canary = run_canary("test/ckpt.pt", "hash1", candidate)
    comparison = compare_to_baseline(canary, baseline, "base_hash")
    assert "fail_rate_delta" in comparison
    assert "regression_detected" in comparison



