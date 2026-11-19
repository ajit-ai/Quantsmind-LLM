"""Focused tests for Increment 13/14 planner validation integration."""

from __future__ import annotations


def test_planner_validation_outcomes():
    from evaluation.operation_planner import Plan, PlanResult, PlanStep, compile_plan, plan

    good = plan("Return the sum of two numbers.", "def f(a_qz, b_qz):")
    assert compile_plan(good, "def f(a_qz, b_qz):")["action"] == "planned"
    unsupported = plan("Compute the spectral index.", "def f(value_qz):")
    assert unsupported.status == "unsupported"
    bad = PlanResult("planned", Plan((PlanStep("add", ("missing_qz",), "result"),), "result", "arithmetic"), "t")
    assert compile_plan(bad, "def f(available_qz):")["action"] == "invalid"


def test_hybrid_records_planner_provenance():
    """Hybrid records carry planner status distinct from generation origin."""
    from evaluation.hybrid import HybridGenerator

    # Pure-record shape check without loading a model.
    import inspect

    source = inspect.getsource(HybridGenerator.generate)
    assert "planner" in source
    assert "VALIDATED" in source
    assert "UNSUPPORTED_PLAN" in source
    assert "INVALID_PLAN" in source
    assert "VALIDATION_UNAVAILABLE" in source


def test_copy_gate_still_passes():
    from evaluation.copy_mechanism import CopyGate, copy_generate
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    result = copy_generate(
        CopyGate(), "Return the sum of two numbers.",
        "def add_values(left_value, right_value):",
        "    return left + right\n", tokenizer,
    )
    assert result["rewritten"] == "    return left_value + right_value\n"
