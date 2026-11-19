"""Focused tests for the Architecture B planner prototype. No training."""

from __future__ import annotations


def test_planner_builds_validated_plans():
    from evaluation.operation_planner import compile_plan, plan, validate

    result = plan("Return the sum of two numbers.", "def add_values(left_value, right_value):")
    assert result.status == "planned"
    assert validate(result, "def add_values(left_value, right_value):") is True
    compiled = compile_plan(result, "def add_values(left_value, right_value):")
    assert compiled["action"] == "planned"
    assert "left_value + right_value" in compiled["body"]


def test_planner_supports_five_operation_families():
    from evaluation.operation_planner import compile_plan, plan

    checks = [
        ("Multiply the pair, then add two.", "def f(a_qz, b_qz):", "arithmetic"),
        ("Return the text in upper case.", "def f(raw_qz):", "string"),
        ("Keep only positive numbers.", "def f(values_qz):", "collection"),
        ("Add up every entry.", "def f(values_qz):", "aggregation"),
        ("Return the larger of two values.", "def f(a_qz, b_qz):", "condition"),
        ("Store the amount, then return it doubled.", "def f(amount_qz):", "arithmetic"),
    ]
    for instruction, signature, want in checks:
        result = plan(instruction, signature)
        assert result.status == "planned", (instruction, result)
        assert result.plan.family == want, (instruction, result.plan.family)
        compiled = compile_plan(result, signature)
        assert compiled["action"] == "planned"


def test_planner_rejects_unresolved_dependencies_and_unsupported():
    from evaluation.operation_planner import Plan, PlanResult, PlanStep, compile_plan

    bad = PlanResult("planned", Plan((PlanStep("add", ("missing_qz",), "result"),), "result", "arithmetic"), "test")
    assert compile_plan(bad, "def f(available_qz):")["action"] == "invalid"
    from evaluation.operation_planner import plan

    unsupported = plan("Cryptographically hash the input.", "def h(value_qz):")
    assert unsupported.status == "unsupported"
    assert unsupported.plan is None


def test_planner_execution_through_safe_gate():
    from evaluation.functional_eval import evaluate_functional
    from evaluation.operation_planner import compile_plan, plan

    instruction = "Keep the positive numbers, then count them."
    signature = "def and_a(values_qz):"
    result = plan(instruction, signature)
    compiled = compile_plan(result, signature)
    assert compiled["action"] == "planned"
    functional = evaluate_functional(
        signature + "\n" + compiled["body"] + "\n",
        "and_a",
        [{"args": [[-1, 2, 3]], "kwargs": {}, "expected": 2}],
    )
    assert functional.passed is True


def test_planner_provenance_distinct_from_model_output():
    from evaluation.operation_planner import compile_plan, plan

    compiled = compile_plan(plan("Return the sum.", "def f(a_qz, b_qz):"), "def f(a_qz, b_qz):")
    assert compiled["provenance"] == "planner"
