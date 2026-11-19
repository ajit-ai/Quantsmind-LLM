"""TAKSH v1.3.2-B generalization tests (novel names, not fixture cases).

Covers ordered composition planning, precedence, predicate semantics,
identifier preservation, invalid/unsupported handling, and provenance.
No case IDs, fixture names, expected-output dictionaries, or prompt
substitutions are used as implementation logic.
"""

from __future__ import annotations

from evaluation.functional_eval import evaluate_functional
from evaluation.operation_planner import Plan, PlanResult, PlanStep, compile_plan, describe_validation, plan, validate
from evaluation.operation_synth import synthesize


def _run(instruction, signature, fn, tests):
    result = synthesize(instruction, signature)
    assert result["action"] == "synthesized", (instruction, result)
    outcome = evaluate_functional(signature + "\n" + result["body"] + "\n", fn, tests)
    assert outcome.passed, (instruction, result, outcome.reason)


# --- ordered composition (synthesizer) ---

def test_add_then_divide_by_novel():
    _run("Add the two numbers, then divide by four.",
         "def calc_zx(alpha_q, beta_q):", "calc_zx",
         [{"args": [6, 2], "kwargs": {}, "expected": 2.0}, {"args": [0, 8], "kwargs": {}, "expected": 2.0}])


def test_join_then_length_novel():
    _run("Join the pair with a space, then report its length.",
         "def calc_zx(left_z, right_z):", "calc_zx",
         [{"args": ["ab", "cd"], "kwargs": {}, "expected": 5}, {"args": ["", ""], "kwargs": {}, "expected": 1}])


def test_filter_above_threshold_count_novel():
    _run("Filter the entries above five and count them.",
         "def calc_zx(items_z):", "calc_zx",
         [{"args": [[1, 7, 9, 3]], "kwargs": {}, "expected": 2}, {"args": [[5]], "kwargs": {}, "expected": 0}])


def test_uppercase_each_then_count_novel():
    _run("Uppercase each entry, then count the entries.",
         "def calc_zx(words_z):", "calc_zx",
         [{"args": [["a", "b", "c"]], "kwargs": {}, "expected": 3}, {"args": [[]], "kwargs": {}, "expected": 0}])


def test_accumulate_then_report_novel():
    _run("Accumulate all the entries, then report the total.",
         "def calc_zx(values_z):", "calc_zx",
         [{"args": [[1, 2, 3]], "kwargs": {}, "expected": 6}, {"args": [[]], "kwargs": {}, "expected": 0}])


def test_product_of_all_novel():
    _run("Multiply all entries together.",
         "def calc_zx(nums_z):", "calc_zx",
         [{"args": [[2, 3, 4]], "kwargs": {}, "expected": 24}, {"args": [[]], "kwargs": {}, "expected": 1}])


def test_without_last_novel():
    _run("Return the collection without its last entry.",
         "def calc_zx(items_z):", "calc_zx",
         [{"args": [[1, 2, 3]], "kwargs": {}, "expected": [1, 2]}, {"args": [[7]], "kwargs": {}, "expected": []}])


# --- precedence and predicate semantics ---

def test_nested_precedence_not_left_to_right():
    _run("Compute first plus second times third.",
         "def calc_zx(x_z, y_z, z_z):", "calc_zx",
         [{"args": [2, 3, 4], "kwargs": {}, "expected": 14}, {"args": [1, 1, 10], "kwargs": {}, "expected": 11}])


def test_subtract_second_from_first():
    _run("Subtract the second value from the first.",
         "def calc_zx(x_z, y_z):", "calc_zx",
         [{"args": [9, 4], "kwargs": {}, "expected": 5}, {"args": [3, 9], "kwargs": {}, "expected": -6}])


def test_length_exactly_novel():
    _run("Return True when the length is exactly four.",
         "def calc_zx(text_z):", "calc_zx",
         [{"args": ["abcd"], "kwargs": {}, "expected": True}, {"args": ["abc"], "kwargs": {}, "expected": False}])


def test_difference_at_least_novel():
    _run("Return True when the first value exceeds the second by at least three.",
         "def calc_zx(x_z, y_z):", "calc_zx",
         [{"args": [10, 7], "kwargs": {}, "expected": True}, {"args": [9, 7], "kwargs": {}, "expected": False}])


def test_needle_absent_novel():
    _run("Return True when the needle is absent.",
         "def calc_zx(items_z, needle_z):", "calc_zx",
         [{"args": [[1, 2], 3], "kwargs": {}, "expected": True}, {"args": [[1, 2], 2], "kwargs": {}, "expected": False}])


def test_upper_equals_novel():
    _run("Return True when the upper-cased text equals the text.",
         "def calc_zx(text_z):", "calc_zx",
         [{"args": ["XYZ"], "kwargs": {}, "expected": True}, {"args": ["xyz"], "kwargs": {}, "expected": False}])


def test_count_is_even_novel():
    _run("Return True when the count is even.",
         "def calc_zx(items_z):", "calc_zx",
         [{"args": [[1, 2]], "kwargs": {}, "expected": True}, {"args": [[1]], "kwargs": {}, "expected": False}])


def test_second_character_novel():
    _run("Return the second character of the code.",
         "def calc_zx(code_z):", "calc_zx",
         [{"args": ["abc"], "kwargs": {}, "expected": "b"}, {"args": ["xy"], "kwargs": {}, "expected": "y"}])


def test_duplicate_text_doubles():
    _run("Duplicate the text.", "def calc_zx(text_z):", "calc_zx",
         [{"args": ["ab"], "kwargs": {}, "expected": "abab"}, {"args": ["x"], "kwargs": {}, "expected": "xx"}])


def test_duplicate_input_is_identity():
    _run("Duplicate the input into the output.", "def calc_zx(item_z):", "calc_zx",
         [{"args": [42], "kwargs": {}, "expected": 42}, {"args": [0], "kwargs": {}, "expected": 0}])


# --- identifier preservation ---

def test_rename_returns_input():
    _run("Rename the amount and return it.", "def calc_zx(amount_z):", "calc_zx",
         [{"args": [9], "kwargs": {}, "expected": 9}])


def test_copy_under_fresh_local_returns_input():
    _run("Copy the label under a fresh local name.", "def calc_zx(label_z):", "calc_zx",
         [{"args": ["tag"], "kwargs": {}, "expected": "tag"}])


def test_double_within_local_scalar():
    _run("Double within a local, then return it.", "def calc_zx(value_z):", "calc_zx",
         [{"args": [6], "kwargs": {}, "expected": 12}, {"args": [-4], "kwargs": {}, "expected": -8}])


def test_carry_into_cleaned_local_strips():
    _run("Carry the text into a cleaned local.", "def calc_zx(text_z):", "calc_zx",
         [{"args": [" hi "], "kwargs": {}, "expected": "hi"}, {"args": ["go"], "kwargs": {}, "expected": "go"}])


# --- planner: ordered composition, validation, invalid/unsupported ---

def test_planner_ordered_add_divide_validated():
    result = plan("Add the two numbers, then divide by four.", "def calc_zx(alpha_q, beta_q):")
    assert result.status == "planned"
    assert validate(result, "def calc_zx(alpha_q, beta_q):")
    status = describe_validation(result, "def calc_zx(alpha_q, beta_q):")
    assert status["status"] == "VALIDATED"
    steps = [s["operation"] for s in status["plan"]["steps"]]
    assert steps == ["add", "divide_by"]
    # Intermediate value is distinct from the final output.
    assert status["plan"]["steps"][0]["result"] == "combined"
    assert status["plan"]["output"] == "result"
    compiled = compile_plan(result, "def calc_zx(alpha_q, beta_q):")
    assert compiled["action"] == "planned" and compiled.get("provenance") == "planner"


def test_planner_rejects_undefined_identifier():
    bad = PlanResult("planned", Plan((PlanStep("multiply", ("missing_q", "other_q"), "result"),), "result", "arithmetic"), "test")
    assert not validate(bad, "def calc_zx(alpha_q, beta_q):")
    assert describe_validation(bad, "def calc_zx(alpha_q, beta_q):")["status"] == "INVALID_PLAN"


def test_planner_rejects_duplicate_result():
    bad = PlanResult("planned", Plan((
        PlanStep("add", ("alpha_q", "beta_q"), "combined"),
        PlanStep("multiply", ("combined", "beta_q"), "combined"),
    ), "combined", "arithmetic"), "test")
    assert not validate(bad, "def calc_zx(alpha_q, beta_q):")


def test_planner_rejects_cycle():
    bad = PlanResult("planned", Plan((PlanStep("add", ("alpha_q", "stage_q"), "stage_q"),), "stage_q", "arithmetic"), "test")
    assert not validate(bad, "def calc_zx(alpha_q, beta_q):")


def test_planner_rejects_forward_reference_ordering():
    bad = PlanResult("planned", Plan((
        PlanStep("multiply", ("combined", "2"), "result"),
        PlanStep("add", ("alpha_q", "beta_q"), "combined"),
    ), "result", "arithmetic"), "test")
    assert not validate(bad, "def calc_zx(alpha_q, beta_q):")


def test_planner_unsupported_is_not_invalid():
    result = plan("Synchronously hash the harbor entry.", "def calc_zx(value_q):")
    assert result.status == "unsupported"
    assert describe_validation(result, "def calc_zx(value_q):")["status"] == "UNSUPPORTED_PLAN"
    assert validate(result, "def calc_zx(value_q):")  # unsupported != proven invalid


def test_synthesizer_abstains_on_ambiguous():
    result = synthesize("Explain the fenestration.", "def calc_zx(value_q):")
    assert result["action"] == "abstain"
