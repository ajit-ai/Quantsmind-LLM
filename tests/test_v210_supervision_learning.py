"""Taksh v2.1.0 qualification tests: supervision alignment, operation
curriculum, aligned training metrics and preservation.

Offline, CPU-bounded, deterministic. No training is executed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.operation_tasks import (
    ALTERNATE_NAMES,
    HELD_OUT_FAMILIES,
    OPERATIONS,
    PARAPHRASES,
    build_operation_curriculum,
    contamination_report,
    curriculum_digest,
)
from training.supervised import (
    SupervisionError,
    build_example,
    supervise,
    verify_supervision,
)
from training.supervised_train import (
    EvaluationError,
    analyse_program,
    reconstruct_program,
    supervised_metrics,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(REPO_ROOT / "tokenizer" / "qm_code_bpe_v0.json"))


@pytest.fixture(scope="module")
def curriculum():
    return build_operation_curriculum(20261021)


# --------------------------------------------------------------------------- #
# Supervision alignment
# --------------------------------------------------------------------------- #
def test_instruction_precedes_the_target():
    example = build_example(
        example_id="e1", instruction="Return the sum.", signature="def f(a, b):",
        body="    return a + b", operation="add", family="arithmetic",
        split="T", seed=1,
    )
    assert example.prompt_text.startswith("Return the sum.")
    assert example.prompt_text.endswith("def f(a, b):\n")
    assert example.target_text == "    return a + b\n"
    assert example.full_text.index("Return the sum.") < example.full_text.index("return a + b")


def test_empty_instruction_or_body_is_refused():
    with pytest.raises(SupervisionError):
        build_example(example_id="e", instruction="  ", signature="def f():",
                      body="    return 1", operation="add", family="arithmetic",
                      split="T", seed=1)
    with pytest.raises(SupervisionError):
        build_example(example_id="e", instruction="Do it.", signature="def f():",
                      body="   ", operation="add", family="arithmetic",
                      split="T", seed=1)


def test_prompt_tokens_are_masked_and_the_target_is_supervised(tokenizer):
    example = build_example(
        example_id="e2", instruction="Return the sum.", signature="def f(a, b):",
        body="    return a + b", operation="add", family="arithmetic",
        split="T", seed=1,
    )
    supervision = supervise(tokenizer, example, max_length=128)
    assert supervision["prompt_len"] > 0
    assert supervision["target_len"] > 0
    # the first target token is supervised: labels[prompt_len - 1] predicts it
    assert supervision["labels"][supervision["prompt_len"] - 1] != -100
    assert len(supervision["masked_positions"]) == supervision["prompt_len"] - 1
    assert supervision["eos_position"] == len(supervision["ids"]) - 1
    assert supervision["truncated"] is False


def test_verification_reports_no_defect_for_a_good_example(tokenizer):
    example = build_example(
        example_id="e3", instruction="Return the sum.", signature="def f(a, b):",
        body="    return a + b", operation="add", family="arithmetic",
        split="T", seed=1,
    )
    assert verify_supervision(tokenizer, example, max_length=128) == []


def test_truncation_is_reported_and_can_remove_the_operation(tokenizer):
    example = build_example(
        example_id="e4", instruction="Return the sum.", signature="def f(a, b):",
        body="    return a + b", operation="add", family="arithmetic",
        split="T", seed=1,
    )
    supervision = supervise(
        tokenizer, example, max_length=12, required_markers=(" + ",)
    )
    assert supervision["truncated"] is True
    assert supervision["operation_preserved"] is False
    problems = verify_supervision(
        tokenizer, example, max_length=12, required_markers=(" + ",)
    )
    assert any("truncation" in problem for problem in problems)


def test_eos_is_supervised_and_padding_is_excluded(tokenizer):
    example = build_example(
        example_id="e5", instruction="Return the sum.", signature="def f(a, b):",
        body="    return a + b", operation="add", family="arithmetic",
        split="T", seed=1,
    )
    supervision = supervise(tokenizer, example, max_length=128)
    eos_id = tokenizer.token_to_id("<eos>")
    assert eos_id in supervision["ids"]
    assert supervision["labels"][-1] == eos_id
    assert all(label != -100 for label in supervision["labels"][supervision["prompt_len"] - 1:])


# --------------------------------------------------------------------------- #
# Operation curriculum
# --------------------------------------------------------------------------- #
def test_curriculum_is_deterministic(curriculum):
    again = build_operation_curriculum(20261021)
    assert curriculum_digest(again) == curriculum_digest(curriculum)


def test_thirteen_operations_are_defined():
    assert len(OPERATIONS) == 13
    for operation, entry in OPERATIONS.items():
        family, instruction, signature, body, markers = entry
        assert family and instruction and signature and body and markers


def test_every_operation_has_a_paraphrase():
    for operation in OPERATIONS:
        assert operation in PARAPHRASES


def test_partitions_are_present_and_sized(curriculum):
    assert set(curriculum) == {"T", "V", "H1", "H2", "H3", "C"}
    assert len(curriculum["T"]) >= 20
    assert len(curriculum["C"]) == 12


def test_held_out_families_are_absent_from_training(curriculum):
    trained = {example.family for example in curriculum["T"]}
    for family in HELD_OUT_FAMILIES:
        assert family not in trained
    assert {example.family for example in curriculum["H3"]} == set(HELD_OUT_FAMILIES)


def test_contrastive_pairs_differ_by_one_requirement(curriculum):
    pairs = {}
    for example in curriculum["C"]:
        pairs.setdefault(example.metadata["pair_id"], []).append(example)
    assert pairs
    for members in pairs.values():
        assert len(members) == 2
        assert members[0].instruction != members[1].instruction
        assert members[0].operation != members[1].operation


def test_validation_partition_uses_unseen_names(curriculum):
    renamed = [
        example
        for example in curriculum["V"]
        if any(old in example.signature for old in ALTERNATE_NAMES.values())
    ]
    assert renamed


def test_contamination_report_is_clean(curriculum):
    report = contamination_report(curriculum)
    assert report["contaminated"] is False
    assert report["violation_count"] == 0


def test_no_training_text_is_shared_across_partitions(curriculum):
    """The same operation may repeat inside a partition, but no training text
    may appear in two partitions."""
    seen = {}
    for split in curriculum:
        for example in curriculum[split]:
            key = example.content_sha256
            assert key not in seen or seen[key] == split
            seen.setdefault(key, split)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_metrics_report_denominators():
    records = [{
        "syntax_valid": True, "complete_function": True, "operation_agreement": True,
        "exact_target": False, "degenerate": False, "eos_terminated": True,
        "truncated": False, "tokens": 12, "example_id": "x", "operation": "add",
        "family": "arithmetic", "split": "T", "text": "",
    }]
    metrics = supervised_metrics(records)
    assert metrics["cases"] == 1
    assert metrics["operation_agreement"] == 1
    assert metrics["exact_target"] == 0


# --------------------------------------------------------------------------- #
# Evaluation reconstruction (correctness of the v2.1.0 assessment)
# --------------------------------------------------------------------------- #
def _assess(signature, reference_body, generated_body):
    reference = analyse_program(reconstruct_program(signature, reference_body))
    generated = analyse_program(reconstruct_program(signature, generated_body))
    expected = reference["operations"]
    structural = bool(
        generated["parses"] and expected and expected <= generated["operations"]
    )
    exact = bool(
        generated["parses"] and generated["body_dump"] == reference["body_dump"]
    )
    if exact:
        status = "complete_and_exact"
    elif structural:
        status = "complete_correct_operation"
    elif generated["parses"]:
        status = "complete_wrong_operation"
    else:
        status = "incomplete"
    return generated, structural, exact, status


def test_reconstruction_excludes_the_natural_language_instruction():
    source = reconstruct_program(
        "def add_values(left, right):", "    return left + right"
    )
    assert source == "def add_values(left, right):\n    return left + right\n"
    assert "Write a function" not in source
    with pytest.raises(EvaluationError):
        reconstruct_program("   ", "    return 1")


def test_every_reference_body_parses_and_agrees_with_itself(curriculum):
    for partition, examples in curriculum.items():
        for example in examples:
            generated, structural, exact, status = _assess(
                example.signature, example.body, example.body
            )
            assert generated["parses"] is True, (partition, example.example_id)
            assert generated["operations"], (partition, example.example_id)
            assert structural is True, (partition, example.example_id)
            assert exact is True, (partition, example.example_id)
            assert status == "complete_and_exact", (partition, example.example_id)


def test_complete_wrong_operation_is_not_credited():
    generated, structural, exact, status = _assess(
        "def add_values(left, right):", "    return left + right",
        "    return left - right",
    )
    assert generated["parses"] is True
    assert structural is False
    assert exact is False
    assert status == "complete_wrong_operation"


def test_unterminated_or_empty_function_is_incomplete():
    _, _, _, status = _assess(
        "def multiply_values(left, right):", "    return left * right",
        "    return left * right:",
    )
    assert status == "incomplete"
    _, _, _, empty = _assess("def f():", "    return 1", "")
    assert empty == "incomplete"


def test_prematurely_terminated_function_is_not_exact():
    generated, structural, exact, status = _assess(
        "def sum_values(values):", "    total = 0\n    return total",
        "    return total",
    )
    assert generated["parses"] is True
    assert structural is False
    assert exact is False
    assert status == "complete_wrong_operation"


def test_boolean_comparison_is_not_a_maximum():
    markers = OPERATIONS["maximum"][4]
    assert any(marker in "    return left > right" for marker in markers)
    _, structural, exact, _ = _assess(
        "def maximum_value(left, right):",
        "    if left > right:\n        return left\n    return right",
        "    return left > right",
    )
    assert structural is False
    assert exact is False


def test_selection_with_a_different_but_equivalent_body_is_structural():
    _, structural, exact, status = _assess(
        "def maximum_value(left, right):",
        "    if left > right:\n        return left\n    return right",
        "    if left > right:\n        return left\n    else:\n        return right",
    )
    assert structural is True
    assert exact is False
    assert status == "complete_correct_operation"


def test_operator_in_an_unrelated_expression_is_not_credited():
    _, structural, exact, _ = _assess(
        "def add_values(left, right):", "    return left + right",
        "    unused = left + right\n    return left - right",
    )
    assert structural is False
    assert exact is False
    _, structural, exact, _ = _assess(
        "def count_values(values):",
        "    count = 0\n    for value in values:\n        count += 1\n    return count",
        "    total = 0\n    count = len(values)\n    return total",
    )
    assert structural is False
    assert exact is False


def test_analysis_is_static_and_never_executes_source():
    result = analyse_program(
        reconstruct_program("def f():", "    raise SystemExit('boom')")
    )
    assert result["parses"] is True


def test_metrics_report_structural_and_exact_counts_separately():
    def record(status, parses, structural, exact):
        return {
            "syntax_valid": parses, "parses": parses,
            "structural_operation": structural, "exact_target": exact,
            "status": status, "degenerate": False, "eos_terminated": True,
            "truncated": False, "tokens": 10,
        }

    records = [
        record("complete_and_exact", True, True, True),
        record("complete_correct_operation", True, True, False),
        record("complete_wrong_operation", True, False, False),
        record("incomplete", False, False, False),
    ]
    metrics = supervised_metrics(records)
    assert metrics["structural_operation"] == 2
    assert metrics["exact_target"] == 1
    assert metrics["complete_and_exact"] == 1
    assert metrics["complete_correct_operation"] == 1
    assert metrics["complete_wrong_operation"] == 1
    assert metrics["incomplete"] == 1


def test_production_authorization_is_unchanged():
    from data.python_v1.readiness import assess

    report = assess(REPO_ROOT / "data" / "contracts" / "python_v1.yaml")
    assert report["sources"]["approved"] == 0
    assert report["production_training_authorized"] is False


def test_protected_artifacts_are_untouched():
    import subprocess

    result = subprocess.run(
        ["git", "status", "--porcelain", "--", "tokenizer/", "runs/",
         "evaluation/python_eval/data/", "docs/roadmap/", "configs/",
         "data/contracts/"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.stdout.strip() == ""
