"""Focused tests for Increment 9 identifier supervision and curriculum."""

from __future__ import annotations

import pytest
import torch


def test_identifier_classification_categorizes_occurrences():
    from training.identifier_supervision import classify_identifiers

    occurrences = classify_identifiers(
        "def add_values(left_value, right_value):",
        "    total = left_value + right_value\n    return total\n",
    )
    by_name = {}
    for occurrence in occurrences:
        by_name.setdefault(occurrence["name"], set()).add(occurrence["category"])
    assert by_name["left_value"] == {"copied"}
    assert by_name["right_value"] == {"copied"}
    # total is substituted at its definition and propagated at use.
    assert by_name["total"] == {"propagated", "substituted"}


def test_substitution_and_builtin_categories():
    from training.identifier_supervision import classify_identifiers

    occurrences = classify_identifiers(
        "def rename_it(source_value):",
        "    renamed = source_value\n    return renamed\n",
    )
    assert occurrences[0]["name"] == "renamed" and occurrences[0]["category"] == "substituted"
    categories = {}
    for occurrence in occurrences:
        categories.setdefault(occurrence["name"], set()).add(occurrence["category"])
    assert categories["source_value"] == {"copied"}
    assert categories["renamed"] == {"substituted", "propagated"}

    occurrences = classify_identifiers(
        "def count_it(values):",
        "    return len(values)\n",
    )
    categories = {o["name"]: o["category"] for o in occurrences}
    assert categories["len"] == "builtin"
    assert categories["values"] == "copied"


def test_loss_weights_preserve_alignment_and_boost_copy_tokens():
    from tokenizers import Tokenizer

    from training.identifier_supervision import classify_identifiers, loss_weights
    from training.supervised import build_example, supervise

    tokenizer = Tokenizer.from_file("tokenizer/qm_code_bpe_v0.json")
    example = build_example(
        example_id="TEST-copy-000",
        instruction="Return the sum of two numbers.",
        signature="def add_values(left_value, right_value):",
        body="    total = left_value + right_value\n    return total\n",
        operation="add",
        family="arithmetic",
        split="train",
        seed=1,
    )
    supervision = supervise(tokenizer, example)
    assert not __import__("training.supervised", fromlist=["verify_supervision"]).verify_supervision(tokenizer, example)
    occurrences = classify_identifiers(example.signature, example.body)
    weighted = loss_weights(supervision, occurrences, tokenizer=tokenizer)
    weights = weighted["weights"]
    assert len(weights) == len(supervision["labels"])
    # Prompt positions stay at weight 1.0 and carry IGNORE_INDEX labels.
    from training.objective import IGNORE_INDEX

    assert all(
        supervision["labels"][i] == IGNORE_INDEX for i in range(max(0, supervision["prompt_len"] - 1))
    )
    assert weighted["occurrence_counts"]["copied"] >= 2
    assert max(weights) > 1.0  # copy tokens boosted, then mean-normalized
    assert weighted["weighted_tokens"] >= 1


@pytest.mark.parametrize(
    "copy_weight",
    [0.5, 1.5],
)
def test_loss_weight_config_validation(copy_weight):
    from training.identifier_supervision import IdentifierSupervisionConfig

    if copy_weight < 1.0:
        with pytest.raises(ValueError):
            IdentifierSupervisionConfig(copy_weight=copy_weight)
    else:
        IdentifierSupervisionConfig(copy_weight=copy_weight)


def test_weighted_loss_matches_unweighted_when_weights_uniform():
    from training.identifier_supervision import weighted_token_mean_loss

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 20)
    labels = torch.tensor([[0, 1, 2, -100, 3, 4], [5, 6, -100, -100, 7, 8]])
    weights = torch.ones(2, 6)
    weighted, count = weighted_token_mean_loss(logits, labels, weights)
    from training.objective import token_mean_loss

    summed, plain_count = token_mean_loss(logits, labels)
    assert count == plain_count
    assert torch.allclose(weighted, summed / count, atol=1e-5)


def test_copy_curriculum_stages_and_separation():
    from evaluation.behavior_suite import build_cases
    from evaluation.diagnostics import COPY_CASES
    from training.copy_curriculum import (
        STAGES,
        build_training_examples,
        build_validation_examples,
        copy_curriculum_digest,
    )

    train = build_training_examples()
    val = build_validation_examples()
    assert [e.example_id for e in train] == [e.example_id for e in build_training_examples()]
    assert copy_curriculum_digest() == copy_curriculum_digest()
    assert {e.metadata["stage"] for e in train} == set(STAGES)
    assert {(e.instruction, e.signature) for e in train}.isdisjoint(
        {(e.instruction, e.signature) for e in val}
    )
    # Frozen and supplemental suites must not leak into training data.
    import json
    from pathlib import Path

    ad1 = json.loads(Path(r"C:\Users\Lenovo\AppData\Local\Temp\opencode\ad1\ad1_suite.json").read_text(encoding="utf-8"))
    c3 = json.loads(Path("evaluation/python_benchmark_heldout_v1.json").read_text(encoding="utf-8"))
    frozen = {(c["instruction"], c["signature"]) for c in ad1["cases"]} | {
        (c["instruction"], c["signature"]) for c in c3["cases"]
    }
    supplemental = {(c["instruction"], c["signature"]) for c in build_cases()}
    copy_pairs = {(instruction, signature) for _i, _k, instruction, signature, _f in COPY_CASES}
    trained = {(e.instruction, e.signature) for e in train}
    assert trained.isdisjoint(frozen)
    assert trained.isdisjoint(supplemental)
    assert trained.isdisjoint(copy_pairs)


def test_copy_curriculum_examples_are_functionally_correct():
    from evaluation.functional_eval import evaluate_functional
    from training.copy_curriculum import build_training_examples, build_validation_examples

    for example in build_training_examples() + build_validation_examples():
        tests = [
            {"args": list(args), "kwargs": {}, "expected": expected}
            for args, expected in zip(example.metadata["inputs"], example.metadata["expected_outputs"])
        ]
        name = example.signature.split("(")[0].removeprefix("def ").strip()
        result = evaluate_functional(example.signature + "\n" + example.body + "\n", name, tests)
        assert result.passed is True, (example.example_id, result)


def test_copy_eval_report_separates_metrics():
    from evaluation.copy_eval import copy_report

    scored = [
        {"kind": "add", "familiar": True, "parses": True, "exact_copy": True,
         "substitution": False, "propagation": True, "multi_binding": True},
        {"kind": "add", "familiar": False, "parses": True, "exact_copy": False,
         "substitution": False, "propagation": True, "multi_binding": False},
    ]
    report = copy_report(scored)
    assert report["all"]["cases"] == 2
    assert report["familiar"]["exact_copy"] == 1.0
    assert report["unfamiliar"]["exact_copy"] == 0.0
    assert report["all"]["multi_binding"] == 0.5
