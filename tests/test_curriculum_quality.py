"""Unit tests for curriculum diversity and mode-collapse safeguards."""

from __future__ import annotations

import pytest

from scripts.evaluate_v120_increment1 import aggregate, diversity_report as eval_diversity
from training.binding_curriculum import (
    TRAINED_OPERATIONS,
    build_training_examples,
    build_validation_examples,
    curriculum_digest,
)
from training.curriculum_quality import (
    CurriculumQualityError,
    check_curriculum,
    detect_mode_collapse,
    diversity_report,
    normalised_shape,
    template_family,
)
from training.supervised import build_example

# Fixed sample inputs proving every surface variant is semantically correct.
_SAMPLES = {
    "add": ((6, 7), 13),
    "subtract": ((10, 4), 6),
    "multiply": ((6, 7), 42),
    "maximum": ((3, 9), 9),
    "minimum": ((3, 9), 3),
    "keep_positive": (([-2, 3, 0, 5],), [3, 5]),
    "keep_negative": (([-2, 3, 0, -5],), [-2, -5]),
    "upper": (("hello",), "HELLO"),
    "lower": (("HELLO",), "hello"),
    "count_values": (([4, 5, 6],), 3),
    "sum_values": (([4, 5, 6],), 15),
    "choose_first": (("alpha", "beta"), "alpha"),
    "choose_second": (("alpha", "beta"), "beta"),
}


def _run(example, args):
    namespace: dict = {}
    exec(example.signature + "\n" + example.body, {}, namespace)
    name = example.signature.split("(")[0].removeprefix("def ").strip()
    return namespace[name](*args)


def test_every_variant_executes_correctly():
    examples = build_training_examples()
    assert {example.operation for example in examples} == set(TRAINED_OPERATIONS)
    for example in examples:
        args, expected = _SAMPLES[example.operation]
        assert _run(example, args) == expected, example.example_id
    for example in build_validation_examples():
        args, expected = _SAMPLES[example.operation]
        assert _run(example, args) == expected, example.example_id


def test_template_families_distinguish_surface_forms():
    assert template_family("def f(v):", "    return [i for i in v if i > 0]") == "direct_listcomp_return"
    assert template_family("def f(v):", "    return len(v)") == "direct_len_call_return"
    assert template_family("def f(v):", "    result = []\n    result.extend(i for i in v if i > 0)\n    return result") == "assign_name_return"
    assert template_family("def f(v):", "    return len(v)") == "direct_len_call_return"
    assert template_family("def f(a, b):", "    return a + b") == "direct_binop_return"
    assert template_family("def f(a, b):", "    total = a + b\n    return total") == "assign_name_return"
    assert template_family("def f(a, b):", "    result = a\n    result += b\n    return result") == "augassign_name_return"
    # Renamings share one family; restructurings do not.
    assert template_family("def f(x, y):", "    return x + y") == template_family(
        "def f(alpha, beta):", "    return alpha + beta"
    )
    assert normalised_shape("def f(a, b):", "    return a + b") != normalised_shape(
        "def f(a, b):", "    total = a + b\n    return total"
    )


def test_new_curriculum_passes_all_quality_checks():
    train = build_training_examples()
    val = build_validation_examples()
    report = check_curriculum(train, val)
    assert report["train_examples"] == 13 * 12
    assert report["validation_examples"] == 13 * 4
    assert report["operations"] == {op: 12 for op in TRAINED_OPERATIONS}
    assert report["listcomp_share"] < 0.10
    assert report["exact_duplicates"] == 0
    assert report["train_validation_leakage"] == 0
    assert all(len(v) >= 3 for v in report["templates_per_operation"].values())
    # Digest is reproducible.
    assert curriculum_digest() == curriculum_digest()


def _single_template_corpus():
    """Increment 1 shape: 12 renamings of one template per operation."""
    return [
        build_example(
            example_id=f"OLD-add-{index:03d}",
            instruction="Return the sum of two numbers.",
            signature=f"def adds_pair(a{index}, b{index}):",
            body=f"    return a{index} + b{index}",
            operation="add",
            family="arithmetic",
            split="train",
            seed=index,
        )
        for index in range(12)
    ]


def test_quality_checks_reject_single_template_corpus():
    # Relax balance caps so the template-diversity check is what fires.
    from training.curriculum_quality import QualityThresholds

    thresholds = QualityThresholds(max_operation_share=1.0, max_template_share=1.0, max_listcomp_share=1.0)
    with pytest.raises(CurriculumQualityError, match="template diversity"):
        check_curriculum(_single_template_corpus(), thresholds=thresholds)


def _comprehension_heavy_corpus():
    examples = []
    for operation, signature, values in (
        ("keep_positive", "def f(v):", "v"),
        ("keep_negative", "def f(v):", "v"),
    ):
        for index in range(12):
            symbol = ">" if operation == "keep_positive" else "<"
            examples.append(
                build_example(
                    example_id=f"OLD-{operation}-{index:03d}",
                    instruction="Filter the list.",
                    signature=f"def f{index}(v{index}):".replace(f"f{index}(v{index})", f"fgc{index}(v{index})"),
                    body=f"    return [i for i in v{index} if i {symbol} 0]",
                    operation=operation,
                    family="collection",
                    split="train",
                    seed=index,
                )
            )
    return examples


def test_quality_checks_reject_comprehension_dominance():
    # Relax balance caps so the list-comprehension check is what fires.
    from training.curriculum_quality import QualityThresholds

    thresholds = QualityThresholds(
        max_operation_share=1.0,
        max_template_share=1.0,
        min_templates_per_operation=1,
        max_variant_share_within_operation=1.0,
    )
    with pytest.raises(CurriculumQualityError, match="list-comprehension dominance"):
        check_curriculum(_comprehension_heavy_corpus(), thresholds=thresholds)


def test_leakage_and_duplicate_checks_fire():
    train = build_training_examples()
    with pytest.raises(CurriculumQualityError, match="duplicate"):
        check_curriculum(train + [train[0]])
    with pytest.raises(CurriculumQualityError, match="leakage"):
        check_curriculum(train, [train[0]])


# --- Evaluation-side mode-collapse detection -------------------------------

_INCREMENT1_STYLE_COMPLETIONS = [
    ("arith", "    return [value for item in in values if item > 0]\n"),
    ("arith", "    return [value for item in in in values if item < 0]\n"),
    ("binding", "    return [item for item in in text.lower()\n"),
    ("string", "    return [value for item in in in values if item > 0]\n"),
    ("conditional", "    return [value for item in in in in values if item < 0]\n"),
    ("loop", "    return [value for item in in in values if item < 0]\n"),
]


def _records(pairs):
    return [
        {"category": category, "completion": completion, "signature": "def probe(a, b):"}
        for category, completion in pairs
    ]


def test_increment1_failure_mode_is_detected():
    result = detect_mode_collapse(_records(_INCREMENT1_STYLE_COMPLETIONS))
    assert result["collapsed"] is True
    assert result["degenerate_listcomp_text"] >= 3
    assert result["reasons"]


def test_diverse_outputs_are_not_flagged_collapsed():
    diverse = _records([
        ("arith", "    return a + b\n"),
        ("string", "    return text.upper()\n"),
        ("collection", "    result = []\n    for item in values:\n        result.append(item)\n    return result\n"),
        ("count", "    return len(values)\n"),
        ("binding", "    return a\n"),
        ("conditional", "    return a if a > b else b\n"),
    ])
    result = detect_mode_collapse(diverse)
    assert result["collapsed"] is False
    report = diversity_report(diverse)
    assert report["distinct_templates"] >= 5
    assert report["degenerate_listcomp_text"] == 0


def test_same_template_single_category_is_not_collapsed():
    # Five legitimate list-shaped outputs in one category must not trip
    # the category-spanning collapse rule.
    records = _records([
        ("collection", f"    return [i for i in v{k}]\n") for k in range(5)
    ])
    result = detect_mode_collapse(records)
    assert result["collapsed"] is False


def test_evaluation_aggregate_includes_diversity():
    ad1 = []
    c3 = []
    for index, (category, completion) in enumerate(_INCREMENT1_STYLE_COMPLETIONS):
        record = {
            "case_id": f"T-{index}",
            "category": category,
            "parses": False,
            "exact": False,
            "structural": False,
            "binding": False,
            "functional": {"passed": None, "reason": "not_executed"},
            "completion": completion,
            "latency_seconds": 0.01,
        }
        (ad1 if index % 2 == 0 else c3).append(record)
    summary = aggregate(ad1, c3)
    # Original keys are preserved for comparability.
    assert summary["ad1"]["cases"] == 3
    assert summary["c3_heldout"]["cases"] == 3
    # New Increment 2 keys are present and agree with the standalone helpers.
    assert summary["ad1_diversity"] == eval_diversity(ad1)
    assert summary["mode_collapse"]["ad1"]["collapsed"] is True
    assert summary["mode_collapse"]["c3"]["collapsed"] is True
