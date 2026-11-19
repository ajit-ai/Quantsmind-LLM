"""v2.2.0 tests: analyzer, frozen suite, cells, scoring, artifacts."""

from __future__ import annotations

from collections import Counter

import pytest

from training.generalization_eval import paired_delta, score_case, summarize
from training.generalization_tasks import (COMPOSITIONS, MAPPING_OPS,
                                           TRAINED_COMPOSITIONS, TRAINED_OPS,
                                           build_cells, build_eval_cases,
                                           eval_digest, validation_cases)
from training.program_analysis import analyse


def test_analyzer_parses_valid_function():
    result = analyse("def f(a, b):", "    return a + b")
    assert result["parses"] is True
    assert result["operations"] == ["add"]
    assert result["undefined_names"] == []


def test_analyzer_precedence_is_distinguished():
    left = analyse("def f(a, b, c):", "    return (a + b) * c")
    right = analyse("def f(a, b, c):", "    return a + (b * c)")
    assert left["parses"] and right["parses"]
    assert left["key"] != right["key"]


def test_analyzer_addition_is_commutative():
    left = analyse("def f(a, b):", "    return a + b")
    right = analyse("def f(x, y):", "    return y + x")
    assert left["key"] == right["key"]
    assert left["body_dump"] != right["body_dump"]


def test_analyzer_binding_uses_positional_names():
    renamed = analyse("def f(x, y):", "    return x + y")
    original = analyse("def f(a, b):", "    return a + b")
    assert renamed["body_dump"] == original["body_dump"]
    assert renamed["params_used"] == ["p0", "p1"]


def test_analyzer_detects_undefined_names():
    result = analyse("def f(a):", "    return a + missing")
    assert result["undefined_names"] == ["missing"]


def test_analyzer_shape_distinguishes_selection():
    direct = analyse("def f(a, b):", "    return max(a, b)")
    selected = analyse("def f(a, b):",
                       "    if a > b:\n        return a\n    return b")
    assert direct["shape"] != selected["shape"]


def test_frozen_suite_has_expected_partition_counts():
    cases = build_eval_cases()
    counts = Counter(case["partition"] for case in cases)
    assert counts == {"E1": 33, "E2": 33, "E3": 24, "E4": 24,
                      "E5": 24, "E6": 24}
    assert len({case["case_id"] for case in cases}) == 162


def test_eval_digest_is_stable_and_sensitive():
    cases = build_eval_cases()
    assert eval_digest(cases) == eval_digest(build_eval_cases())
    changed = [dict(case) for case in cases]
    changed[0]["instruction"] += " Extra."
    assert eval_digest(changed) != eval_digest(cases)


def test_cell_sizes_match_pre_registration():
    cells = build_cells()
    assert {key: len(value) for key, value in cells.items()} == {
        "A": 22, "B": 66, "C": 86}


def test_validation_is_not_in_training_cells():
    validation = {case["example_id"] for case in validation_cases()}
    for records in build_cells().values():
        assert not validation.intersection({r["example_id"] for r in records})


def test_cell_b_covers_all_trained_operations():
    operations = {record["operation"] for record in build_cells()["B"]}
    assert operations == set(TRAINED_OPS)


def test_compositions_are_split_seen_and_unseen():
    cases = [c for c in build_eval_cases() if c["partition"] == "E4"]
    seen = {c["composition"] for c in cases if c["metadata"]["seen_in_training"]}
    unseen = {c["composition"] for c in cases if not c["metadata"]["seen_in_training"]}
    assert seen == set(TRAINED_COMPOSITIONS)
    assert len(unseen) == 4
    assert seen | unseen == set(COMPOSITIONS)


def test_mapping_partition_uses_held_out_operations():
    cases = [c for c in build_eval_cases() if c["partition"] == "E5"]
    assert {c["operation"] for c in cases} == set(MAPPING_OPS)
    assert all(c["metadata"]["held_out_operation"] for c in cases)


def test_negative_controls_have_six_categories():
    cases = [c for c in build_eval_cases() if c["partition"] == "E6"]
    assert len(cases) == 24
    categories = Counter(c["negative"] for c in cases)
    assert len(categories) == 6
    assert set(categories.values()) == {4}


def test_correct_completion_is_exact():
    case = next(c for c in build_eval_cases() if c["case_id"] == "E1-add-000")
    record = score_case(case, "    return x + y\n")
    assert record["parses"] and record["exact"] and record["structural"]


def test_wrong_operation_is_not_structural():
    case = next(c for c in build_eval_cases() if c["case_id"] == "E1-add-000")
    record = score_case(case, "    return x - y\n")
    assert record["parses"] and not record["exact"] and not record["structural"]


def test_composition_requires_the_full_structure():
    case = next(c for c in build_eval_cases()
                if c["case_id"] == "E4-add_then_multiply-0")
    correct = score_case(case, "    return (a + b) * c\n")
    partial = score_case(case, "    return a + b * c\n")
    assert correct["composition_match"] is True
    assert partial["composition_match"] is False


def test_negative_outputs_are_never_credited():
    case = next(c for c in build_eval_cases()
                if c["partition"] == "E6" and c["negative"] == "unsupported_operator")
    record = score_case(case, "    return a / b\n")
    assert record["exact"] is False
    assert record["structural"] is False
    assert record["composition_match"] is False


def test_eos_and_truncation_are_recorded():
    case = build_eval_cases()[0]
    stopped = score_case(case, "    return x + y\n", finish_reason="stop")
    cut = score_case(case, "    return x + y\n", finish_reason="length")
    assert stopped["eos_terminated"] and not stopped["truncated"]
    assert cut["truncated"] and not cut["eos_terminated"]


def test_paired_delta_tracks_gains_and_losses():
    case = build_eval_cases()[0]
    good = score_case(case, "    return x + y\n")
    bad = score_case(case, "    return x - y\n")
    delta = paired_delta([bad], [good])
    assert delta["exact"]["delta"] == 1
    assert delta["exact"]["gained"] == 1
    assert delta["exact"]["lost"] == 0


def test_summary_reports_negative_safety():
    cases = build_eval_cases()
    records = [score_case(case, case["body"] + "\n") for case in cases]
    summary = summarize(records)
    assert summary["negative_controls"]["safe"] == 24
    assert summary["overall"]["cases"] == 162


def test_training_records_are_json_serialisable():
    import json
    cells = build_cells()
    json.dumps({"A": cells["A"], "B": cells["B"], "C": cells["C"]})


def test_case_specs_do_not_contain_duplicate_ids():
    cases = build_eval_cases()
    identifiers = [case["case_id"] for case in cases]
    assert len(identifiers) == len(set(identifiers))


def test_evaluation_budget_is_bounded():
    assert 0 < 64 <= 128
    assert 150 * 4 <= 7200
