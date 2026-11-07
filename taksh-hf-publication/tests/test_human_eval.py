"""Tests for the human evaluation framework (M2.10)."""

from __future__ import annotations

import json

import pytest

from evaluation.human_eval import (
    RUBRIC,
    cohens_kappa,
    load_judgements,
    report,
    validate_judgements,
)


def row(item, annotator, winner, scores=None):
    return {
        "item_id": item,
        "annotator": annotator,
        "winner": winner,
        "scores": scores or {"correctness": 2, "safety": 2, "clarity": 1},
    }


def write_rows(tmp_path, rows, name="j.jsonl"):
    path = tmp_path / name
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# --------------------------------------------------------------------------- #
# Rubric
# --------------------------------------------------------------------------- #
def test_rubric_is_versioned_and_anchored():
    assert RUBRIC["rubric_version"] == "1"
    for criterion in RUBRIC["criteria"]:
        assert criterion["anchors"], f"{criterion['id']} has no anchors"
        # Every scale point must be anchored, or annotators guess.
        for point in criterion["scale"]:
            assert str(point) in criterion["anchors"]


def test_rubric_states_its_own_unvalidated_state():
    assert any(
        "not yet been exercised" in limitation
        for limitation in RUBRIC["known_limitations"]
    )
    assert RUBRIC["protocol"]["reliability_threshold"]["minimum"] == 0.6
    assert RUBRIC["protocol"]["min_annotators_per_item"] >= 2


def test_rubric_is_blinded():
    assert "neither system name" in RUBRIC["protocol"]["blinding"]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation_accepts_well_formed_judgements():
    valid, problems = validate_judgements([row("i1", "a1", "A")])
    assert len(valid) == 1
    assert problems == []


def test_validation_rejects_missing_fields():
    valid, problems = validate_judgements([{"item_id": "i1", "winner": "A"}])
    assert valid == []
    assert "annotator" in problems[0]


def test_validation_rejects_unknown_winner():
    valid, problems = validate_judgements([row("i1", "a1", "C")])
    assert valid == []
    assert "winner must be" in problems[0]


def test_validation_rejects_off_scale_scores():
    bad = row("i1", "a1", "A", {"correctness": 5})
    valid, problems = validate_judgements([bad])
    assert valid == []
    assert "outside the rubric scale" in problems[0]


def test_validation_rejects_unknown_criteria():
    bad = row("i1", "a1", "A", {"vibes": 2})
    valid, problems = validate_judgements([bad])
    assert valid == []
    assert "unknown criteria" in problems[0]


def test_load_judgements_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"item_id": "i"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_judgements(path)


def test_load_judgements_tolerates_a_utf8_bom(tmp_path):
    """Windows annotation tooling writes a BOM; that must not reject the file."""
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(row("i1", "a1", "A")).encode())
    assert len(load_judgements(path)) == 1


# --------------------------------------------------------------------------- #
# Agreement
# --------------------------------------------------------------------------- #
def test_single_annotator_cannot_produce_a_score():
    result = report([row("i1", "a1", "A")])
    assert result["releasable"] is False
    assert result["agreement"]["kappa"] is None
    assert "must not be released" in result["agreement"]["note"]


def test_full_agreement_gives_perfect_kappa():
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "A"),
        row("i2", "a1", "B"), row("i2", "a2", "B"),
    ]
    agreement = cohens_kappa(rows)
    assert agreement["kappa"] == pytest.approx(1.0)
    assert agreement["raw_agreement"] == pytest.approx(1.0)
    assert agreement["unanimous_items"] == 2
    assert agreement["disputed_items"] == []


def test_single_label_unanimity_leaves_kappa_undefined_not_perfect():
    """Every judgement "A" has no variance, so kappa cannot be computed.

    Reporting 1.0 here would be a fabricated reliability figure.
    """
    rows = [row("i1", "a1", "A"), row("i1", "a2", "A")]
    agreement = cohens_kappa(rows)
    assert agreement["kappa"] is None
    assert agreement["raw_agreement"] == pytest.approx(1.0)
    assert "same label" in agreement["kappa_undefined_reason"]


def test_disagreement_is_exposed_not_averaged_away():
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "B"),
        row("i2", "a1", "B"), row("i2", "a2", "B"),
    ]
    agreement = cohens_kappa(rows)
    assert agreement["disputed_items"] == ["i1"]
    assert agreement["kappa"] < 1.0
    assert agreement["unanimous_all"] is False


def test_anti_correlated_annotators_score_below_chance():
    """Two annotators who disagree systematically must not look mediocre.

    A marginal-based chance term would score this as merely ordinary; paired
    kappa correctly reports that they are worse than chance.
    """
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "B"),
        row("i2", "a1", "B"), row("i2", "a2", "A"),
        row("i3", "a1", "A"), row("i3", "a2", "B"),
    ]
    agreement = cohens_kappa(rows)
    assert agreement["raw_agreement"] == 0.0
    assert agreement["kappa"] < 0.0


def test_report_blocks_release_below_threshold():
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "A"),
        row("i2", "a1", "B"), row("i2", "a2", "A"),
        row("i3", "a1", "A"), row("i3", "a2", "B"),
    ]
    result = report(rows)
    assert result["releasable"] is False
    assert "NOT releasable" in result["decision"]


def test_report_blocks_release_on_validation_problems():
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "A"),
        {"item_id": "i2", "annotator": "a1"},
    ]
    result = report(rows)
    assert result["releasable"] is False
    assert result["validation_problems"]


def test_unanimous_batch_is_releasable_via_the_unanimity_route():
    rows = [
        row(f"i{i}", a, w)
        for i in range(4)
        for a, w in (("a1", "A"), ("a2", "A"))
    ]
    result = report(rows)
    # kappa is undefined (single label), yet the batch is still releasable:
    # every item was judged identically by two independent annotators.
    assert result["agreement"]["kappa"] is None
    assert result["unanimity_route"] is True
    assert result["releasable"] is True


def test_unanimity_route_needs_two_independent_annotators():
    rows = [row("i1", "a1", "A")]
    result = report(rows)
    assert result["unanimity_route"] is False
    assert result["releasable"] is False


def test_unanimity_route_does_not_rescue_a_mixed_batch():
    """Unanimous items do not excuse a disputed one."""
    rows = [
        row("i1", "a1", "A"), row("i1", "a2", "A"),
        row("i2", "a1", "A"), row("i2", "a2", "B"),
        row("i3", "a1", "A"), row("i3", "a2", "A"),
        row("i4", "a1", "A"), row("i4", "a2", "A"),
    ]
    result = report(rows)
    assert result["unanimity_route"] is False
    assert result["releasable"] is False
    assert result["agreement"]["disputed_items"] == ["i2"]