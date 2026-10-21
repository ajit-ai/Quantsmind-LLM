"""Tests for the M5 evaluation system: contract, leakage, adapters, scoring,
runner behaviour and reproducibility."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.m5_schema import (
    is_scorable,
    manifest_hash,
    normalise,
    stable_id,
    text_hash,
    validate_dataset,
    validate_record,
)


def example(**overrides):
    record = {
        "id": stable_id("code_generation", "do the thing"),
        "task_type": "code_generation",
        "prompt": "Do the thing.",
        "expected_behavior": "does it",
        "test_specification": "reverse_string",
        "language": "python",
        "source_id": "org-a",
        "license_id": "MIT",
        "split": "eval",
        "provenance": {"origin": "test", "license_evidence": "MIT"},
    }
    record.update(overrides)
    return record


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Dataset contract
# --------------------------------------------------------------------------- #
def test_valid_record_accepted():
    assert validate_record(example()) == []


def test_missing_required_field_rejected():
    record = example()
    del record["provenance"]
    assert any("provenance" in p for p in validate_record(record))


def test_invalid_task_type_rejected():
    assert any("task_type" in p
               for p in validate_record(example(task_type="vibes")))


def test_duplicate_ids_detected(tmp_path):
    record = example()
    report = validate_dataset(write_jsonl(tmp_path / "b.jsonl", [record, record]))
    reasons = {r["reason"] for r in report["rejections"]}
    assert "duplicate_id" in reasons
    assert report["accepted"] == 1


def test_duplicate_prompts_detected_even_with_new_ids(tmp_path):
    first = example()
    second = example(id=stable_id("code_generation", "different anchor"))
    second["prompt"] = "Do   the THING. "  # differs only by case and spacing
    report = validate_dataset(write_jsonl(tmp_path / "b.jsonl", [first, second]))
    assert any(r["reason"] == "duplicate_prompt" for r in report["rejections"])


def test_malformed_line_is_reported_by_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\nnot json\n', encoding="utf-8")
    report = validate_dataset(path)
    assert report["malformed_lines"][0]["line"] == 2


def test_missing_provenance_rejected():
    assert any("provenance" in p
               for p in validate_record(example(provenance={"origin": "x"})))


def test_unverified_source_rejected_even_with_a_license_string(tmp_path):
    record = example(source_id="not-verified", license_id="Apache-2.0")
    report = validate_dataset(write_jsonl(tmp_path / "b.jsonl", [record]),
                              verified_sources={"other-repo"})
    assert any(r["reason"] == "unauthorized_source" for r in report["rejections"])
    assert report["accepted"] == 0


def test_invalid_split_rejected():
    assert any("split" in p for p in validate_record(example(split="whatever")))


def test_empty_dataset_is_handled(tmp_path):
    report = validate_dataset(write_jsonl(tmp_path / "empty.jsonl", []))
    assert report["total_records"] == 0
    assert report["accepted"] == 0
    assert report["scorable"] == 0


# --------------------------------------------------------------------------- #
# Scorability
# --------------------------------------------------------------------------- #
def test_task_with_a_test_is_scorable_even_without_a_checklist():
    assert is_scorable(example()) is True


def test_task_with_only_a_checklist_is_scorable():
    record = example(test_specification=None,
                     task_type="instruction_following",
                     checklist=[{"type": "non_empty"}])
    assert is_scorable(record) is True


def test_task_with_no_oracle_is_not_scorable():
    record = example(test_specification=None)
    assert is_scorable(record) is False


def test_human_only_task_is_never_scorable():
    record = example(task_type="software_architecture",
                     test_specification=None)
    assert is_scorable(record) is False


# --------------------------------------------------------------------------- #
# Normalisation and hashing
# --------------------------------------------------------------------------- #
def test_normalisation_collapses_formatting_differences():
    assert normalise("Do  THE   Thing.") == normalise("do the thing.")


def test_hash_is_stable_and_formatting_insensitive():
    assert text_hash("Do the Thing.") == text_hash("do  the thing.")


def test_manifest_hash_is_order_independent():
    a = example()
    b = example(id=stable_id("sql_generation", "other"), task_type="sql_generation")
    assert manifest_hash([a, b]) == manifest_hash([b, a])


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #
def test_exact_prompt_overlap_detected(tmp_path):
    from evaluation.m5_leakage import leakage_report

    corpus = write_jsonl(tmp_path / "train.jsonl", [example()])
    benchmark = write_jsonl(tmp_path / "bench.jsonl", [example()])
    report = leakage_report(benchmark, [("train", corpus, "prompt")])
    assert report["leaked"] is True
    assert report["references"][0]["overlapping_examples"]


def test_normalised_overlap_detected(tmp_path):
    from evaluation.m5_leakage import leakage_report

    corpus = write_jsonl(tmp_path / "train.jsonl",
                        [example(prompt="Do the thing.")])
    benchmark = write_jsonl(tmp_path / "bench.jsonl",
                            [example(prompt="DO   THE   THING.")])
    report = leakage_report(benchmark, [("train", corpus, "prompt")])
    assert report["references"][0]["normalised_overlaps"]


def test_no_overlap_reported_clean(tmp_path):
    from evaluation.m5_leakage import leakage_report

    corpus = write_jsonl(tmp_path / "train.jsonl",
                        [example(prompt="something else entirely")])
    benchmark = write_jsonl(tmp_path / "bench.jsonl", [example()])
    report = leakage_report(benchmark, [("train", corpus, "prompt")])
    assert report["leaked"] is False


def test_missing_source_information_is_a_limitation_not_clean(tmp_path):
    from evaluation.m5_leakage import leakage_report

    report = leakage_report(
        write_jsonl(tmp_path / "bench.jsonl", [example()]),
        [("absent", tmp_path / "nope.jsonl", "prompt")])
    finding = report["references"][0]
    assert finding["available"] is False
    assert "source_overlap_is_risk_not_contamination" in finding


def test_source_overlap_is_flagged_as_risk(tmp_path):
    from evaluation.m5_leakage import leakage_report

    corpus = write_jsonl(tmp_path / "train.jsonl", [example()])
    benchmark = write_jsonl(tmp_path / "bench.jsonl",
                           [example(prompt="different prompt")])
    report = leakage_report(benchmark, [("train", corpus, "prompt")])
    assert report["unresolved_risks"] == ["org-a"]
    assert report["references"][0]["source_overlap_is_risk_not_contamination"]


def test_shipped_benchmark_has_zero_leakage():
    manifest = json.loads(
        Path("evaluation/m5_benchmark.manifest.json").read_text(encoding="utf-8"))
    assert manifest["leakage"]["total_overlapping_examples"] == 0