"""Unified dataset record schema (M2.4).

Synthetic fixtures for every record kind and level, plus a read-only conformance
check against the corpora that actually ship. No file is rewritten by this suite.
"""

import hashlib
import json

import pytest

from data.schema import (
    KIND_CORPUS,
    KIND_SFT,
    KIND_VQA,
    REQUIRED_FIELDS,
    SCHEMA_VERSION,
    VALID_SPLITS,
    SchemaError,
    audit_dataset,
    detect_kind,
    validate_record,
)

TEXT = "def f():\n    return 1\n"


def corpus_record(**over):
    record = {
        "text": TEXT,
        "content_sha1": hashlib.sha1(TEXT.encode()).hexdigest(),
        "source_id": "org/repo",
        "repo": "org/repo",
        "commit": "a" * 40,
        "spdx": "MIT",
        "attribution": "Copyright Example",
        "file_path": "src/f.py",
        "retrieved_at": "2026-10-18",
        "split": "train",
        "language": "python",
    }
    record.update(over)
    return record


def sft_record(**over):
    record = {"instruction": "add two numbers", "response": "def add(a,b): return a+b"}
    record.update(over)
    return record


def vqa_record(**over):
    record = {
        "question": "What colour?",
        "image": "images/a.png",
        "expected_answer": "red",
    }
    record.update(over)
    return record


# --------------------------------------------------------------------------- #
# Kind detection
# --------------------------------------------------------------------------- #
def test_detect_kind_for_each_shape():
    assert detect_kind(corpus_record()) == KIND_CORPUS
    assert detect_kind(sft_record()) == KIND_SFT
    assert detect_kind(vqa_record()) == KIND_VQA


def test_detect_kind_returns_none_for_unrecognisable():
    assert detect_kind({"foo": "bar"}) is None
    assert detect_kind("not a dict") is None
    assert detect_kind(None) is None


# --------------------------------------------------------------------------- #
# Conforming records
# --------------------------------------------------------------------------- #
def test_conforming_corpus_record_has_no_problems():
    assert validate_record(corpus_record()) == []


def test_conforming_sft_and_vqa_records_have_no_problems():
    assert validate_record(sft_record()) == []
    assert validate_record(vqa_record()) == []


@pytest.mark.parametrize("kind", sorted(REQUIRED_FIELDS))
def test_every_required_field_is_actually_required(kind):
    fixture = {
        KIND_CORPUS: corpus_record,
        KIND_SFT: sft_record,
        KIND_VQA: vqa_record,
    }[kind]
    for field in REQUIRED_FIELDS[kind]:
        broken = fixture()
        broken.pop(field)
        problems = validate_record(broken, kind=kind)
        assert any(field in p for p in problems), f"{field} was not enforced"


# --------------------------------------------------------------------------- #
# Malformed input is rejected, never silently accepted
# --------------------------------------------------------------------------- #
def test_non_object_record_is_rejected():
    problems = validate_record(["not", "an", "object"])
    assert "not a JSON object" in problems[0]


def test_non_string_field_is_rejected():
    problems = validate_record(corpus_record(spdx=123))
    assert any("spdx" in p and "non-empty string" in p for p in problems)


def test_empty_string_field_is_rejected():
    problems = validate_record(corpus_record(attribution="   "))
    assert any("attribution" in p for p in problems)


def test_unrecognisable_record_is_reported():
    problems = validate_record({"foo": "bar"})
    assert "cannot determine record kind" in problems[0]


def test_unknown_level_is_a_programming_error():
    with pytest.raises(ValueError):
        validate_record(corpus_record(), level="nonsense")


# --------------------------------------------------------------------------- #
# Integrity invariants
# --------------------------------------------------------------------------- #
def test_content_hash_mismatch_is_detected():
    problems = validate_record(corpus_record(content_sha1="0" * 40))
    assert any("content_sha1 does not match text" in p for p in problems)


def test_tampered_text_with_stale_hash_is_detected():
    record = corpus_record()
    record["text"] = "def f():\n    return 2\n"  # content changed, hash left alone
    assert any("content_sha1 does not match" in p for p in validate_record(record))


def test_unknown_split_is_detected():
    problems = validate_record(corpus_record(split="holdout"))
    assert any("is not one of" in p for p in problems)


@pytest.mark.parametrize("split", VALID_SPLITS)
def test_every_documented_split_is_accepted(split):
    assert validate_record(corpus_record(split=split)) == []


def test_hash_is_not_checked_at_encoding_level():
    """The encoding boundary cares about text, not provenance."""
    minimal = {"text": TEXT}
    assert validate_record(minimal, level="encoding") == []
    assert validate_record(minimal, level="release") != []


def test_schema_error_type_exists():
    assert issubclass(SchemaError, ValueError)


# --------------------------------------------------------------------------- #
# Auditing a file (read-only)
# --------------------------------------------------------------------------- #
def _write(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def test_audit_reports_a_conforming_file(tmp_path):
    path = _write(tmp_path / "ok.jsonl", [corpus_record(), corpus_record()])
    summary = audit_dataset(path)
    assert summary["conforms"] is True
    assert summary["valid"] == 2 and summary["invalid"] == 0
    assert summary["kinds"] == {KIND_CORPUS: 2}
    assert summary["schema_version"] == SCHEMA_VERSION


def test_audit_counts_missing_provenance_without_failing(tmp_path):
    path = _write(tmp_path / "legacy.jsonl", [{"text": TEXT}, {"text": TEXT}])
    summary = audit_dataset(path)
    assert summary["conforms"] is False
    assert summary["invalid"] == 2
    assert summary["missing_fields"]["content_sha1"] == 2
    assert summary["missing_fields"]["split"] == 2


def test_audit_counts_unparsable_lines(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(corpus_record()) + "\n{not json\n", encoding="utf-8")
    summary = audit_dataset(path)
    assert summary["unparsable"] == 1
    assert summary["valid"] == 1
    assert summary["conforms"] is False


def test_audit_skips_blank_lines(tmp_path):
    path = tmp_path / "spaced.jsonl"
    path.write_text(
        json.dumps(corpus_record()) + "\n\n   \n" + json.dumps(corpus_record()) + "\n",
        encoding="utf-8",
    )
    summary = audit_dataset(path)
    assert summary["records"] == 2
    assert summary["conforms"] is True


def test_audit_handles_a_missing_file():
    summary = audit_dataset("definitely/absent.jsonl")
    assert summary["exists"] is False
    assert summary["conforms"] is False


def test_audit_problem_list_is_capped_but_count_is_complete(tmp_path):
    path = _write(tmp_path / "many.jsonl", [{"text": TEXT} for _ in range(30)])
    summary = audit_dataset(path, limit=5)
    assert summary["invalid"] == 30
    assert len(summary["problems"]) == 5


def test_mixed_kinds_are_counted_separately(tmp_path):
    path = _write(
        tmp_path / "mixed.jsonl",
        [corpus_record(), sft_record(), vqa_record()],
    )
    summary = audit_dataset(path)
    assert summary["kinds"] == {KIND_CORPUS: 1, KIND_SFT: 1, KIND_VQA: 1}
    assert summary["valid"] == 3


# --------------------------------------------------------------------------- #
# Conformance of the datasets that actually ship
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "path", ["data/code_corpus_v2.jsonl", "data/code_corpus_v1.jsonl"]
)
def test_released_corpora_conform_at_release_level(path):
    """The licence-gated code corpora must satisfy the release contract."""
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        pytest.skip(f"{path} not present in this working tree")
    summary = audit_dataset(target)
    assert summary["conforms"] is True, summary["problems"][:3]


def test_sft_corpus_conforms_to_the_sft_kind():
    from pathlib import Path

    target = Path("data/code_sft_v0.jsonl")
    if not target.exists():
        pytest.skip("code_sft_v0.jsonl not present")
    summary = audit_dataset(target)
    assert summary["conforms"] is True
    assert summary["kinds"] == {KIND_SFT: summary["records"]}


def test_legacy_corpora_are_reported_as_deviating_not_repaired():
    """Pre-M2.4 shards are reported as-is; this suite must not rewrite them."""
    from pathlib import Path

    target = Path("data/corpus_v0.jsonl")
    if not target.exists():
        pytest.skip("corpus_v0.jsonl not present")
    before = target.stat().st_mtime_ns
    size_before = target.stat().st_size

    summary = audit_dataset(target)
    assert summary["conforms"] is False
    assert summary["missing_fields"], "the deviation must be explained"

    assert target.stat().st_size == size_before
    assert target.stat().st_mtime_ns == before


# --------------------------------------------------------------------------- #
# The encoder refuses malformed input (and stays usable for valid input)
# --------------------------------------------------------------------------- #
CODE_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"


def _encode(corpus, out, extra=()):
    import sys

    from data.encode_dataset import main as encode_main

    saved = sys.argv
    sys.argv = [
        "encode_dataset",
        "--corpus", str(corpus),
        "--tokenizer", CODE_TOKENIZER,
        "--out", str(out),
        *extra,
    ]
    try:
        encode_main()
    finally:
        sys.argv = saved
    return out.with_suffix(".meta.json")


def test_encoder_accepts_a_conforming_corpus(tmp_path):
    corpus = _write(tmp_path / "ok.jsonl", [corpus_record(), corpus_record()])
    meta = json.loads(_encode(corpus, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["num_documents"] == 2
    assert meta["dataset_schema"] == SCHEMA_VERSION
    assert meta["dataset_schema_validated"] is True


def test_encoder_accepts_a_minimal_corpus_without_provenance(tmp_path):
    """Encoding needs text, not provenance; provenance is enforced upstream."""
    corpus = _write(tmp_path / "min.jsonl", [{"text": TEXT}])
    meta = json.loads(_encode(corpus, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["num_documents"] == 1


def test_encoder_warns_on_malformed_records_by_default(tmp_path, capsys):
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(
        json.dumps(corpus_record()) + "\n" + json.dumps({"text": ""}) + "\n",
        encoding="utf-8",
    )
    _encode(corpus, tmp_path / "s.bin")
    assert "encoding schema" in capsys.readouterr().out


def test_encoder_refuses_malformed_records_in_strict_mode(tmp_path):
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(json.dumps({"text": ""}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        _encode(corpus, tmp_path / "s.bin", extra=("--strict",))
    assert "encoding schema" in str(exc.value)
    assert not (tmp_path / "s.bin").exists(), "no artifact on a refused encode"


def test_encoder_can_be_told_not_to_validate(tmp_path):
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text(json.dumps({"text": ""}) + "\n", encoding="utf-8")
    meta = json.loads(
        _encode(corpus, tmp_path / "s.bin", extra=("--no-validate",)).read_text(
            encoding="utf-8"
        )
    )
    assert meta["dataset_schema_validated"] is False


def test_plain_text_corpora_are_exempt_from_json_validation(tmp_path):
    plain = tmp_path / "prose.txt"
    plain.write_text("first line\nsecond line\n", encoding="utf-8")
    meta = json.loads(_encode(plain, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["num_documents"] == 2