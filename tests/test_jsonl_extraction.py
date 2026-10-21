"""Corrective slice 1: JSONL content extraction and kind-specific encoding.

Regression coverage for two verified defects:

1. JSONL corpora were tokenized as raw JSON lines, so ``{"text":``,
   ``"id":`` and every provenance value became training tokens.
2. A schema-valid VQA record failed the ``encoding`` contract because ``text``
   was required of *every* kind.
"""

import hashlib
import json

import pytest

from data.schema import (
    ENCODABLE_KINDS,
    KIND_CORPUS,
    KIND_SFT,
    KIND_VQA,
    NotEncodable,
    audit_dataset,
    content_field_for,
    detect_kind,
    extract_content,
    validate_record,
)

CODE_TOKENIZER = "tokenizer/qm_code_bpe_v0.json"

DISTINCT = "def distinctive_fn(v):\n    return v + 1\n"


def corpus_record(text=DISTINCT, **over):
    record = {
        "text": text,
        "content_sha1": hashlib.sha1(text.encode()).hexdigest(),
        "source_id": "org/repo",
        "repo": "org/repo",
        "commit": "a" * 40,
        "spdx": "MIT",
        "attribution": "Copyright Example",
        "file_path": "src/f.py",
        "retrieved_at": "2026-10-18",
        "split": "train",
        "id": "RECID-XYZ",
        "meta_blob": "deadbeefcafebabe",
    }
    record.update(over)
    return record


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return path


def encode(corpus, out, tokenizer=CODE_TOKENIZER, extra=()):
    import sys

    from data.encode_dataset import main as encode_main

    saved = sys.argv
    sys.argv = [
        "encode_dataset",
        "--corpus", str(corpus),
        "--tokenizer", tokenizer,
        "--out", str(out),
        *extra,
    ]
    try:
        encode_main()
    finally:
        sys.argv = saved
    return out.with_suffix(".meta.json")


def decode_stream(path):
    from tokenizers import Tokenizer

    import numpy as np

    tok = Tokenizer.from_file(CODE_TOKENIZER)
    ids = np.fromfile(path, dtype=np.uint16).tolist()
    return tok.decode(ids, skip_special_tokens=False), ids, tok


# --------------------------------------------------------------------------- #
# Defect 1 -- JSONL content extraction
# --------------------------------------------------------------------------- #
def test_extracted_content_is_the_text_field_only():
    assert extract_content(corpus_record()) == DISTINCT


def test_extraction_ignores_id_and_provenance():
    record = corpus_record()
    extracted = extract_content(record)
    for leaked in ('{"text"', '"id"', "RECID-XYZ", "content_sha1",
                   "deadbeef", "org/repo", "MIT"):
        assert leaked not in extracted


def test_encoded_stream_contains_source_not_json_scaffolding(tmp_path):
    """The decisive regression: decode the stream and inspect it."""
    corpus = write_jsonl(tmp_path / "c.jsonl", [corpus_record()])
    encode(corpus, tmp_path / "s.bin")
    text, ids, tok = decode_stream(tmp_path / "s.bin")

    assert "def distinctive_fn" in text
    assert DISTINCT in text
    for leaked in ('{"text"', '"id"', "RECID-XYZ", "content_sha1",
                   "deadbeef", '"repo"', '"spdx"', "MIT", "}"):
        assert leaked not in text, f"JSON scaffolding leaked into tokens: {leaked!r}"


def test_token_and_separator_counts_are_correct(tmp_path):
    from training.document_separator import resolve_separator_id
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(CODE_TOKENIZER)
    sep = resolve_separator_id(tok)

    records = [corpus_record(text=f"def g{i}(v):\n    return v + {i}\n") for i in range(3)]
    corpus = write_jsonl(tmp_path / "c.jsonl", records)
    meta = json.loads(encode(corpus, tmp_path / "s.bin").read_text(encoding="utf-8"))
    ids = meta["num_tokens"]
    _, stream, _ = decode_stream(tmp_path / "s.bin")

    assert meta["num_documents"] == 3
    assert ids == len(stream)
    expected = 0
    for i, record in enumerate(records):
        expected += len(tok.encode(record["text"]).ids) + (1 if i else 0)
    assert ids == expected, "token count must equal the sum of the text fields"
    assert sum(1 for i in stream if i == sep) == 2, "one separator per boundary"


def test_metadata_hashes_describe_the_input_records(tmp_path):
    corpus = write_jsonl(tmp_path / "c.jsonl", [corpus_record()])
    meta = json.loads(encode(corpus, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert meta["tokenizer_sha256"] == hashlib.sha256(
        open(CODE_TOKENIZER, "rb").read()
    ).hexdigest()
    assert meta["dataset_schema"] == "quantsmind.dataset/1"
    assert meta["dataset_schema_validated"] is True


def test_encoding_is_deterministic(tmp_path):
    corpus = write_jsonl(tmp_path / "c.jsonl", [corpus_record(), corpus_record(
        text="def other(v):\n    return 0\n"
    )])
    encode(corpus, tmp_path / "a.bin")
    encode(corpus, tmp_path / "b.bin")
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes()


def test_malformed_json_is_rejected_and_writes_nothing(tmp_path, capsys):
    good = json.dumps(corpus_record())
    corpus = tmp_path / "broken.jsonl"
    corpus.write_text(good + "\n{not json\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        encode(corpus, tmp_path / "out.bin")
    assert "malformed JSON" in str(exc.value)
    assert ":2" in str(exc.value), "the failing line must be identified"
    assert not (tmp_path / "out.bin").exists()
    assert not (tmp_path / "out.meta.json").exists()


def test_plain_text_compatibility_is_unchanged(tmp_path):
    plain = tmp_path / "prose.txt"
    plain.write_text("first line of prose\n\nsecond line of prose\n", encoding="utf-8")
    meta = json.loads(encode(plain, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["num_documents"] == 2
    text, _, _ = decode_stream(tmp_path / "s.bin")
    assert "first line of prose" in text
    assert "second line of prose" in text
    assert "{" not in text


def test_legacy_jsonl_of_bare_text_records_still_encodes(tmp_path):
    """Legacy fixtures carry only 'text'; they remain encodable."""
    corpus = write_jsonl(tmp_path / "legacy.jsonl", [{"text": "plain one\n"},
                                                     {"text": "plain two\n"}])
    meta = json.loads(encode(corpus, tmp_path / "s.bin").read_text(encoding="utf-8"))
    assert meta["num_documents"] == 2
    text, _, _ = decode_stream(tmp_path / "s.bin")
    assert "plain one" in text and "plain two" in text


# --------------------------------------------------------------------------- #
# Defect 2 -- kind-specific encoding contract
# --------------------------------------------------------------------------- #
def test_valid_vqa_record_no_longer_fails_the_encoding_contract(tmp_path):
    """The defect: a schema-valid VQA record failed because it lacked 'text'."""
    vqa = {"question": "What colour?", "image": "a.png",
           "expected_answer": "red"}
    assert validate_record(vqa) == [], "VQA must be schema-valid at release level"
    assert validate_record(vqa, level="encoding") == [], (
        "a valid VQA record must not fail solely for lacking 'text'"
    )


def test_valid_corpus_and_sft_pass_encoding_validation():
    sft = {"instruction": "add a and b", "response": "def add(a,b): return a+b"}
    assert validate_record(corpus_record(), level="encoding") == []
    assert validate_record(sft, level="encoding") == []


def test_release_validation_is_not_weakened():
    """Defect 2 must not have loosened the release contract."""
    vqa = {"question": "What colour?", "image": "a.png",
           "expected_answer": "red"}
    assert validate_record({k: v for k, v in vqa.items() if k != "image"}) != []
    assert validate_record({k: v for k, v in vqa.items() if k != "question"}) != []
    # corpus release checks stay enforced
    assert validate_record({"text": "x"}) != []
    assert validate_record(corpus_record(content_sha1="0" * 40)) != []
    assert validate_record(corpus_record(split="holdout")) != []


def test_encodable_kinds_is_deliberately_narrow():
    assert KIND_CORPUS in ENCODABLE_KINDS
    assert KIND_SFT not in ENCODABLE_KINDS
    assert KIND_VQA not in ENCODABLE_KINDS


def test_content_field_is_declared_only_for_encodable_kinds():
    assert content_field_for(KIND_CORPUS) == "text"
    assert content_field_for(KIND_SFT) is None
    assert content_field_for(KIND_VQA) is None


def test_sft_and_vqa_are_refused_with_an_actionable_message(tmp_path):
    sft = write_jsonl(tmp_path / "sft.jsonl",
                      [{"instruction": "q", "response": "a"}])
    vqa = write_jsonl(tmp_path / "vqa.jsonl",
                      [{"question": "q", "image": "a.png",
                        "expected_answer": "a"}])
    for path, hint in ((sft, "training/sft.py"), (vqa, "eval_vqa.py")):
        with pytest.raises(SystemExit) as exc:
            encode(path, tmp_path / "out.bin")
        message = str(exc.value)
        assert "not a supported path" in message
        assert hint in message, "must name the real consumer"
        assert not (tmp_path / "out.bin").exists()


def test_extract_content_raises_for_non_encodable_kinds():
    with pytest.raises(NotEncodable) as exc:
        extract_content({"instruction": "q", "response": "a"})
    assert exc.value.kind == KIND_SFT
    with pytest.raises(NotEncodable):
        extract_content({"question": "q", "image": "a.png",
                         "expected_answer": "a"})


def test_audit_counts_non_encodable_records_separately(tmp_path):
    """Valid VQA records are counted as valid, and reported as not encodable."""
    path = write_jsonl(tmp_path / "vqa.jsonl",
                       [{"question": "q1", "image": "a.png",
                         "expected_answer": "a"}])
    summary = audit_dataset(path, level="encoding")
    assert summary["conforms"] is True
    assert summary["valid"] == 1
    assert summary["invalid"] == 0
    assert summary["not_encodable"] == 1
    assert summary["kinds"] == {KIND_VQA: 1}


def test_mixed_corpus_and_vqa_audit_is_clean_at_encoding_level(tmp_path):
    path = tmp_path / "mixed.jsonl"
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(corpus_record()) + "\n")
        f.write(json.dumps({"question": "q", "image": "a.png",
                            "expected_answer": "a"}) + "\n")
    summary = audit_dataset(path, level="encoding")
    assert summary["conforms"] is True
    assert summary["valid"] == 2
    assert summary["kinds"] == {KIND_CORPUS: 1, KIND_VQA: 1}
    assert summary["not_encodable"] == 1


def test_detect_kind_is_unchanged():
    assert detect_kind(corpus_record()) == KIND_CORPUS
    assert detect_kind({"instruction": "q", "response": "a"}) == KIND_SFT
    assert detect_kind({"question": "q", "image": "a.png",
                        "expected_answer": "a"}) == KIND_VQA


# --------------------------------------------------------------------------- #
# Diagnostics -- missing inputs and empty data must be actionable
# --------------------------------------------------------------------------- #
def test_missing_corpus_is_a_controlled_error_not_a_traceback(tmp_path):
    missing = tmp_path / "absent.jsonl"
    with pytest.raises(SystemExit) as exc:
        encode(missing, tmp_path / "out.bin")
    assert "corpus file not found" in str(exc.value)
    assert str(missing) in str(exc.value)


def test_missing_tokenizer_is_a_controlled_error(tmp_path):
    corpus = write_jsonl(tmp_path / "c.jsonl", [corpus_record()])
    with pytest.raises(SystemExit) as exc:
        encode(corpus, tmp_path / "out.bin", tokenizer=str(tmp_path / "absent.json"))
    assert "tokenizer file not found" in str(exc.value)


def test_unloadable_tokenizer_is_reported_clearly(tmp_path):
    corpus = write_jsonl(tmp_path / "c.jsonl", [corpus_record()])
    corrupt = tmp_path / "bad.json"
    corrupt.write_text("not json at all", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        encode(corpus, tmp_path / "out.bin", tokenizer=str(corrupt))
    assert "could not load tokenizer" in str(exc.value)


def test_audit_distinguishes_missing_from_empty(tmp_path):
    missing = audit_dataset(tmp_path / "absent.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    empty_summary = audit_dataset(empty)

    assert missing["state"] == "missing"
    assert missing["exists"] is False
    assert empty_summary["state"] == "empty"
    assert empty_summary["exists"] is True
    # Both are non-conforming, but for different, stated reasons.
    assert missing["conforms"] is False
    assert empty_summary["conforms"] is False
    assert missing["problems"][0]["error"] == "file does not exist"


def test_audit_marks_audited_state_when_records_exist(tmp_path):
    path = write_jsonl(tmp_path / "c.jsonl", [corpus_record()])
    summary = audit_dataset(path)
    assert summary["state"] == "audited"
    assert summary["conforms"] is True


def test_audit_reads_a_utf8_bom_file(tmp_path):
    """Windows tooling writes a BOM; a BOM is not part of the first record."""
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(corpus_record()).encode())
    summary = audit_dataset(path)
    assert summary["unparsable"] == 0
    assert summary["conforms"] is True


def test_encoder_reports_missing_or_empty_jsonl_clearly(tmp_path):
    from data.encode_dataset import validate_corpus_records

    assert any("does not exist" in p for p in
               validate_corpus_records(tmp_path / "absent.jsonl"))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert any("contains no records" in p for p in validate_corpus_records(empty))
    # Plain text is exempt from JSON validation entirely.
    plain = tmp_path / "prose.txt"
    plain.write_text("hello", encoding="utf-8")
    assert validate_corpus_records(plain) == []