"""Unified dataset record schema (M2.4).

Five dataset shapes shipped side by side before this module existed::

    corpus_v0.jsonl          {license, source, text}
    code_corpus_v0.jsonl     {language, source, text}
    code_corpus_v1/v2.jsonl  {text, content_sha1, source_id, repo, commit,
                              spdx, attribution, file_path, retrieved_at,
                              split, language}
    internal_corpus_v0.jsonl {text, source, file_path, git_commit, language}
    code_sft_v0.jsonl        {instruction, response, task, language}

Nothing declared which fields a record needed, so a malformed or
provenance-free record could be encoded into training data unnoticed. This
module declares the contract and validates against it, without rewriting any
existing shard.

Two levels, because two boundaries genuinely differ
----------------------------------------------------
``encoding``
    The contract at the tokenizer boundary (``data.encode_dataset``). A record
    must be a JSON object carrying the text it contributes. Synthetic fixtures
    and legacy prose corpora satisfy this; it says nothing about provenance.

``release``
    The contract for a corpus that may be distributed or trained on. Every
    provenance field must be present and non-empty, the content hash must match
    the text, and the split must be known. ``code_corpus_v2.jsonl`` satisfies
    this; the legacy shards do not, and the audit reports that as a fact rather
    than repairing it.

Use :func:`audit_dataset` to report on an existing file. It never modifies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "quantsmind.dataset/1"

KIND_CORPUS = "corpus"
KIND_SFT = "sft"
KIND_VQA = "vqa"

#: Fields every record of a kind must carry at ``release`` level.
REQUIRED_FIELDS = {
    KIND_CORPUS: (
        "text",
        "content_sha1",
        "source_id",
        "repo",
        "commit",
        "spdx",
        "attribution",
        "file_path",
        "retrieved_at",
        "split",
    ),
    KIND_SFT: ("instruction", "response"),
    KIND_VQA: ("question", "image", "expected_answer"),
}

#: Fields required at ``encoding`` level (the tokenizer boundary), by kind.
#:
#: Only kinds that ``data.encode_dataset`` can actually turn into a token stream
#: appear here. That is deliberately narrow:
#:
#: * ``corpus`` -- its ``text`` field *is* the training content.
#: * ``sft``    -- consumed by ``training.sft.py``, which renders the chat
#:   template over instruction/response and masks the prompt. Encoding the raw
#:   record here would bypass that template and train on the wrong string.
#: * ``vqa``    -- carries an ``image`` path; a text tokenizer cannot represent
#:   it. There is no VQA token-stream format in this repository, and inventing
#:   one would be a design decision, not a bug fix.
ENCODING_REQUIRED = {KIND_CORPUS: ("text",)}

#: Kinds this encoder can represent. A schema-valid record of any other kind is
#: *valid* but is not an encoding path.
ENCODABLE_KINDS = frozenset(ENCODING_REQUIRED)

VALID_SPLITS = ("train", "val", "test")

#: Signature fields that identify a record kind when it is not declared.
_KIND_SIGNATURES = {
    KIND_CORPUS: ("text",),
    KIND_SFT: ("instruction", "response"),
    KIND_VQA: ("question", "image"),
}


class SchemaError(ValueError):
    """A record does not satisfy the schema it claims."""


def detect_kind(record):
    """Infer the record kind from its fields. ``None`` when unrecognisable."""
    if not isinstance(record, dict):
        return None
    keys = set(record)
    for kind, signature in _KIND_SIGNATURES.items():
        if set(signature) <= keys:
            return kind
    return None


def validate_record(record, kind=None, level="release"):
    """Return a list of problems; an empty list means the record conforms.

    ``level`` is ``"encoding"`` or ``"release"``. Never raises, so a caller can
    report every problem in a file in one pass.
    """
    if level not in ("encoding", "release"):
        raise ValueError(f"unknown level: {level!r}")

    if not isinstance(record, dict):
        return [f"record is {type(record).__name__}, not a JSON object"]

    if kind is None:
        kind = detect_kind(record)
        if kind is None:
            return [
                "cannot determine record kind: expected one of "
                + ", ".join(f"{k}({'+'.join(s)})" for k, s in _KIND_SIGNATURES.items())
            ]

    problems = []
    if level == "encoding" and kind not in ENCODING_REQUIRED:
        # A schema-valid record of a kind this encoder cannot represent is not a
        # deviation: it is simply not an encoding path. Release-level rules below
        # still apply.
        return []

    required = (
        REQUIRED_FIELDS[kind]
        if level == "release"
        else ENCODING_REQUIRED.get(kind, ("text",))
    )
    for field in required:
        if field not in record:
            problems.append(f"missing required field '{field}' ({kind}/{level})")
        elif not isinstance(record[field], str) or not record[field].strip():
            problems.append(
                f"field '{field}' must be a non-empty string, got "
                f"{type(record[field]).__name__}"
            )

    if kind == KIND_CORPUS and level == "release":
        text = record.get("text")
        digest = record.get("content_sha1")
        if isinstance(text, str) and isinstance(digest, str) and digest:
            actual = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if actual != digest:
                problems.append(
                    f"content_sha1 does not match text (recorded {digest[:12]}..., "
                    f"actual {actual[:12]}...)"
                )
        split = record.get("split")
        if isinstance(split, str) and split and split not in VALID_SPLITS:
            problems.append(
                f"split '{split}' is not one of {', '.join(VALID_SPLITS)}"
            )
    return problems


#: The field that holds the training content, per encodable kind. The encoder
#: extracts this and tokenizes it, so JSON syntax, ids, hashes and provenance are
#: never trained on.
CONTENT_FIELD = {KIND_CORPUS: "text"}


def content_field_for(kind):
    """Name of the field holding training content for ``kind``, or ``None``."""
    return CONTENT_FIELD.get(kind)


class NotEncodable(ValueError):
    """A schema-valid record of a kind this encoder cannot represent."""

    def __init__(self, kind, consumer_hint):
        super().__init__(
            f"record kind {kind!r} is not a supported path for "
            "data.encode_dataset; it has no text token-stream representation "
            f"in this repository. {consumer_hint}"
        )
        self.kind = kind


#: Where each non-encodable kind is actually consumed.
KIND_CONSUMER = {
    KIND_SFT: (
        "SFT records are consumed by training/sft.py, which renders the chat "
        "template over instruction/response and masks the prompt."
    ),
    KIND_VQA: (
        "VQA records carry an 'image' path and are consumed by the multimodal "
        "evaluation path (evaluation/eval_vqa.py). No VQA token-stream format "
        "exists; defining one would be a design decision."
    ),
}


def extract_content(record, kind=None, allow_empty=False):
    """Return the training text for an encodable record.

    Raises :class:`NotEncodable` for a valid record of a kind this encoder does
    not support, so a caller can never silently train on the raw JSON line.

    ``allow_empty=True`` returns ``""`` for an empty content field instead of
    raising. The encoder uses that to *skip* records with no content, mirroring
    how blank lines are skipped in a plain-text corpus; aborting there would
    change the documented warn/``--strict``/``--no-validate`` policy.
    """
    kind = kind or detect_kind(record)
    field = CONTENT_FIELD.get(kind)
    if field is None:
        raise NotEncodable(kind or "unknown", KIND_CONSUMER.get(kind, ""))
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(
            f"field {field!r} must be a string, got {type(value).__name__}"
        )
    if not value.strip() and not allow_empty:
        raise ValueError(f"field {field!r} is empty or not a string")
    return value


def iter_records(path):
    """Yield ``(line_number, raw_line)`` for non-empty lines."""
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for number, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped:
                yield number, stripped


def audit_dataset(path, kind=None, level="release", limit=20):
    """Report schema conformance for a JSONL dataset. Read-only.

    Returns a summary dict; ``problems`` is capped at ``limit`` entries while
    ``invalid`` is counted in full.

    ``state`` distinguishes the cases that were previously indistinguishable in
    the printed output:

    ``missing``
        the path does not exist
    ``empty``
        the file exists but holds no records
    ``audited``
        records were read
    """
    path = Path(path)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "path": str(path),
        "level": level,
        "exists": path.exists(),
        "state": "missing" if not path.exists() else "empty",
        "records": 0,
        "valid": 0,
        "invalid": 0,
        "unparsable": 0,
        "not_encodable": 0,
        "kinds": {},
        "missing_fields": {},
        "problems": [],
        "conforms": False,
    }
    if not path.exists():
        summary["problems"].append({"line": 0, "error": "file does not exist"})
        return summary

    for number, raw in iter_records(path):
        summary["records"] += 1
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            summary["unparsable"] += 1
            summary["invalid"] += 1
            if len(summary["problems"]) < limit:
                summary["problems"].append(
                    {"line": number, "error": f"invalid JSON: {exc.msg}"}
                )
            continue

        record_kind = kind or detect_kind(record)
        label = record_kind or "unknown"
        summary["kinds"][label] = summary["kinds"].get(label, 0) + 1
        if level == "encoding" and record_kind and record_kind not in ENCODABLE_KINDS:
            # Valid record, but this encoder cannot represent its kind.
            summary["not_encodable"] += 1
            summary["valid"] += 1
            continue
        problems = validate_record(record, kind=record_kind, level=level)
        if problems:
            summary["invalid"] += 1
            for problem in problems:
                field = problem.split("'")[1] if "'" in problem else "?"
                summary["missing_fields"][field] = (
                    summary["missing_fields"].get(field, 0) + 1
                )
            if len(summary["problems"]) < limit:
                summary["problems"].append({"line": number, "error": problems[0]})
        else:
            summary["valid"] += 1

    summary["conforms"] = summary["invalid"] == 0 and summary["records"] > 0
    summary["state"] = "audited" if summary["records"] else "empty"
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit JSONL datasets against the unified record schema"
    )
    parser.add_argument("datasets", nargs="+", help="JSONL files to audit")
    parser.add_argument("--kind", choices=sorted(REQUIRED_FIELDS), default=None)
    parser.add_argument(
        "--level", choices=("encoding", "release"), default="release"
    )
    args = parser.parse_args(argv)

    deviating = 0
    for path in args.datasets:
        summary = audit_dataset(path, kind=args.kind, level=args.level)
        state = summary["state"]
        if state == "missing":
            verdict = "MISSING"
        elif state == "empty":
            verdict = "EMPTY"
        elif summary["conforms"]:
            verdict = "CONFORMS"
        else:
            verdict = "DEVIATES"
        print(f"{path}")
        print(
            f"  {verdict}: {summary['valid']}/{summary['records']} valid"
            f"  kinds={summary['kinds']}"
        )
        if summary["missing_fields"]:
            print(f"  missing_fields={summary['missing_fields']}")
        if summary["not_encodable"]:
            print(
                "  not_encodable="
                f"{summary['not_encodable']} (schema-valid, but no text "
                "token-stream path exists for that kind)"
            )
        for problem in summary["problems"][:5]:
            where = f"line {problem['line']}: " if problem["line"] else ""
            print(f"  {where}{problem['error']}")
        if verdict != "CONFORMS":
            deviating += 1
    return 0 if deviating == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())