"""Encode a corpus to a packed token stream with explicit document boundaries.

Previously every JSONL line was concatenated with no separator, while
``TokenDataset.batch`` samples random offsets across the whole stream. A training
window therefore routinely straddled two unrelated files with nothing marking
the join, teaching the model that any document may continue into any other
(``OPEN_ISSUES`` B6).

The separator is ``<eos>``, a token the shipped tokenizer already defines and
that the serving stack already stops on, so preprocessing, training and inference
agree. See ``training/document_separator.py`` for the full rationale and for the
documented behaviour of records longer than ``seq_len``.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.document_separator import (
    DOC_SEPARATOR_NAME,
    SeparatorUnavailable,
    encode_records,
    resolve_separator_id,
)

DATASET_SCHEMA_VERSION = "quantsmind.dataset/1"


def iter_plain_lines(path: Path):
    """Yield each non-empty line verbatim.

    Used for plain-text corpora (``.txt``): a line *is* the content, so the
    existing behaviour is preserved exactly.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield stripped


def iter_records(path: Path):
    """Yield the training content of a corpus, one document per non-empty line.

    Plain-text corpora yield the line itself. JSONL corpora are parsed and the
    declared content field is extracted, so JSON syntax, ids, hashes and
    provenance are never tokenized as training text.
    """
    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        yield from iter_plain_lines(path)
        return

    from data.schema import NotEncodable, detect_kind, extract_content

    with path.open("r", encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{number}: malformed JSON record: {exc.msg}. "
                    "Refusing to tokenize a raw JSON line as training text."
                ) from exc
            kind = detect_kind(record)
            try:
                content = extract_content(record, kind=kind, allow_empty=True)
            except NotEncodable as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            if not content.strip():
                # Nothing to encode. Skipped, exactly as a blank line is in a
                # plain-text corpus; the schema warning (or --strict refusal)
                # still governs the run.
                continue
            yield content


def validate_corpus_records(path: Path, max_reported: int = 5):
    """Check a JSONL corpus against the schema's ``encoding`` level.

    Returns a list of human-readable problems, capped at ``max_reported`` so a
    badly broken file does not produce an unreadable wall of text. The count of
    offending lines is always included in the first entry.

    This is the *encoding* contract only (a record must carry its content
    field). Provenance and licence are enforced where they are produced -- by
    ``data.build_code_corpus_v1`` -- not at the tokenizer boundary, so a legacy
    prose corpus remains encodable while still being audited.
    """
    from data.schema import audit_dataset

    path = Path(path)
    if path.suffix.lower() != ".jsonl":
        return []
    summary = audit_dataset(path, level="encoding")
    if summary["conforms"]:
        return []
    if summary["state"] == "missing":
        return [f"{path} does not exist"]
    if summary["state"] == "empty":
        return [f"{path} contains no records"]
    problems = [
        f"{summary['invalid']} of {summary['records']} records fail the "
        f"encoding schema ({summary['unparsable']} unparsable)"
    ]
    for problem in summary["problems"][:max_reported]:
        problems.append(f"  line {problem['line']}: {problem['error']}")
    if summary["invalid"] > max_reported:
        problems.append(
            f"  ... and more; run `python -m data.schema {path} --level encoding`"
        )
    return problems


def main():
    p = argparse.ArgumentParser(
        description="Encode a text corpus to packed token ids with doc separators"
    )
    p.add_argument("--corpus", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument(
        "--no-separator",
        action="store_true",
        help="reproduce the pre-B6 concatenation behaviour (diagnostics only)",
    )
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="skip dataset schema validation at the encoding boundary",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of warning when records violate the encoding schema",
    )
    args = p.parse_args()

    corpus = Path(args.corpus)
    tokenizer_path = Path(args.tokenizer)
    for label, candidate in (("corpus", corpus), ("tokenizer", tokenizer_path)):
        if not candidate.exists():
            raise SystemExit(
                f"{label} file not found: {candidate}\n"
                "Check the path, or generate the artifact first "
                "(python -m data.encode_dataset --help)."
            )
    try:
        tok = Tokenizer.from_file(args.tokenizer)
    except Exception as exc:  # noqa: BLE001 - surface any tokenizer load failure
        raise SystemExit(
            f"could not load tokenizer {tokenizer_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # A malformed record must not silently become training tokens. Plain-text
    # corpora (.txt) are exempt: they carry no JSON structure to validate.
    if not args.no_validate:
        problems = validate_corpus_records(corpus)
        if problems and args.strict:
            raise SystemExit(
                f"{corpus} does not satisfy the dataset encoding schema:\n"
                + "\n".join(problems)
                + "\nRefusing to encode. Fix the file, or pass --no-validate "
                "to bypass deliberately."
            )
        for problem in problems:
            print(f"WARNING: {problem}")

    separator_id = None
    if not args.no_separator:
        try:
            separator_id = resolve_separator_id(tok)
        except SeparatorUnavailable as exc:
            raise SystemExit(str(exc)) from exc

    if args.no_separator:
        ids: list[int] = []
        emitted = 0
        for record in iter_records(corpus):
            if args.max_tokens and len(ids) >= args.max_tokens:
                break
            ids.extend(tok.encode(record).ids)
            emitted += 1
            if args.max_tokens and len(ids) > args.max_tokens:
                ids = ids[: args.max_tokens]
                break
    else:
        try:
            ids, emitted = encode_records(
                tok, iter_records(corpus), max_tokens=args.max_tokens
            )
        except ValueError as exc:
            # Malformed JSONL, an unsupported record kind, or an empty content
            # field. Report it as a CLI error rather than a traceback: the
            # message already names the file, line and reason.
            raise SystemExit(str(exc)) from exc

    max_id = max(ids) if ids else 0
    dtype = np.uint16 if max_id < 65536 else np.uint32
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=dtype).tofile(out)

    meta_path = out.with_suffix(".meta.json")
    corpus_sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    tokenizer_sha = hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
    meta = {
        "schema": "quantsmind.token_stream/1",
        "dataset_schema": DATASET_SCHEMA_VERSION,
        "dataset_schema_validated": not args.no_validate,
        "vocab_size": tok.get_vocab_size(),
        "num_tokens": len(ids),
        "num_documents": emitted,
        "dtype": np.dtype(dtype).name,
        "corpus": str(corpus),
        "corpus_sha256": corpus_sha,
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": tokenizer_sha,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "max_tokens": args.max_tokens,
        "representation": (
            "flat stream of uint16/uint32 token ids with document separators; "
            "not pre-packed sequences. training.data.TokenDataset samples random "
            "windows of length seq_len from this stream."
        ),
        "document_separator": (
            None if args.no_separator else DOC_SEPARATOR_NAME
        ),
        "document_separator_id": separator_id,
        "separator_policy": (
            "none (pre-B6 behaviour)"
            if args.no_separator
            else "<eos> inserted between consecutive documents; never leading, "
            "never duplicated, never trailing"
        ),
        "oversized_document_policy": (
            "documents longer than seq_len are written whole and span "
            "consecutive training windows; they are never truncated"
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"tokens={len(ids)} docs={emitted} vocab={meta['vocab_size']} "
        f"dtype={meta['dtype']} separator={meta['document_separator']} -> {out}"
    )


if __name__ == "__main__":
    main()
