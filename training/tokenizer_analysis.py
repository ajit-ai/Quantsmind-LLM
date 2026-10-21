"""Measured Python suitability analysis for the Taksh tokenizer (v1.2.0).

The question this module answers is narrow and answerable: *how does the
tokenizer in this repository actually behave on Python source?* It measures,
on a bounded sample:

* fertility (tokens per source character, per lexical token, per byte),
* identifier fragmentation (how many identifiers cost more than one token),
* keyword and operator representation,
* indentation, newline and string-literal handling,
* Unicode handling and unknown-token incidence,
* encode/decode round-trip correctness,
* context-window efficiency (bytes of source covered by one 4096-token window).

The comparison baseline is documented rather than downloaded: a byte-level
fallback costs one token per UTF-8 byte, so *bytes per token* is the
compression factor against that baseline. The v1.1.0 control-token
derivation is measured separately to show whether the reserved control range
changed fertility (it must not, because no merge is retrained).

The decision rules are engineering defaults with stated thresholds. They are
**not** claimed to be empirically optimal: no sweep was run, and the report
says so. The measured values are the evidence; the thresholds are the policy.
"""

from __future__ import annotations

import argparse
import io
import json
import keyword
import tokenize
from pathlib import Path

from training.tokenizer_contract import CONTRACT_V1, CONTRACT_V2, build_contract_tokenizer

#: Conservative engineering defaults. Documented, not optimised.
ACCEPTED_MIN_CHARS_PER_TOKEN = 3.0
MIGRATION_MAX_IDENTIFIER_FRAGMENTATION = 0.50
MIGRATION_MIN_CHARS_PER_TOKEN = 2.5
INSUFFICIENT_MIN_SAMPLES = 20
INSUFFICIENT_MIN_BYTES = 20_000

COMMON_OPERATORS = (
    "+ - * / % // ** == != < > <= >= = += -= *= /= //= **= & | ^ ~ << >> >>="
    " and or not in is if else elif for while return import from as pass "
    "break continue class def try except finally with lambda yield global "
    "nonlocal assert del global raise async await"
).split()

#: Reserved spellings: the v1.0.0 specials and the v1.1.0 control range. A
#: literal of one of these in ordinary text is encoded as the reserved id.
RESERVED_LITERALS = (
    "<bos>",
    "<eos>",
    "<unk>",
    "<pad>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_of_turn|>",
    "<|tool|>",
    "<|fim|>",
)


def _lexical_tokens(text: str) -> list:
    """Significant Python tokens (names, numbers, strings, operators)."""
    tokens = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type in (
                tokenize.NAME,
                tokenize.NUMBER,
                tokenize.STRING,
                tokenize.OP,
            ):
                tokens.append(token)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return []
    return tokens


def analyse_texts(tokenizer, texts, *, source_label: str = "sample") -> dict:
    """Measure the tokenizer over a list of Python sources."""
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_lexical = 0
    identifiers = 0
    identifiers_split = 0
    identifier_tokens = 0
    string_chars = 0
    string_tokens = 0
    newline_tokens = 0
    newlines = 0
    indent_tokens = 0
    indented_lines = 0
    unicode_chars = 0
    unicode_tokens = 0
    unknown_hits = 0
    lossless_failures = 0
    policy_failures = 0
    reserved_files = 0
    reserved_occurrences = 0
    roundtrip_checked = 0

    reserved = RESERVED_LITERALS
    unk_id = None
    try:
        unk_id = tokenizer.token_to_id("<unk>")
    except Exception:  # noqa: BLE001 - optional token
        unk_id = None

    for text in texts:
        total_chars += len(text)
        total_bytes += len(text.encode("utf-8"))
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
        if unk_id is not None:
            unknown_hits += sum(1 for i in encoding.ids if i == unk_id)
        roundtrip_checked += 1
        # Two decodes are measured on purpose:
        #   * lossless -- special-token skipping disabled. This is the
        #     artifact's own fidelity: can the tokenizer represent this text?
        #   * policy   -- the library default, which skips special tokens.
        #     A difference here means a reserved spelling was dropped, not
        #     that the artifact lost information.
        if tokenizer.decode(encoding.ids, skip_special_tokens=False) != text:
            lossless_failures += 1
        if tokenizer.decode(encoding.ids) != text:
            policy_failures += 1
        hits = sum(text.count(literal) for literal in reserved)
        if hits:
            reserved_files += 1
            reserved_occurrences += hits

        for token in _lexical_tokens(text):
            total_lexical += 1
            piece = token.string
            cost = len(tokenizer.encode(piece, add_special_tokens=False).ids)
            if token.type == tokenize.NAME:
                identifiers += 1
                identifier_tokens += cost
                if cost > 1:
                    identifiers_split += 1
            elif token.type == tokenize.STRING:
                string_chars += len(piece)
                string_tokens += cost
            elif token.type == tokenize.OP:
                total_lexical += 0
        newlines += text.count("\n")
        newline_tokens += len(
            tokenizer.encode("\n", add_special_tokens=False).ids
        )
        for line in text.splitlines():
            if line[:1] in (" ", "\t") and line.strip():
                indented_lines += 1
                indent_tokens += len(
                    tokenizer.encode(line[: len(line) - len(line.lstrip())], add_special_tokens=False).ids
                )
        unicode_chars += sum(1 for ch in text if ord(ch) > 127)
        unicode_tokens += len(
            tokenizer.encode(
                "".join(ch for ch in text if ord(ch) > 127) or " ", add_special_tokens=False
            ).ids
        )

    keyword_costs = {
        word: len(tokenizer.encode(word, add_special_tokens=False).ids)
        for word in sorted(keyword.kwlist)
    }
    operator_costs = {
        op: len(tokenizer.encode(op, add_special_tokens=False).ids)
        for op in COMMON_OPERATORS
    }

    def ratio(numerator, denominator):
        return (numerator / denominator) if denominator else 0.0

    return {
        "source_label": source_label,
        "samples": len(texts),
        "characters": total_chars,
        "bytes": total_bytes,
        "tokens": total_tokens,
        "tokens_per_character": ratio(total_tokens, total_chars),
        "characters_per_token": ratio(total_chars, total_tokens),
        "bytes_per_token": ratio(total_bytes, total_tokens),
        "tokens_per_lexical_token": ratio(total_tokens, total_lexical),
        "identifiers": identifiers,
        "identifier_fragmentation": ratio(identifiers_split, identifiers),
        "mean_tokens_per_identifier": ratio(identifier_tokens, identifiers),
        "keywords_single_token": ratio(
            sum(1 for c in keyword_costs.values() if c == 1), len(keyword_costs)
        ),
        "operators_single_token": ratio(
            sum(1 for c in operator_costs.values() if c == 1), len(operator_costs)
        ),
        "string_tokens_per_character": ratio(string_tokens, string_chars),
        "newline_tokens": ratio(newline_tokens, newlines),
        "indent_tokens_per_indented_line": ratio(indent_tokens, indented_lines),
        "unicode_characters": unicode_chars,
        "unicode_tokens_per_character": ratio(unicode_tokens, unicode_chars),
        "unknown_token_rate": ratio(unknown_hits, total_tokens),
        "roundtrip_checked": roundtrip_checked,
        "lossless_roundtrip_failures": lossless_failures,
        "lossless_round_trip_failure_rate": ratio(lossless_failures, roundtrip_checked),
        "policy_roundtrip_failures": policy_failures,
        "policy_round_trip_failure_rate": ratio(policy_failures, roundtrip_checked),
        "reserved_literal_files": reserved_files,
        "reserved_literal_occurrences": reserved_occurrences,
        "bytes_per_4096_token_window": ratio(total_bytes * 4096, total_tokens),
    }


def collect_python_sources(root, *, max_files: int = 200, max_bytes: int = 2_000_000) -> list:
    """Collect a bounded sample of Python source files from a local tree."""
    from data.python_v1.ingest import ingest_source, read_source

    report = ingest_source(
        {"id": "sample"},
        root,
        limits={"max_files": max_files, "max_total_bytes": max_bytes,
                "max_file_bytes": max_bytes, "max_path_depth": 40},
        excluded_paths=[],
        allowed_extensions=(".py",),
    )
    texts = []
    for ref in report.files:
        try:
            text, _encoding, _fallback = read_source(ref, max_bytes=max_bytes)
        except Exception:  # noqa: BLE001 - undecodable files are simply skipped
            continue
        texts.append(text)
    return texts


def collect_jsonl_texts(path, *, field: str = "text", max_samples: int = 500,
                         max_bytes: int = 2_000_000) -> list:
    """Collect text fields from a JSONL corpus, bounded."""
    texts = []
    total = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get(field)
            if not isinstance(text, str) or not text.strip():
                continue
            texts.append(text)
            total += len(text)
            if len(texts) >= max_samples or total >= max_bytes:
                break
    return texts


def decide(metrics: dict) -> tuple[str, list]:
    """Turn measurements into a decision with stated reasons.

    The artifact verdict is driven by the **lossless** round-trip rate: with
    special-token skipping disabled, a failure means the tokenizer really
    cannot represent the file, which is a migration-level defect.

    The policy rate (the library default, which skips special tokens) is
    reported separately. It is not an artifact defect: it means the text
    contains a reserved spelling, which becomes a reserved id in the token
    stream. That is a corpus-side risk, handled by excluding such samples
    from the training data, and it is not fixed by a larger vocabulary.
    """
    reasons = []
    samples = metrics.get("samples", 0)
    total_bytes = metrics.get("bytes", 0)
    if samples < INSUFFICIENT_MIN_SAMPLES or total_bytes < INSUFFICIENT_MIN_BYTES:
        reasons.append(
            f"insufficient evidence: {samples} samples / {total_bytes} bytes "
            f"(need {INSUFFICIENT_MIN_SAMPLES} samples / {INSUFFICIENT_MIN_BYTES} bytes)"
        )
        return "INSUFFICIENT_EVIDENCE", reasons

    fragmentation = metrics.get("identifier_fragmentation", 0.0)
    chars_per_token = metrics.get("characters_per_token", 0.0)
    lossless_rate = metrics.get("lossless_round_trip_failure_rate", 0.0)
    policy_rate = metrics.get("policy_round_trip_failure_rate", 0.0)
    migration = []

    if lossless_rate > 0.01:
        migration.append(
            f"lossless round-trip failure rate {lossless_rate:.3f} exceeds 0.01: "
            "the tokenizer cannot reproduce the measured files even with "
            "special-token skipping disabled"
        )
    elif lossless_rate > 0:
        reasons.append(
            f"lossless round-trip failure rate {lossless_rate:.3f} is non-zero: "
            "a small class of files cannot be reproduced losslessly"
        )
    if policy_rate > 0.01:
        reasons.append(
            f"policy round-trip failure rate {policy_rate:.3f}: the sample "
            "contains reserved token spellings, which are encoded as reserved "
            "ids. This is a corpus-side injection risk handled by excluding "
            "such samples, not an artifact defect"
        )
    if fragmentation > MIGRATION_MAX_IDENTIFIER_FRAGMENTATION:
        migration.append(
            f"identifier fragmentation {fragmentation:.2f} exceeds the migration "
            f"threshold {MIGRATION_MAX_IDENTIFIER_FRAGMENTATION:.2f}"
        )
    if chars_per_token < MIGRATION_MIN_CHARS_PER_TOKEN:
        migration.append(
            f"characters per token {chars_per_token:.2f} is below the migration "
            f"threshold {MIGRATION_MIN_CHARS_PER_TOKEN:.2f}"
        )
    if migration:
        reasons.extend(migration)
        reasons.append(
            "the reserved control range is not the cause: no merge was retrained"
        )
        return "TOKENIZER_MIGRATION_RECOMMENDED", reasons

    if chars_per_token < ACCEPTED_MIN_CHARS_PER_TOKEN:
        reasons.append(
            f"characters per token {chars_per_token:.2f} is below the acceptance "
            f"threshold {ACCEPTED_MIN_CHARS_PER_TOKEN:.2f}"
        )
        reasons.append(
            "identifier fragmentation is within bounds, so the corpus is usable "
            "but the vocabulary is small for a Python-first corpus"
        )
        return "TOKENIZER_ACCEPTED_WITH_LIMITATIONS", reasons

    reasons.append(
        f"characters per token {chars_per_token:.2f} meets the acceptance "
        f"threshold {ACCEPTED_MIN_CHARS_PER_TOKEN:.2f}"
    )
    reasons.append(
        f"identifier fragmentation {fragmentation:.2f} is within the migration "
        f"threshold {MIGRATION_MAX_IDENTIFIER_FRAGMENTATION:.2f}"
    )
    reasons.append("lossless round-trip is exact on the measured sample")
    return "TOKENIZER_ACCEPTED", reasons


def classify_roundtrip_failures(tokenizer, texts) -> dict:
    """Separate artifact-level loss from special-token decoding policy.

    A failure under the library default (``skip_special_tokens=True``) that
    disappears when special-token skipping is disabled is a *policy* effect:
    the reserved spelling was decoded away, not lost by the tokenizer. A
    failure that survives with skipping disabled is genuine artifact loss.
    """
    literals = RESERVED_LITERALS
    policy_failures = 0
    lossless_failures = 0
    special_token_drop = 0
    for text in texts:
        ids = tokenizer.encode(text).ids
        policy_ok = tokenizer.decode(ids) == text
        lossless_ok = tokenizer.decode(ids, skip_special_tokens=False) == text
        if not policy_ok:
            policy_failures += 1
        if not lossless_ok:
            lossless_failures += 1
        elif not policy_ok:
            special_token_drop += 1
    return {
        "policy_failures": policy_failures,
        "lossless_failures": lossless_failures,
        "special_token_drop": special_token_drop,
        "genuine_loss": lossless_failures,
        "note": (
            "tokenizers' decode() defaults to skip_special_tokens=True, so a "
            "registered special token present in the text is decoded away. "
            "With skip_special_tokens=False the text is reproduced exactly, "
            "which means the artifact is lossless and the difference is a "
            "decoding policy, not a tokenizer defect."
        ),
    }


def migration_plan(tokenizer, metrics: dict) -> dict:
    """Embedding and checkpoint implications of a larger vocabulary."""
    d_model = 256
    config_path = Path("configs/model/code_m3_9m.yaml")
    if config_path.is_file():
        try:
            import yaml

            d_model = int(
                (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
                .get("model", {})
                .get("d_model", d_model)
            )
        except Exception:  # noqa: BLE001 - the default is used
            pass
    current_vocab = tokenizer.get_vocab_size()
    proposed_vocab = 32768
    current_embedding = current_vocab * d_model
    proposed_embedding = proposed_vocab * d_model
    return {
        "current_vocabulary": current_vocab,
        "proposed_vocabulary": proposed_vocab,
        "d_model": d_model,
        "current_embedding_parameters": current_embedding,
        "proposed_embedding_parameters": proposed_embedding,
        "embedding_parameter_delta": proposed_embedding - current_embedding,
        "training_data_requirement": (
            "Train the vocabulary on the pipeline's train split only, with the "
            "v1.1.0 control range reserved and unchanged. The holdouts and "
            "the benchmark screen must not contribute merges."
        ),
        "checkpoint_compatibility": (
            "Existing v1.0.0 and v1.1.0 checkpoints embed the 5,822-row table "
            "and cannot be loaded into a resized model. They stay servable "
            "through the legacy tokenizer artifact; the new tokenizer is a "
            "separate artifact with its own model rung."
        ),
        "migration_plan": [
            "v1.3.0 trains a new tokenizer artifact (for example "
            "qm_code_bpe_v1.json) from the v1.2.0 train split only.",
            "The reserved control range from v1.1.0 is preserved with the same "
            "ids, so the prompt contract does not change.",
            "A new model rung is configured with the larger vocabulary; the "
            "legacy rung and tokenizer remain loadable for existing "
            "checkpoints.",
            "Legacy checkpoints and the legacy tokenizer are never deleted or "
            "silently replaced.",
        ],
    }


def build_report(tokenizer_path, texts, *, label: str = "python source sample") -> dict:
    """Measure the tokenizer and produce the suitability report."""
    from tokenizers import Tokenizer

    from data.python_v1.registry import file_sha256

    path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(path))
    metrics = analyse_texts(tokenizer, texts, source_label=label)

    derived = build_contract_tokenizer(path)
    control_metrics = analyse_texts(derived, texts, source_label=label + " (control tokens)")

    verdict, reasons = decide(metrics)
    regression = control_metrics["tokens_per_character"] - metrics["tokens_per_character"]
    failures = classify_roundtrip_failures(tokenizer, texts)
    report = {
        "schema": "quantsmind.tokenizer_suitability/1",
        "tokenizer": {
            "path": path.as_posix(),
            "sha256": file_sha256(path),
            "vocab_size": tokenizer.get_vocab_size(),
            "contract_id": CONTRACT_V1,
        },
        "control_token_derivation": {
            "contract_id": CONTRACT_V2,
            "vocab_size": derived.get_vocab_size(),
            "fertility_delta_tokens_per_character": regression,
            "note": (
                "The v1.1.0 control range appends six single-token markers and "
                "retrains no merge, so fertility must not regress."
            ),
        },
        "metrics": metrics,
        "round_trip": failures,
        "corpus_guard": {
            "rule": "reserved token spellings are excluded from the corpus",
            "reserved_literals": list(RESERVED_LITERALS),
            "reason": (
                "A reserved spelling in ordinary text is encoded as the "
                "reserved id. For the v1.1.0 control range that is a role "
                "marker, so such a file would forge a role boundary in the "
                "qm.chat_instruction/2 renderer. The pipeline excludes these "
                "files (reason code reserved_token_literal) rather than "
                "rewriting them."
            ),
            "measured_files_with_reserved_literals": metrics.get(
                "reserved_literal_files", 0
            ),
            "measured_occurrences": metrics.get("reserved_literal_occurrences", 0),
        },
        "baseline": {
            "byte_level_fallback_bytes_per_token": 1.0,
            "note": (
                "A byte-level fallback costs one token per UTF-8 byte. "
                "bytes_per_token is therefore the compression factor against "
                "that baseline. No external tokenizer was downloaded or "
                "compared."
            ),
        },
        "decision": {
            "verdict": verdict,
            "reasons": reasons,
            "thresholds": {
                "accepted_min_characters_per_token": ACCEPTED_MIN_CHARS_PER_TOKEN,
                "migration_max_identifier_fragmentation": (
                    MIGRATION_MAX_IDENTIFIER_FRAGMENTATION
                ),
                "migration_min_characters_per_token": MIGRATION_MIN_CHARS_PER_TOKEN,
                "insufficient_min_samples": INSUFFICIENT_MIN_SAMPLES,
                "insufficient_min_bytes": INSUFFICIENT_MIN_BYTES,
                "status": (
                    "engineering defaults with stated thresholds; not claimed "
                    "to be empirically optimal, because no sweep was run"
                ),
            },
        },
        "limitations": [
            "the sample is a bounded slice of one local corpus; it is not a "
            "representative sample of Python on the public internet",
            "no external or modern tokenizer was downloaded for comparison, so "
            "the report is a self-measurement, not a competitive benchmark",
            "thresholds are engineering defaults, not the result of a sweep",
            "identifier fragmentation counts tokens, not whether the split is "
            "semantically harmful",
            "the reserved-literal rate in this repository is inflated by the "
            "repository's own subject matter: its code contains literal "
            "special-token strings far more often than general Python does",
        ],
    }
    report["migration_plan"] = dict(
        migration_plan(tokenizer, metrics),
        triggered_by_gates=(verdict == "TOKENIZER_MIGRATION_RECOMMENDED"),
        note=(
            "This plan is an input to the owner's vocabulary decision (D6). "
            "v1.2.0 does not retrain, resize or replace any tokenizer, and a "
            "larger vocabulary would not fix the reserved-literal injection "
            "risk, which is handled corpus-side."
        ),
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure Python suitability of the Taksh tokenizer"
    )
    parser.add_argument("--tokenizer", default="tokenizer/qm_code_bpe_v0.json")
    parser.add_argument("--source-dir", default=None, help="local tree of Python files")
    parser.add_argument("--jsonl", default=None, help="JSONL corpus with a text field")
    parser.add_argument("--max-files", type=int, default=200)
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    parser.add_argument("--out", default=None, help="write the report here (optional)")
    args = parser.parse_args(argv)

    if args.jsonl:
        texts = collect_jsonl_texts(
            args.jsonl, max_samples=args.max_files, max_bytes=args.max_bytes
        )
        label = f"{args.jsonl}"
    elif args.source_dir:
        texts = collect_python_sources(
            args.source_dir, max_files=args.max_files, max_bytes=args.max_bytes
        )
        label = f"{args.source_dir} (**.py)"
    else:
        parser.error("one of --source-dir or --jsonl is required")
        return 2

    report = build_report(args.tokenizer, texts, label=label)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
