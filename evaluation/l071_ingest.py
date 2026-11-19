"""TAKSH L0.7.1 — corpus ingestion adapters.

Adapters implement a uniform protocol: authorization -> source identity ->
manifest integrity -> record iteration -> deterministic sample records.
Only the internal authorized Python corpus and SYNTHETIC fixtures may be
executed in L0.7.1; every other adapter refuses BEFORE content ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l070_license_qualification import load_decisions  # noqa: E402
from evaluation.l070_source_registry import load_registry  # noqa: E402
from evaluation import l071_corpus_record as cr  # noqa: E402

L04_TRAIN = REPO / "data" / "l04_code_corpus" / "train.jsonl"
L04_MANIFEST = REPO / "data" / "l04_code_corpus" / "dataset_manifest.json"
CANONICAL_TOK = REPO / "tokenizer" / "taksh_code_bpe_v1.json"

REAL = "REAL_AUTHORIZED_CORPUS"
SYNTHETIC = "SYNTHETIC_TEST_FIXTURE"
CANDIDATE = "CANDIDATE_SOURCE"
UNVERIFIED = "UNVERIFIED_SOURCE"
BLOCKED = "BLOCKED_SOURCE"


class IngestRejected(RuntimeError):
    """Unapproved or invalid source — raised before any content is read."""


def tokenizer_fingerprint() -> str:
    return hashlib.sha256(CANONICAL_TOK.read_bytes()).hexdigest()


_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(CANONICAL_TOK))
    return _TOKENIZER


def _measure_tokens(text: str) -> int:
    return len(_tokenizer().encode(text).ids)


def license_decision_for(source_id: str) -> str | None:
    data = load_decisions()
    for record in data.get("records", []):
        if record.get("source_id") == source_id:
            return record.get("overall_decision")
    return None


def internal_python_authorization() -> dict:
    """Verify the internal l04 corpus is approved for L0.7.1 internal research."""
    registry = load_registry()
    src = next((s for s in registry["sources"] if s["source_id"] == "src-taksh-l04-code-corpus"), None)
    problems: list[str] = []
    if src is None:
        problems.append("source src-taksh-l04-code-corpus missing from registry")
        return {"authorized": False, "problems": problems}
    if src.get("acquisition_status") != "ACQUIRED":
        problems.append("internal corpus must be ACQUIRED")
    decision = license_decision_for("src-taksh-l04-code-corpus")
    if decision != "APPROVED_WITH_CONDITIONS":
        problems.append(f"license decision must be APPROVED_WITH_CONDITIONS, got {decision!r}")
    if not L04_TRAIN.is_file():
        problems.append(f"missing corpus file: {L04_TRAIN}")
    if not L04_MANIFEST.is_file():
        problems.append(f"missing corpus manifest: {L04_MANIFEST}")
    return {
        "authorized": not problems,
        "problems": problems,
        "scope": "internal research engineering qualification (L0.7.1)",
        "conditions": "internal research only; no external distribution; benchmark contamination re-screen required; no model artifact is produced",
        "source": src,
    }


def ingest_internal_python() -> list[dict]:
    auth = internal_python_authorization()
    if not auth["authorized"]:
        raise IngestRejected("; ".join(auth["problems"]))

    manifest = json.loads(L04_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("train_sha256") and hashlib.sha256(L04_TRAIN.read_bytes()).hexdigest() != manifest["train_sha256"]:
        raise IngestRejected("train.jsonl SHA-256 does not match dataset_manifest.json — refusing to ingest")

    tok_fp = hashlib.sha256(CANONICAL_TOK.read_bytes()).hexdigest()
    records = []
    with L04_TRAIN.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            row = json.loads(line)
            if row.get("domain") != "code":
                continue                       # English prose is out of scope for the Python code pipeline
            records.append(cr.build_record(
                source_id="src-taksh-l04-code-corpus",
                source_version="train.jsonl @ 2026-11-10 (dataset_manifest.json)",
                original_ref="data/l04_code_corpus/train.jsonl",
                original_index=index,
                text=row["text"],
                language="python",
                language_family="l1_target",
                license_id=row.get("license", "unknown"),
                license_decision_ref="configs/data/l070_license_decisions.yaml#src-taksh-l04-code-corpus",
                provenance_ref="data/l04_code_corpus/dataset_manifest.json",
                tokenizer_id="taksh_code_bpe_v1",
                tokenizer_version="1",
                tokenizer_fingerprint=tok_fp,
                token_count=_measure_tokens(row["text"]),
                snapshot_ref="ungated-local",
            ))
    return records


SYNTHETIC_FIXTURES = [
    {"text": "def synth_add(a, b):\n    return a + b\n", "language": "python", "license": "synthetic"},
    {"text": "fn synth_main() { print(\"synth\"); }\n", "language": "karkain", "license": "synthetic"},
    {"text": "const synthAdd = (a, b) => a + b;\n", "language": "javascript", "license": "synthetic"},
    {"text": "int synth_add(int a, int b) { return a + b; }\n", "language": "c", "license": "synthetic"},
]


def ingest_synthetic_fixtures() -> list[dict]:
    tok_fp = hashlib.sha256(CANONICAL_TOK.read_bytes()).hexdigest()
    records = []
    for index, fixture in enumerate(SYNTHETIC_FIXTURES):
        rec = cr.build_record(
            source_id=f"synthetic-fixture-{index:03d}",
            source_version="l071-synthetic/1",
            original_ref=f"synthetic_fixtures[{index}]",
            original_index=index,
            text=fixture["text"],
            language=fixture["language"],
            language_family="l1_target" if fixture["language"] != "karkain" else "first-class",
            license_id="synthetic",
            license_decision_ref="n/a (synthetic fixtures are never training-eligible)",
            provenance_ref="configs/data/l071_pipeline_config.yaml#synthetic",
            tokenizer_id="taksh_code_bpe_v1",
            tokenizer_version="1",
            tokenizer_fingerprint=tok_fp,
            token_count=_measure_tokens(fixture["text"]),
            synthetic=True,
        )
        rec["exclusion_reason"] = "SYNTHETIC_TEST_FIXTURE"
        records.append(rec)
    return records


def ingest_unapproved(source_id: str) -> list[dict]:
    """Every external/unapproved source refuses BEFORE content ingestion."""
    registry = load_registry()
    src = next((s for s in registry["sources"] if s["source_id"] == source_id), None)
    if src is None:
        raise IngestRejected(f"unknown source: {source_id}")
    status = src.get("acquisition_status")
    if status == "ACQUIRED":
        raise IngestRejected(f"internal source {source_id} must be ingested via ingest_internal_python()")
    raise IngestRejected(f"source {source_id} is {status}; ingestion is blocked until legal review, owner approval and acquisition authorization complete")


def classify_source(source_id: str) -> str:
    if source_id.startswith("synthetic-fixture"):
        return SYNTHETIC
    if source_id == "src-taksh-l04-code-corpus":
        return REAL
    registry = load_registry()
    src = next((s for s in registry["sources"] if s["source_id"] == source_id), None)
    if src is None:
        return UNVERIFIED
    status = src.get("acquisition_status")
    if status == "BLOCKED":
        return BLOCKED
    if status in ("LICENSE_REVIEW", "METADATA_REVIEW", "PROVENANCE_REVIEW",
                  "TECHNICAL_REVIEW", "DEFERRED"):
        return CANDIDATE
    return UNVERIFIED


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_ingest.json")
    parser.add_argument("--source", default="src-taksh-l04-code-corpus")
    args = parser.parse_args(argv)

    auth = internal_python_authorization()
    if args.source == "src-taksh-l04-code-corpus":
        records = ingest_internal_python()
        classification = REAL
    elif args.source.startswith("synthetic"):
        records = ingest_synthetic_fixtures()
        classification = SYNTHETIC
    else:
        try:
            ingest_unapproved(args.source)
            records, classification = [], classify_source(args.source)
        except IngestRejected as exc:
            report = {"schema": "taksh.l071_ingest/1", "valid": False,
                      "source_id": args.source, "classification": classify_source(args.source),
                      "rejected": str(exc)}
            Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"REJECTED: {exc}")
            return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "taksh.l071_ingest/1",
        "source_id": args.source,
        "classification": classification,
        "authorized": auth.get("authorized") if args.source == "src-taksh-l04-code-corpus" else None,
        "records": len(records),
        "python_records": sum(1 for r in records if r["language"] == "python"),
        "synthetic_records": sum(1 for r in records if r["source_id"].startswith("synthetic-fixture")),
        "first_record_keys": list(records[0]) if records else [],
        "eligibility": "NOT_DETERMINED until quality+dedup+contamination+split complete",
    }, indent=2), encoding="utf-8")
    print(f"ingested {len(records)} records ({classification}) from {args.source}")
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
