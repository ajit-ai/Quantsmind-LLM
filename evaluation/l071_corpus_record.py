"""TAKSH L0.7.1 — canonical corpus record.

Builds, validates and fingerprints the immutable corpus record (schema
taksh.corpus_record/1). Provenance fields are set at creation and never
rewritten; normalization appends transformation history entries.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

PIPELINE_VERSION = "l071-pipeline/1"
NORMALIZER_VERSION = "l071-norm/1"

QUALITY_STATES = frozenset(
    ["VALIDATED", "VALIDATED_WITH_WARNINGS", "NO_AUTOMATED_FLAGS",
     "HUMAN_REVIEW_REQUIRED", "INVALID", "BLOCKED", "NOT_CHECKED"]
)
DUPLICATE_STATES = frozenset(["UNIQUE", "EXACT_DUPLICATE", "NEAR_DUPLICATE", "NOT_SCANNED"])
CONTAMINATION_STATES = frozenset(["CLEAN", "CONTAMINATED", "NOT_SCANNED"])
ELIGIBILITY_STATES = frozenset(["ELIGIBLE", "INELIGIBLE", "NOT_DETERMINED"])
SPLITS = frozenset(["train", "val", "test", "unassigned"])

REQUIRED_FIELDS = [
    "sample_id", "source_id", "source_version", "snapshot_ref", "original_ref",
    "original_index", "original_fingerprint", "normalized_fingerprint", "language",
    "language_family", "license_id", "license_decision_ref", "provenance_ref",
    "content_format", "encoding", "tokenizer_id", "tokenizer_version",
    "tokenizer_fingerprint", "token_count", "sample_count", "quality_status",
    "duplicate_status", "duplicate_group", "contamination_status",
    "training_eligibility", "split", "exclusion_reason",
    "pipeline_version", "normalization_version", "processing_timestamp",
    "transformation_history",
]


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def original_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_text(text: str) -> tuple[str, int]:
    """l071-norm/1 universal rules, code-safe (no semantic destruction).

    Returns (normalized_text, interior_feff_dropped).
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = t.lstrip("﻿")                          # BOM_STRIP
    interior = t.count("﻿")
    t = t.replace("﻿", "")                      # INTERIOR_FEFF_DROP
    t = re.sub(r"[ \t]+\n", "\n", t)              # TRAILING_WS_TRIM
    t = t.rstrip("\n") + "\n"                     # FINAL_NEWLINE
    return t, interior


def normalized_fingerprint(text: str) -> str:
    return hashlib.sha256(normalized_text(text)[0].encode("utf-8")).hexdigest()


def make_sample_id(source_id: str, original_fp: str, index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"taksh:{source_id}:{original_fp}:{index}"))


def build_record(*, source_id: str, source_version: str, original_ref: str,
                 original_index: int, text: str, language: str, language_family: str,
                 license_id: str, license_decision_ref: str, provenance_ref: str,
                 tokenizer_id: str, tokenizer_version: str, tokenizer_fingerprint: str,
                 token_count: int, snapshot_ref: str = "ungated-local",
                 content_format: str = "utf-8 text", encoding: str = "utf-8",
                 synthetic: bool = False) -> dict:
    ofp = original_fingerprint(text)
    normalized, feff_dropped = normalized_text(text)
    nfp = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    history = [
        {"step": "ingest", "version": PIPELINE_VERSION, "detail": f"index={original_index}"},
        {"step": "normalize", "version": NORMALIZER_VERSION, "detail": "BOM_STRIP, LINE_ENDINGS_LF, TRAILING_WS_TRIM, FINAL_NEWLINE (l071-norm/1; code-safe; indentation untouched)"},
    ]
    if feff_dropped:
        history.append({"step": "INTERIOR_FEFF_DROP", "version": NORMALIZER_VERSION,
                        "detail": f"removed {feff_dropped} mid-file U+FEFF (encoding artifact)"})
    record = {
        "sample_id": make_sample_id(source_id, ofp, original_index),
        "source_id": source_id,
        "source_version": source_version,
        "snapshot_ref": snapshot_ref,
        "original_ref": original_ref,
        "original_index": int(original_index),
        "original_fingerprint": ofp,
        "normalized_fingerprint": nfp,
        "language": language,
        "language_family": language_family,
        "supporting_technology": None,
        "license_id": license_id,
        "license_decision_ref": license_decision_ref,
        "provenance_ref": provenance_ref,
        "content_format": content_format,
        "encoding": encoding,
        "tokenizer_id": tokenizer_id,
        "tokenizer_version": tokenizer_version,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "token_count": int(token_count),
        "sample_count": 1,
        "quality_status": "NOT_CHECKED",
        "duplicate_status": "NOT_SCANNED",
        "duplicate_group": None,
        "contamination_status": "NOT_SCANNED",
        "training_eligibility": "INELIGIBLE" if synthetic else "NOT_DETERMINED",
        "split": "unassigned",
        "exclusion_reason": "SYNTHETIC_TEST_FIXTURE" if synthetic else None,
        "pipeline_version": PIPELINE_VERSION,
        "normalization_version": NORMALIZER_VERSION,
        "processing_timestamp": utcnow(),
        "transformation_history": history,
    }
    return record


def validate_record(record: dict) -> list[str]:
    problems: list[str] = []
    nullable = {"exclusion_reason", "duplicate_group"}
    for field_name in REQUIRED_FIELDS:
        if field_name not in record:
            problems.append(f"missing field: {field_name}")
        elif record[field_name] is None and field_name not in nullable:
            problems.append(f"field is null but required: {field_name}")
    if problems:
        return problems

    if record["quality_status"] not in QUALITY_STATES:
        problems.append(f"quality_status invalid: {record['quality_status']!r}")
    if record["duplicate_status"] not in DUPLICATE_STATES:
        problems.append(f"duplicate_status invalid: {record['duplicate_status']!r}")
    if record["contamination_status"] not in CONTAMINATION_STATES:
        problems.append(f"contamination_status invalid: {record['contamination_status']!r}")
    if record["training_eligibility"] not in ELIGIBILITY_STATES:
        problems.append(f"training_eligibility invalid: {record['training_eligibility']!r}")
    if record["split"] not in SPLITS:
        problems.append(f"split invalid: {record['split']!r}")
    if not isinstance(record["token_count"], int) or record["token_count"] < 0:
        problems.append("token_count must be a non-negative integer")
    for fp_field in ("original_fingerprint", "normalized_fingerprint", "tokenizer_fingerprint"):
        if len(str(record[fp_field])) != 64:
            problems.append(f"{fp_field} must be a 64-char hex digest")
    if record["source_id"].startswith("synthetic-fixture") and record["training_eligibility"] != "INELIGIBLE":
        problems.append("synthetic fixtures must be training-INELIGIBLE")
    if record["exclusion_reason"] and not record["sample_id"]:
        problems.append("excluded records remain identifiable (sample_id required)")
    if not isinstance(record.get("transformation_history"), list) or not record["transformation_history"]:
        problems.append("transformation_history must be a non-empty list")
    return problems


def record_fingerprint(record: dict) -> str:
    """Snapshot-stable fingerprint: excludes wall-clock and snapshot fields."""
    stable = {k: v for k, v in record.items()
              if k not in ("processing_timestamp", "snapshot_ref")}
    blob = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()
