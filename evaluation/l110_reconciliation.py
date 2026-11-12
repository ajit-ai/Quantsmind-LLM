"""Reconcile the L1.1.0 foundation-config hash without rewriting history.

The historical report ``reports/l110_native_model_foundation.json`` records
``fd96…`` for the foundation config. On a Windows checkout with CRLF line
endings the same file hashes to ``744e…``. Those are the same content: the
difference is line-ending representation, not a configuration change.

This module records both hashes, classifies the cause, and leaves the
historical report byte-for-byte untouched. It never trains, creates a
checkpoint, uploads, or contacts a network service.

Run:

    python -m evaluation.l110_reconciliation
    python -m evaluation.l110_reconciliation --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HISTORICAL_REPORT = "reports/l110_native_model_foundation.json"
FOUNDATION_CONFIG = "configs/model/foundation_111m.yaml"
TOKENIZER = "tokenizer/qm_code_bpe_v0.json"
DEFAULT_OUT = "reports/l110_config_hash_reconciliation.json"

ALLOWED_STATUSES = ("PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED")


class ReconciliationError(RuntimeError):
    """The reconciliation inputs are missing or malformed."""


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_lf(path: Path) -> str:
    return _sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def _line_endings(data: bytes) -> str:
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        return "MIXED"
    if crlf:
        return "CRLF"
    return "LF"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReconciliationError(f"file not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ReconciliationError(
            f"unreadable JSON in {path}: {type(exc).__name__}"
        ) from exc
    if not isinstance(document, dict):
        raise ReconciliationError(f"expected a JSON object in {path}")
    return document


def _artifact(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": _relative(path),
        "sha256_raw": _sha256_bytes(data),
        "sha256_lf": _sha256_bytes(data.replace(b"\r\n", b"\n")),
        "line_endings": _line_endings(data),
        "bytes": len(data),
    }


def reconcile(
    *,
    historical_report: str | Path = HISTORICAL_REPORT,
    foundation_config: str | Path = FOUNDATION_CONFIG,
    tokenizer: str | Path = TOKENIZER,
) -> dict[str, Any]:
    """Return a deterministic reconciliation record; never writes files."""
    report_path = _path(historical_report)
    config_path = _path(foundation_config)
    tokenizer_path = _path(tokenizer)

    historical = _load_json(report_path)
    try:
        fingerprints = historical["tokenizer"]["fingerprints"]
        historical_config = fingerprints["foundation_config_sha256"]
        historical_tokenizer = fingerprints["tokenizer_sha256"]
        historical_method = fingerprints.get("tokenizer_method", "unknown")
    except (KeyError, TypeError) as exc:
        raise ReconciliationError(
            "historical report is missing tokenizer fingerprint fields"
        ) from exc

    config = _artifact(config_path)
    token = _artifact(tokenizer_path)

    config_matches_lf = config["sha256_lf"] == historical_config
    config_matches_raw = config["sha256_raw"] == historical_config
    tokenizer_matches_raw = token["sha256_raw"] == historical_tokenizer
    tokenizer_matches_lf = token["sha256_lf"] == historical_tokenizer

    if config_matches_lf and not config_matches_raw:
        config_cause = "LINE_ENDING_NORMALISATION"
    elif config_matches_raw:
        config_cause = "NO_HASH_DIFFERENCE"
    else:
        config_cause = "CONTENT_DIFFERENCE"

    # The historical tokenizer method was raw-byte hashing, so raw equality is
    # the like-for-like comparison. LF equality is recorded for cross-platform
    # diagnosis only.
    tokenizer_cause = (
        "NO_HASH_DIFFERENCE"
        if tokenizer_matches_raw
        else "LINE_ENDING_NORMALISATION"
        if tokenizer_matches_lf
        else "CONTENT_DIFFERENCE"
    )

    findings = {
        "config_content_drift": not config_matches_lf,
        "tokenizer_content_drift": not tokenizer_matches_raw,
        "config_hash_cause": config_cause,
        "tokenizer_hash_cause": tokenizer_cause,
        "historical_evidence_rewritten": False,
        "hash_method_risk": (
            "Raw-byte SHA-256 is line-ending sensitive. The historical "
            "foundation-config hash matches LF-normalised bytes, while the "
            "working-tree raw hash is CRLF. Recompute both forms when "
            "comparing evidence across checkouts."
        ),
        "recommended_future_method": (
            "Record raw and LF-normalised SHA-256 side by side, or use the git "
            "blob id, for every evidence artifact."
        ),
    }

    content_clean = not (
        findings["config_content_drift"] or findings["tokenizer_content_drift"]
    )
    status = "PARTIAL"  # hash method remains platform-sensitive, even if clean
    if status not in ALLOWED_STATUSES:
        raise ReconciliationError(f"invalid status: {status}")

    return {
        "schema": "taksh.l110_config_hash_reconciliation/1",
        "generated_by": "evaluation/l110_reconciliation.py",
        "recorded_on": "2026-11-12",
        "historical_evidence": {
            "path": _relative(report_path),
            "sha256_lf": _sha256_lf(report_path),
            "recorded_foundation_config_sha256": historical_config,
            "recorded_tokenizer_sha256": historical_tokenizer,
            "recorded_hash_method": historical_method,
            "rewritten": False,
        },
        "current_artifacts": {
            "foundation_config": {
                **config,
                "matches_historical_raw": config_matches_raw,
                "matches_historical_lf": config_matches_lf,
            },
            "tokenizer": {
                **token,
                "matches_historical_raw": tokenizer_matches_raw,
                "matches_historical_lf": tokenizer_matches_lf,
            },
        },
        "findings": findings,
        "status": status,
        "verdict": (
            "HISTORICAL_EVIDENCE_PRESERVED"
            if content_clean
            else "CONTENT_DRIFT_REQUIRES_OWNER_REVIEW"
        ),
        "policy": {
            "training_authorized": False,
            "compute_authorized": False,
            "publication_authorized": False,
            "historical_reports_are_immutable": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help=f"reconciliation report path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--json", action="store_true", help="print the full report")
    args = parser.parse_args(argv)

    report = reconcile()
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        config = report["current_artifacts"]["foundation_config"]
        findings = report["findings"]
        print(f"status: {report['status']}")
        print(f"config raw: {config['sha256_raw']}")
        print(f"config LF:  {config['sha256_lf']}")
        print(f"config cause: {findings['config_hash_cause']}")
        print(f"config content drift: {findings['config_content_drift']}")
        print(f"historical evidence rewritten: {findings['historical_evidence_rewritten']}")
        print(f"report: {out_path}")
    return 0 if report["verdict"] == "HISTORICAL_EVIDENCE_PRESERVED" else 2


if __name__ == "__main__":
    sys.exit(main())
