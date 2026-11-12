"""Focused tests for the L1.1.0 config-hash reconciliation record.

These tests are read-only with respect to historical evidence. They never
train, create checkpoints, upload, or contact a network service.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reconciliation_records_line_ending_cause_without_content_drift():
    from evaluation.l110_reconciliation import reconcile

    report = reconcile()
    config = report["current_artifacts"]["foundation_config"]
    tokenizer = report["current_artifacts"]["tokenizer"]

    assert config["matches_historical_lf"] is True
    assert report["findings"]["config_content_drift"] is False
    assert report["findings"]["config_hash_cause"] in {
        "LINE_ENDING_NORMALISATION",
        "NO_HASH_DIFFERENCE",
    }
    if config["sha256_raw"] != config["sha256_lf"]:
        assert report["findings"]["config_hash_cause"] == "LINE_ENDING_NORMALISATION"

    assert tokenizer["matches_historical_raw"] is True
    assert report["findings"]["tokenizer_content_drift"] is False
    assert report["findings"]["tokenizer_hash_cause"] == "NO_HASH_DIFFERENCE"


def test_historical_report_is_byte_for_byte_unchanged():
    from evaluation.l110_reconciliation import HISTORICAL_REPORT, reconcile

    path = REPO / HISTORICAL_REPORT
    before = path.read_bytes()

    report = reconcile()

    assert path.read_bytes() == before
    assert report["historical_evidence"]["rewritten"] is False
    assert report["findings"]["historical_evidence_rewritten"] is False
    assert report["historical_evidence"]["sha256_lf"] == hashlib.sha256(
        before.replace(b"\r\n", b"\n")
    ).hexdigest()


def test_reconciliation_is_deterministic():
    from evaluation.l110_reconciliation import reconcile

    assert reconcile() == reconcile()


def test_report_schema_status_and_policy_are_honest():
    from evaluation.l110_reconciliation import reconcile

    report = reconcile()
    assert report["schema"] == "taksh.l110_config_hash_reconciliation/1"
    assert report["status"] in ALLOWED
    assert report["verdict"] == "HISTORICAL_EVIDENCE_PRESERVED"
    assert report["policy"] == {
        "training_authorized": False,
        "compute_authorized": False,
        "publication_authorized": False,
        "historical_reports_are_immutable": True,
    }


def test_cli_writes_report_and_does_not_touch_history(tmp_path, capsys):
    from evaluation.l110_reconciliation import HISTORICAL_REPORT, main

    historical = REPO / HISTORICAL_REPORT
    before = historical.read_bytes()
    out = tmp_path / "reconciliation.json"

    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 0
    assert out.is_file()
    assert "config cause:" in captured.out
    assert historical.read_bytes() == before
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.l110_config_hash_reconciliation/1"


def test_reconcile_detects_real_content_drift(tmp_path):
    from evaluation.l110_reconciliation import (
        FOUNDATION_CONFIG,
        HISTORICAL_REPORT,
        reconcile,
    )

    historical = tmp_path / "historical.json"
    shutil.copyfile(REPO / HISTORICAL_REPORT, historical)
    config = tmp_path / "foundation.yaml"
    config.write_text(
        (REPO / FOUNDATION_CONFIG).read_text(encoding="utf-8")
        + "\n# deliberate test-only content change\n",
        encoding="utf-8",
    )

    report = reconcile(
        historical_report=historical,
        foundation_config=config,
    )

    assert report["findings"]["config_content_drift"] is True
    assert report["findings"]["config_hash_cause"] == "CONTENT_DIFFERENCE"
    assert report["verdict"] == "CONTENT_DRIFT_REQUIRES_OWNER_REVIEW"
