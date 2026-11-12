"""Focused tests for the five-version program gate matrix.

These tests are read-only with respect to release state. They never train,
create checkpoints, upload, tag, release or merge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def test_gate_matrix_separates_program_labels_from_package_versions():
    from evaluation.version_gates import evaluate
    from taksh import __version__

    report = evaluate()

    assert report["schema"] == "taksh.version_gates/1"
    assert report["package"]["current_version"] == __version__ == "1.0.0"
    assert report["policy"]["program_label_is_not_package_version"] is True
    assert report["policy"]["master_roadmap_remains_canonical"] is True
    for version in report["versions"]:
        assert version["kind"] == "program_label"
        assert version["status"] in ALLOWED
        assert version["package_version"] == "1.0.0"
        assert version["release_version"] is None


def test_release_is_not_ready_and_no_version_is_silently_released():
    from evaluation.version_gates import evaluate

    report = evaluate()
    assert report["release_ready"] is False
    assert report["verdict"] == "NO_RELEASE_AUTHORIZED"
    assert report["package"]["release_authorized"] is False
    assert report["policy"]["training_authorized"] is False
    assert report["policy"]["compute_authorized"] is False
    assert report["policy"]["publication_authorized"] is False
    assert report["policy"]["merge_authorized"] is False
    assert report["policy"]["tags_or_releases_authorized"] is False
    assert report["closed_gates"]


def test_version_statuses_match_the_implementation_reports():
    from evaluation.version_gates import evaluate

    report = evaluate()
    statuses = {version["id"]: version["status"] for version in report["versions"]}
    assert statuses == {
        "1.1.0": "PASS",
        "1.1.1": "PASS",
        "1.2.0": "BLOCKED",
        "1.3.0": "PARTIAL",
        "1.4.0": "PARTIAL",
        "2.0.0": "NOT STARTED",
    }
    assert report["status_counts"]["PASS"] == 2
    assert report["status_counts"]["BLOCKED"] == 1
    assert report["status_counts"]["PARTIAL"] == 2
    assert report["status_counts"]["NOT STARTED"] == 1


def test_model_capability_remains_blocked_on_authorization():
    from evaluation.version_gates import evaluate

    report = evaluate()
    version = next(v for v in report["versions"] if v["id"] == "1.2.0")
    assert version["status"] == "BLOCKED"
    joined = " ".join(version["blocked_by"])
    assert "training authorization" in joined
    assert "compute authorization" in joined
    assert "canonical checkpoint" in joined


def test_ga_is_not_started():
    from evaluation.version_gates import evaluate

    report = evaluate()
    version = next(v for v in report["versions"] if v["id"] == "2.0.0")
    assert version["status"] == "NOT STARTED"
    assert version["artifact"] == "GA release (not created)"


def test_gate_matrix_is_deterministic_and_references_master_roadmap():
    from evaluation.version_gates import evaluate

    first = evaluate()
    second = evaluate()
    assert first == second
    assert first["master_roadmap"]["status"] == "CANONICAL"
    assert first["master_roadmap"]["path"] == "docs/roadmap/TAKSH_MASTER_DEVELOPMENT_ROADMAP.md"
    assert (REPO / first["master_roadmap"]["path"]).is_file()


def test_invalid_status_is_rejected(tmp_path):
    from evaluation.version_gates import VersionGateError, evaluate

    document = yaml.safe_load(
        (REPO / "configs/roadmap/five_version_program.yaml").read_text(encoding="utf-8")
    )
    document["versions"][0]["status"] = "DONE"
    path = tmp_path / "program.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(VersionGateError, match="invalid status"):
        evaluate(path)


def test_package_version_mismatch_is_rejected(tmp_path):
    from evaluation.version_gates import VersionGateError, evaluate

    document = yaml.safe_load(
        (REPO / "configs/roadmap/five_version_program.yaml").read_text(encoding="utf-8")
    )
    document["package"]["current_version"] = "9.9.9"
    path = tmp_path / "program.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(VersionGateError, match="taksh.__version__"):
        evaluate(path)


def test_cli_writes_report_and_reports_no_release(tmp_path, capsys):
    from evaluation.version_gates import main

    out = tmp_path / "version_gates.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "release ready: False" in captured.out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "taksh.version_gates/1"
    assert report["release_ready"] is False
