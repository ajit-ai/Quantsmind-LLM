"""Focused tests for the frozen-benchmark manifest.

The tests never download, modify or re-score a benchmark. They verify that
absent artifacts are reported as absent and that historical results stay
immutable.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ALLOWED = {"PASS", "PARTIAL", "BLOCKED", "NOT STARTED", "NOT EVALUATED"}


def _temporary_manifest(
    tmp_path: Path,
    *,
    artifact_text: str,
    expected_sha256: str,
) -> Path:
    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/frozen_benchmarks.yaml").read_text(encoding="utf-8")
    )
    artifact = tmp_path / "benchmark.json"
    artifact.write_bytes(artifact_text.encode("utf-8"))
    provenance = tmp_path / "hash_source.txt"
    provenance.write_text(f"expected {expected_sha256}\n", encoding="utf-8")
    manifest["benchmarks"] = [
        {
            "id": "test-benchmark",
            "name": "Test benchmark",
            "path": str(artifact),
            "expected_sha256": expected_sha256,
            "expected_tasks": 3,
            "hash_sources": [str(provenance)],
            "historical": {
                "result": "not measured",
                "measured": False,
                "preserved": True,
                "re_scored": False,
            },
            "status": "BLOCKED",
            "note": "test fixture",
        }
    ]
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_pins_both_expected_hashes_and_task_counts():
    from evaluation.frozen_benchmarks import load_manifest

    manifest = load_manifest()
    benchmarks = {item["id"]: item for item in manifest["benchmarks"]}
    assert set(benchmarks) == {"l02-python-v1", "l051-multilingual-v1"}
    assert benchmarks["l02-python-v1"]["expected_sha256"] == (
        "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
    )
    assert benchmarks["l02-python-v1"]["expected_tasks"] == 30
    assert benchmarks["l051-multilingual-v1"]["expected_sha256"] == (
        "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
    )
    assert benchmarks["l051-multilingual-v1"]["expected_tasks"] == 64
    for key in ("download", "modify", "rescore", "capability_claim"):
        assert manifest["policy"][key] is False


def test_absent_benchmarks_are_reported_as_absent_not_scored():
    from evaluation.frozen_benchmarks import inspect

    report = inspect()
    assert report["schema"] == "taksh.frozen_benchmarks/1"
    assert report["status"] in ALLOWED
    assert report["status"] == "BLOCKED"
    assert report["verdict"] == "FROZEN_BENCHMARKS_ABSENT"
    assert report["absent"] == ["l02-python-v1", "l051-multilingual-v1"]
    assert report["hash_mismatches"] == []
    for benchmark in report["benchmarks"]:
        assert benchmark["availability"] == "ABSENT"
        assert benchmark["actual_sha256"] is None
        assert benchmark["status"] == "BLOCKED"


def test_historical_results_are_preserved_and_never_rescored():
    from evaluation.frozen_benchmarks import inspect

    report = inspect()
    historical = {item["id"]: item for item in report["historical_results"]}
    assert historical["l02-python-v1"]["result"] == "0/30"
    assert historical["l02-python-v1"]["measured"] is True
    assert historical["l051-multilingual-v1"]["result"] == "64 tasks, not measured"
    assert historical["l051-multilingual-v1"]["measured"] is False
    assert all(item["preserved"] is True for item in historical.values())
    assert all(item["re_scored"] is False for item in historical.values())
    assert report["policy"] == {
        "download": False,
        "modify": False,
        "rescore": False,
        "capability_claim": False,
        "evaluation_requires_owner_authorization": True,
        "historical_results_immutable": True,
    }


def test_expected_hashes_are_confirmed_by_pinned_sources():
    from evaluation.frozen_benchmarks import inspect

    report = inspect()
    for benchmark in report["benchmarks"]:
        assert benchmark["provenance_ok"] is True
        assert benchmark["provenance"]
        for source in benchmark["provenance"]:
            assert source["present"] is True
            assert source["confirms_expected_sha256"] is True


def test_present_artifact_with_matching_hash_passes(tmp_path):
    from evaluation.frozen_benchmarks import inspect

    text = '{"tasks": []}\n'
    expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path = _temporary_manifest(tmp_path, artifact_text=text, expected_sha256=expected)

    report = inspect(path)
    benchmark = report["benchmarks"][0]
    assert benchmark["availability"] == "PRESENT_VERIFIED"
    assert benchmark["actual_sha256"] == expected
    assert benchmark["status"] == "PASS"
    assert report["status"] == "PASS"
    assert report["verdict"] == "FROZEN_BENCHMARKS_INTACT"


def test_present_artifact_with_wrong_hash_fails_closed(tmp_path):
    from evaluation.frozen_benchmarks import inspect

    path = _temporary_manifest(
        tmp_path,
        artifact_text="tampered\n",
        expected_sha256="0" * 64,
    )

    report = inspect(path)
    benchmark = report["benchmarks"][0]
    assert benchmark["availability"] == "PRESENT_HASH_MISMATCH"
    assert benchmark["actual_sha256"] != benchmark["expected_sha256"]
    assert benchmark["status"] == "BLOCKED"
    assert report["status"] == "BLOCKED"
    assert report["verdict"] == "FROZEN_BENCHMARK_INTEGRITY_FAILURE"


def test_manifest_cannot_authorize_rescoring_or_rewrite_history(tmp_path):
    from evaluation.frozen_benchmarks import FrozenBenchmarkError, inspect

    manifest = yaml.safe_load(
        (REPO / "configs/roadmap/frozen_benchmarks.yaml").read_text(encoding="utf-8")
    )
    manifest["policy"]["rescore"] = True
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(FrozenBenchmarkError, match="policy.rescore"):
        inspect(path)

    manifest["policy"]["rescore"] = False
    manifest["benchmarks"][0]["historical"]["re_scored"] = True
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(FrozenBenchmarkError, match="must not re-score"):
        inspect(path)


def test_report_is_deterministic():
    from evaluation.frozen_benchmarks import inspect

    assert inspect() == inspect()


def test_cli_writes_report_and_exits_nonzero_while_absent(tmp_path, capsys):
    from evaluation.frozen_benchmarks import main

    out = tmp_path / "frozen.json"
    code = main(["--out", str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert out.is_file()
    assert "FROZEN_BENCHMARKS_ABSENT" in captured.out
    assert "ABSENT" in captured.out
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "taksh.frozen_benchmarks/1"
    assert written["status"] == "BLOCKED"
