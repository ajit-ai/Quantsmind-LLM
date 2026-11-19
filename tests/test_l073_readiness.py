"""Focused tests for TAKSH L0.7.3 — training readiness & CI regression closure."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluation.l073_readiness as readiness  # noqa: E402
from evaluation.m5_adapters import ExternalHFAdapter  # noqa: E402
from evaluation.m6_sovereign import estimate_scales  # noqa: E402
import evaluation.l071_karkain as kark  # noqa: E402
from evaluation.l070_source_registry import load_registry as load_l070_registry  # noqa: E402

from fixture_s2_checkpoint import build_s2_checkpoint  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"

LEDGER = REPO / "configs" / "data" / "l073_regression_ledger.yaml"


# --------------------------------------------------------------------------- #
# Adapter availability and offline behavior
# --------------------------------------------------------------------------- #
def test_adapter_preflight_never_fetches_by_default():
    adapter = ExternalHFAdapter("Qwen/Qwen2.5-Coder-1.5B-Instruct",
                                license_record={"license_id": "apache-2.0", "verified": True})
    ok, reason = adapter.preflight()
    assert ok is False
    assert "nothing was fetched" in reason
    assert adapter._model is None
    assert adapter.allow_download is False


def test_adapter_missing_transformers_reports_stably(monkeypatch):
    adapter = ExternalHFAdapter("some/model", license_record={"license_id": "mit"})
    monkeypatch.setattr(adapter, "_transformers_available", lambda: None)
    ok, reason = adapter.preflight()
    assert ok is False
    assert "transformers" in reason
    assert "nothing was fetched" in reason      # both facts reported in one stable reason


def test_adapter_reports_weights_present_needs_transformers(monkeypatch, tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"\0" * 1024)
    adapter = ExternalHFAdapter("some/model", license_record={"license_id": "mit"})
    monkeypatch.setattr(adapter, "_snapshot_path", lambda: weights)
    monkeypatch.setattr(adapter, "_transformers_available", lambda: None)
    ok, reason = adapter.preflight()
    assert ok is False
    assert "transformers" in reason


def test_adapter_requires_licence_first():
    adapter = ExternalHFAdapter("some/model", license_record={})
    ok, reason = adapter.preflight()
    assert ok is False
    assert "no licence record" in reason


def test_adapter_weights_discovery_does_not_import_transformers():
    # Discovery path: absence of transformers must never raise or fetch.
    adapter = ExternalHFAdapter("definitely/not-here-model", license_record={"license_id": "mit"})
    ok, reason = adapter.preflight()
    assert ok is False
    assert "nothing was fetched" in reason


# --------------------------------------------------------------------------- #
# Resource policy and RAM constraint handling
# --------------------------------------------------------------------------- #
def test_ram_policy_synthetic_measurement():
    scales = estimate_scales(memory={"total_bytes": 4 * 2**30, "available_bytes": 128 * 2**20})
    assert any(s.fits_total_ram and not s.fits_available_ram for s in scales)


def test_ram_policy_large_host_all_fit_total():
    scales = estimate_scales(memory={"total_bytes": 1 << 40, "available_bytes": 1 << 40})
    assert all(s.fits_total_ram for s in scales)


def test_ram_measurement_live_nonzero():
    from training.preflight import system_memory

    mem = system_memory()
    assert mem["total_bytes"] > 0


# --------------------------------------------------------------------------- #
# Manifest path normalization (tmp-dir regression)
# --------------------------------------------------------------------------- #
def test_benchmark_manifest_has_no_absolute_paths(tmp_path):
    from data.python_v1.benchmarks import load_benchmark_records, load_registry

    bench = tmp_path / "bench.jsonl"
    rows = [
        {"task_id": f"t{i}", "prompt": f"solve {i}", "test": "assert"} for i in range(5)
    ]
    bench.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    registry_path = tmp_path / "benchmarks.yaml"
    registry_path.write_text(yaml.safe_dump({
        "schema": "quantsmind.benchmark_registry/1",
        "suites": [{"id": "fixture-bench", "version": "1", "availability": "available",
                    "path": "bench.jsonl", "fields": ["prompt", "test"]}],
    }), encoding="utf-8")
    loaded = load_registry(registry_path)
    bench = bench  # noqa: F841
    records = load_benchmark_records(loaded, base_dir=registry_path.parent)
    origins = [p["origin"] for p in records.provenance]
    joined = json.dumps(origins)
    assert str(tmp_path) not in joined
    assert all(p["content_sha256"] for p in records.provenance)


# --------------------------------------------------------------------------- #
# Checkpoint fixture and geometry inspection
# --------------------------------------------------------------------------- #
def test_s2_fixture_geometry_is_compatible(tmp_path):
    from training.geometry import inspect

    ckpt = build_s2_checkpoint(tmp_path)
    report = inspect(checkpoint_path=ckpt,
                     tokenizer_path=str(REPO / "tokenizer" / "taksh_code_bpe_v1.json"),
                     model_config_path=str(REPO / "configs" / "model" / "code_m3_9m.yaml"))
    assert report.compatible is True
    assert report.embedding_rows == 5822
    assert report.checkpoint_step == 300


def test_s2_fixture_bundle_loads_safely(tmp_path):
    from serving.local_inference import torch_load_checkpoint

    ckpt = build_s2_checkpoint(tmp_path)
    blob = torch_load_checkpoint(str(ckpt))   # safe weights_only path
    assert blob["schema"] == "quantsmind.fixture/synthetic"
    assert blob["step"] == 300
    assert blob["config"]["run"]["model_config"].endswith("code_m3_9m.yaml")


def test_s2_fixture_is_labelled_synthetic(tmp_path):
    ckpt = build_s2_checkpoint(tmp_path)
    assert "synthetic" in ckpt.name


# --------------------------------------------------------------------------- #
# Regression ledger validation
# --------------------------------------------------------------------------- #
def test_ledger_schema_and_dispositions():
    data = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    assert data["schema"] == "taksh.ci_regression_ledger/1"
    entries = data["entries"]
    assert len(entries) == 7
    ids = [e["test"] for e in entries]
    assert len(set(ids)) == 7
    for e in entries:
        assert e["disposition"] in {"RESOLVED", "BLOCKED", "FORMALLY_DEFERRED"}
        assert e["root_cause"] and e["evidence"] and e["affected_code"]
        assert "pr_63" not in e["disposition"].lower()


def test_ledger_pr63_not_accessible_but_documented():
    data = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    inv = data["pr_63_investigation"]
    assert "NOT ACCESSIBLE" in inv["outcome"]
    assert any("Could not resolve to a PullRequest" in s for s in inv["evidence"])
    assert inv["disposition"].startswith("NOT_ACCESSIBLE")


def test_ledger_covers_every_inherited_failure():
    data = yaml.safe_load(LEDGER.read_text(encoding="utf-8"))
    tests = {e["test"] for e in data["entries"]}
    assert any("test_external_adapter_does_not_download_by_default" in t for t in tests)
    assert any("test_available_ram_is_the_binding_constraint" in t for t in tests)
    assert any("test_g7_manifest_has_no_timestamp" in t for t in tests)
    assert any("test_geometry_inspector_reports_a_matching_triple" in t for t in tests)
    assert any("test_tokenizer_artifact_and_model_geometry_are_unchanged" in t for t in tests)
    assert any("test_g10_geometry_inspector_accepts_the_s2_triple" in t for t in tests)
    assert any("test_sampled_decoding_is_reproducible_with_a_seed" in t for t in tests)


# --------------------------------------------------------------------------- #
# Training-readiness gate behavior
# --------------------------------------------------------------------------- #
def test_gate_verdict_is_blocked():
    report = readiness.evaluate()
    assert report["verdict"] == "BLOCKED"
    assert set(report["blocking_dimensions"]) >= {
        "owner_training_authorization",
        "checkpoint_availability_and_identity",
        "hardware_resource_feasibility",
    }


def test_gate_dimensions_complete():
    report = readiness.evaluate()
    assert report["dimension_count"] == 12
    assert set(report["dimensions"]) == {
        "corpus_snapshot_integrity", "source_provenance", "technical_sample_eligibility",
        "legal_training_permission", "owner_training_authorization", "language_coverage",
        "tokenizer_model_compatibility", "checkpoint_availability_and_identity",
        "hardware_resource_feasibility", "benchmark_thresholds",
        "ci_regression_status", "reproducibility_and_recovery",
    }


def test_gate_training_eligible_tokens_zero():
    report = readiness.evaluate()
    assert report["training_eligible_tokens"] == 0
    assert report["summary"]["training_execution_readiness"].startswith("NOT READY")
    assert report["summary"]["owner_authorized_corpus"] is False


def test_gate_no_synthetic_authorization_possible(tmp_path):
    report_path = tmp_path / "readiness.json"
    rc = readiness.main(["--out", str(report_path)])
    assert rc == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["dimensions"]["owner_training_authorization"]["status"] == "MISSING"
    assert report["frozen_research_snapshot"].startswith("l072-l04-fdb1fee2beaae9c2")


def test_gate_verdict_blocked_regardless_of_memory_headroom(monkeypatch):
    """Even a 'big' host stays blocked: authorization is the binding gate."""
    class _HugeVM:
        total = 1 << 50
        available = 1 << 50
        percent = 1.0

    class _FakePsutil:
        @staticmethod
        def virtual_memory():
            return _HugeVM()

    monkeypatch.setitem(sys.modules, "psutil", _FakePsutil())
    report = readiness.evaluate()
    assert report["verdict"] == "BLOCKED"
    assert "owner_training_authorization" in report["blocking_dimensions"]


def test_legal_permission_scope_is_internal_research_only():
    report = readiness.evaluate()
    dim = report["dimensions"]["legal_training_permission"]
    assert dim["status"] == "PARTIAL"
    assert "internal research" in json.dumps(dim["details"]).lower()
    assert "external_release" in json.dumps(dim["details"]).lower()


def test_tokenizer_and_foundation_compat(tmp_path):
    report = readiness.evaluate()
    assert report["dimensions"]["tokenizer_model_compatibility"]["status"] == "PASS"


def test_checkpoint_availability_missing_foundation(tmp_path):
    report = readiness.evaluate()
    dim = report["dimensions"]["checkpoint_availability_and_identity"]
    assert dim["status"] == "MISSING"
    assert dim["details"]["foundation_111m_available"] is False
    assert dim["details"]["v180_sft_usable_for_foundation"] is False


def test_language_coverage_python_only():
    report = readiness.evaluate()
    dim = report["dimensions"]["language_coverage"]
    assert dim["status"] == "PARTIAL"
    assert dim["details"]["languages_with_real_data"] == ["python"]
    assert dim["details"]["karkain"].startswith("DISCOVERED")


# --------------------------------------------------------------------------- #
# L0.7.2 snapshot integrity and immutability
# --------------------------------------------------------------------------- #
def test_l072_snapshot_id_preserved():
    snap = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json").read_text(encoding="utf-8"))
    assert snap["snapshot_id"] == "l072-l04-fdb1fee2beaae9c2"
    assert snap["frozen_for_training"] is False


def test_l072_records_integrity_hash_matches():
    import evaluation.l071_corpus_record as cr

    root = REPO / "artifacts" / "l072_l04_freeze"
    snap = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (root / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    fps = sorted(cr.record_fingerprint(r) for r in records)
    assert hashlib.sha256("".join(fps).encode()).hexdigest() == snap["integrity_hashes"]["records_sha256"]


def test_l071_parent_snapshot_unchanged():
    l072 = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json").read_text(encoding="utf-8"))
    parent_hash = hashlib.sha256((REPO / "artifacts" / "l071_l04_snapshot" / "snapshot.json").read_bytes()).hexdigest()
    assert l072["parent_snapshot"]["snapshot_sha256"] == parent_hash


def test_snapshot_integrity_dimension_passes():
    report = readiness.evaluate()
    assert report["dimensions"]["corpus_snapshot_integrity"]["status"] == "PASS"


# --------------------------------------------------------------------------- #
# Karkain and registry preservation
# --------------------------------------------------------------------------- #
def test_karkain_still_discovered():
    assert kark.validate() == []
    import yaml as _y

    strategy = _y.safe_load((REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"


def test_l070_source_registry_unchanged_semantics():
    data = load_l070_registry()
    acquired = [s for s in data["sources"] if s["acquisition_status"] == "ACQUIRED"]
    assert [s["source_id"] for s in acquired] == ["src-taksh-l04-code-corpus"]


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_l01_checkpoint_preserved():
    digest = hashlib.sha256()
    with (REPO / "runs" / "v180_sft" / "checkpoint_last.pt").open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_foundation_config_preserved():
    assert hashlib.sha256((REPO / "configs" / "foundation" / "l060_foundation.yaml").read_bytes()).hexdigest() == \
        "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"


def test_registry_still_53():
    import yaml as _y

    data = _y.safe_load((REPO / "configs" / "feasibility" / "l051_languages.yaml").read_text(encoding="utf-8"))
    assert len(data["languages"]) == 53


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA
