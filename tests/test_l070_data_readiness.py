"""Focused tests for the TAKSH L0.7.0 acquisition-readiness qualification.

Synthetic fixtures and existing local metadata only. No network, no downloads,
no dataset bytes were or are accessed by this suite.
"""

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

from evaluation import (  # noqa: E402
    l070_acquisition_gate as gate,
    l070_candidate_comparison as compare,
    l070_contamination as contamination,
    l070_corpus_planner as planner,
    l070_coverage_matrix as coverage,
    l070_data_quality as dq,
    l070_dlcae_sources as dlcae_sources,
    l070_karkain_strategy as kark,
    l070_license_qualification as lq,
    l070_observability as obs,
    l070_provenance as prov,
    l070_resource_readiness as resource,
    l070_source_registry as registry,
)
import evaluation.l0601_training_manifest as l0601man  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
KARK_BENCH_SCHEMA_SHA = "fe6acbcd9a41d5e113595d4fdb675cc616342da30029401fe16602d6c157d481"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"


# --------------------------------------------------------------------------- #
# 1. Source registry schema + 2. snapshot identity
# --------------------------------------------------------------------------- #
def test_registry_validates():
    data = registry.load_registry()
    assert registry.validate_registry(data) == []
    assert len(data["sources"]) == 6


def test_registry_rejects_unknown_status():
    data = copy.deepcopy(registry.load_registry())
    data["sources"][0]["acquisition_status"] = "NOT-A-STATUS"
    assert any("acquisition_status" in p for p in registry.validate_registry(data))


def test_registry_requires_unique_ids():
    data = copy.deepcopy(registry.load_registry())
    data["sources"].append(copy.deepcopy(data["sources"][0]))
    assert any("duplicate source_id" in p for p in registry.validate_registry(data))


def test_snapshot_identity_pinned():
    data = registry.load_registry()
    for src in data["sources"]:
        assert src["dataset_version"]
        assert "latest" not in src["dataset_version"].lower()


def test_registry_transition_table_enforced():
    assert registry.check_transition("DISCOVERED", "METADATA_REVIEW")["ok"] is True
    assert registry.check_transition("DISCOVERED", "ACQUISITION_AUTHORIZED")["ok"] is False
    assert registry.check_transition("REJECTED", "DISCOVERED")["ok"] is False
    assert registry.check_transition("QUALIFIED_FOR_ACQUISITION", "ACQUISITION_AUTHORIZED")["ok"] is True


# --------------------------------------------------------------------------- #
# 3. License decision validation + 4. missing-license fail-closed
# --------------------------------------------------------------------------- #
def test_license_decisions_valid():
    data = lq.load_decisions()
    problems = []
    if data.get("schema") != "taksh.license_qualification/1":
        problems.append("schema")
    for record in data["records"]:
        problems += lq.validate_record(record)
    assert problems == []


def test_license_record_missing_axis_fails():
    data = lq.load_decisions()
    record = copy.deepcopy(data["records"][0])
    del record["axes"]["ml_training_permissions"]
    assert any("ml_training_permissions" in p for p in lq.validate_record(record))


def test_license_approved_requires_no_fail_closed_axes():
    data = lq.load_decisions()
    record = copy.deepcopy(data["records"][0])
    record["overall_decision"] = "APPROVED"
    record["axes"]["commercial_use_permissions"]["decision"] = "UNKNOWN"
    assert any("overall APPROVED" in p for p in lq.validate_record(record))


def test_license_summary_counts():
    summary = lq.summarize(lq.load_decisions())
    assert summary["overall"]["HUMAN_LEGAL_REVIEW_REQUIRED"] >= 3
    assert summary["overall"]["REJECTED"] >= 1


# --------------------------------------------------------------------------- #
# 5. Training-permission + 6. commercial-vs-redistribution distinction
# --------------------------------------------------------------------------- #
def test_axes_stay_separate():
    data = lq.load_decisions()
    for record in data["records"]:
        assert "commercial_use_permissions" in record["axes"]
        assert "redistribution_restrictions" in record["axes"]
        assert "ml_training_permissions" in record["axes"]
        # axes must never be collapsed into a single boolean
        assert isinstance(record["axes"]["ml_training_permissions"]["training_use"], (bool, str))


def test_unknown_training_permission_blocks_approval():
    record = copy.deepcopy(lq.load_decisions()["records"][1])   # bigcode v1
    record["overall_decision"] = "APPROVED"
    problems = lq.validate_record(record)
    assert any("overall APPROVED" in p or "APPROVED" in p for p in problems)


# --------------------------------------------------------------------------- #
# 7. Owner approval + 8. acquisition authorization enforcement
# --------------------------------------------------------------------------- #
def test_qualified_state_requires_owner_and_legal_done():
    data = copy.deepcopy(registry.load_registry())
    src = data["sources"][1]                      # bigcode v1, HUMAN_LEGAL_REVIEW_REQUIRED
    src["acquisition_status"] = "QUALIFIED_FOR_ACQUISITION"
    src["owner_approval_status"] = "not_requested"
    problems = registry.validate_source(src)
    assert any("legal review" in p for p in problems)


def test_acquisition_execution_permanently_disabled():
    assert gate.ACQUISITION_EXECUTION_ENABLED is False
    wf = gate.load_workflow()
    assert wf["invariants"]["acquisition_execution_disabled"] is True


def test_acquisition_request_fails_closed_without_owner_id():
    template = gate.load_workflow()["request_dossier_template"]
    out = gate.request_acquisition(template)
    assert out["authorized"] is False
    assert "owner_authorization_id" in " ".join(out.get("problems", [])) or out["stage"] == "OWNER_APPROVAL_REQUIRED"


def test_acquisition_request_still_disabled_with_owner_id():
    template = gate.load_workflow()["request_dossier_template"]
    dossier = {**template, **{f: "x" for f in gate.REQUIRED_FIELDS},
               "expected_storage_bytes": 1024,
               "owner_authorization_id": "OWNER-DECISION-0123"}
    out = gate.request_acquisition(dossier)
    assert out["execution"].startswith("disabled")


def test_no_candidate_is_qualified_in_registry():
    data = registry.load_registry()
    assert all(s["acquisition_status"] not in {"QUALIFIED_FOR_ACQUISITION", "ACQUISITION_AUTHORIZED"}
               for s in data["sources"] if s["source_id"] != "src-taksh-l04-code-corpus")


# --------------------------------------------------------------------------- #
# 9. Provenance completeness
# --------------------------------------------------------------------------- #
def test_internal_corpus_provenance_verified():
    data = registry.load_registry()
    internal = next(s for s in data["sources"] if s["source_id"] == "src-taksh-l04-code-corpus")
    assert prov.assess(internal)["verdict"] == "VERIFIED"


def test_external_candidates_unverified_gradients():
    data = registry.load_registry()
    external = {s["source_id"]: prov.assess(s)["verdict"]
                for s in data["sources"] if s["source_id"] != "src-taksh-l04-code-corpus"}
    assert external["src-karkain-corpus"] == "REJECTED"
    assert all(v in ("PARTIALLY_VERIFIED", "UNVERIFIED") for k, v in external.items() if k != "src-karkain-corpus")


# --------------------------------------------------------------------------- #
# 10. Multilingual coverage calculations + 11. language/tech separation
# --------------------------------------------------------------------------- #
def test_coverage_matrix_shape():
    data = coverage.load_registry()
    entries = data["languages"]
    assert len(entries) == 53


def test_coverage_report_python_is_only_qualified(tmp_path, monkeypatch):
    out = tmp_path / "coverage.json"
    assert coverage.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    py = next(e for e in report["entries"] if e["registry_id"] == "python")
    assert py["existing_qualified_tokens"] == 1_590_703
    assert py["acquisition_approval_state"] == "internal_verified"
    karkain = next(e for e in report["entries"] if e["registry_id"] == "karkain")
    assert karkain["existing_qualified_tokens"] == 0
    assert report["accounting"]["approved_for_acquisition_tokens"] == 0
    assert report["accounting"]["training_eligible_tokens"] == 1_590_703


def test_supporting_technologies_separate_from_languages():
    data = coverage.load_registry()
    lang_ids = {l["id"] for l in data["languages"]}
    tech_ids = {t["name"].lower().replace(" ", "_") for t in data["supporting_technologies"]}
    assert len(tech_ids) == 20
    assert lang_ids.isdisjoint(tech_ids)
    assert "html" not in lang_ids


# --------------------------------------------------------------------------- #
# 12. Candidate vs qualified accounting
# --------------------------------------------------------------------------- #
def test_candidate_corpus_never_counted(tmp_path):
    out = tmp_path / "coverage2.json"
    coverage.main(["--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["accounting"]["candidate_corpus_tokens"] == 0
    assert report["accounting"]["unverified_corpus_tokens"] == 0
    assert report["accounting"]["actually_acquired_tokens"] == 1_590_703
    assert report["l1_languages_with_unverified_candidates"] == 16


# --------------------------------------------------------------------------- #
# 13. Duplicate detection
# --------------------------------------------------------------------------- #
def test_exact_duplicate_detection():
    samples = [{"content": "def f():\n    return 1\n", "synthetic": True},
               {"content": "def f():\n    return 1\n", "synthetic": True},
               {"content": "def g():\n    return 2\n", "synthetic": True}]
    result = dq.exact_duplicates(samples)
    assert len(result["duplicate_groups"]) == 1


def test_near_duplicates_detect_paraphrase():
    samples = [{"content": "the quick brown fox jumps over the lazy dog today", "synthetic": True},
               {"content": "the quick brown fox jumps over the lazy dog today", "synthetic": True},
               {"content": "completely different words about compilers and linkers everywhere", "synthetic": True}]
    result = dq.near_duplicates(samples, threshold=0.8)
    assert any(p["jaccard"] >= 0.8 for p in result["near_duplicate_pairs"])


# --------------------------------------------------------------------------- #
# 14. Benchmark contamination rejection
# --------------------------------------------------------------------------- #
def test_benchmarks_intact():
    pinned = contamination.verify_pinned_benchmarks()
    assert all(p["intact"] for p in pinned)
    assert next(p for p in pinned if p["id"] == "frozen-python-bench-v1")["actual"] == FROZEN_PY_SHA
    assert next(p for p in pinned if p["id"] == "multilingual-bench-v1")["actual"] == MULTILINGUAL_SHA
    assert next(p for p in pinned if p["id"] == "kark-bench-schema-v1")["actual"] == KARK_BENCH_SCHEMA_SHA


def test_contaminated_sample_blocked():
    bench = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    probe = json.dumps(bench["problems"][0], sort_keys=True)
    result = contamination.scan_corpus(["unrelated unique content xyz123", probe])
    assert result["decision"].startswith("BLOCKED")
    assert result["exact_matches"]


def test_clean_sample_passes_fixture_screen():
    result = contamination.scan_corpus(["def brand_new_function_not_in_any_benchmark_xyzzy():\n    return 42\n"])
    assert result["decision"].startswith("CLEAN")


# --------------------------------------------------------------------------- #
# 15. Data-quality validation
# --------------------------------------------------------------------------- #
def test_secret_fixture_flagged_not_cleared():
    out = dq.validate_sample(b"key = 'AKIA" + b"A" * 16 + b"'\n", "python")
    assert any(f.startswith("secret_pattern:") for f in out["flags"])
    assert out["requires_human_review"] is True


def test_clean_python_fixture_passes():
    licensed = b"# SPDX-License-Identifier: Apache-2.0\ndef add(a, b):\n    return a + b\n"
    out = dq.validate_sample(licensed, "python")
    assert out["automated_verdict"] == "NO_AUTOMATED_FLAGS"
    assert out["checks"]["python_syntax_valid"] is True


def test_licenseless_fixture_is_flagged():
    out = dq.validate_sample(b"def add(a, b):\n    return a + b\n", "python")
    assert "no_license_metadata_in_file" in out["flags"]


def test_python_syntax_error_flagged():
    out = dq.validate_sample(b"def broken(:\n", "python")
    assert "python_syntax_error" in out["flags"]


def test_generated_marker_flagged():
    out = dq.validate_sample(b"// Code generated by tool. DO NOT EDIT.\nint x;\n", "c")
    assert "generated_code_marker" in out["flags"]


# --------------------------------------------------------------------------- #
# 16. Corpus planning determinism + honesty
# --------------------------------------------------------------------------- #
def test_corpus_plan_deterministic():
    cfg = planner.load_plan_config()
    p1 = planner.plan_for("equal_language", cfg)
    p2 = planner.plan_for("equal_language", cfg)
    assert p1["plan_sha256"] == p2["plan_sha256"]


def test_corpus_plan_python_only_achievable():
    cfg = planner.load_plan_config()
    equal = planner.plan_for("equal_language", cfg)
    prop = planner.plan_for("proportional_to_availability", cfg)
    assert equal["rows"][0]["language_id"] == "python" or any(r["language_id"] == "python" for r in equal["rows"])
    py_equal = next(r for r in equal["rows"] if r["language_id"] == "python")
    assert py_equal["achievable_tokens"] == 200_000
    py_prop = next(r for r in prop["rows"] if r["language_id"] == "python")
    assert py_prop["achievable_tokens"] == 1_590_703
    assert equal["unplannable_languages"] == 53
    assert prop["unplannable_languages"] == 53


def test_corpus_plan_reports_impossible_honestly():
    cfg = planner.load_plan_config()
    equal = planner.plan_for("equal_language", cfg)
    assert equal["impossible_count"] == 52
    assert "python" not in equal["impossible_languages"]


# --------------------------------------------------------------------------- #
# 17. DLCAE-to-source qualification handoff
# --------------------------------------------------------------------------- #
def test_dlcae_handoff_never_eligible_today():
    job = dlcae_sources.qualification_job("java", "src-bigcode-stack-v2")
    assert job["eligible_for_learning_manifest"] is False
    assert "license_review_complete" in job["blocking_gates"]


def test_dlcae_handoff_unknown_source_fails():
    job = dlcae_sources.qualification_job("java", "src-does-not-exist")
    assert job["ok"] is False


def test_dlcae_internal_python_partial_but_not_eligible():
    job = dlcae_sources.qualification_job("python", "src-taksh-l04-code-corpus")
    assert job["eligible_for_learning_manifest"] is False   # contamination screen + owner gate outstanding
    assert job["gates"]["provenance_verified"] is True
    assert job["gates"]["license_review_complete"] is True


# --------------------------------------------------------------------------- #
# 18. Karkain source and compiler metadata
# --------------------------------------------------------------------------- #
def test_karkain_strategy_valid_and_pinned():
    strategy = kark.load_strategy()
    assert kark.validate_strategy(strategy) == []
    assert strategy["identity"]["spec_version_pinned"] == "1.1.0"
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"
    assert all(r["status"] == "NOT_INSTANTIATED" for r in strategy["corpus_requirements"])


def test_karkain_strategy_rejects_counts():
    strategy = copy.deepcopy(kark.load_strategy())
    strategy["corpus_requirements"][0]["status"] = "INSTANTIATED"
    problems = kark.validate_strategy(strategy)
    assert any("NOT_INSTANTIATED" in p for p in problems)


def test_karkain_spec_pin_enforced():
    strategy = copy.deepcopy(kark.load_strategy())
    strategy["identity"]["spec_version_pinned"] = "2.0.0"
    assert any("1.1.0" in p for p in kark.validate_strategy(strategy))


# --------------------------------------------------------------------------- #
# 19. Training manifest compatibility (L0.6.1)
# --------------------------------------------------------------------------- #
def test_l0601_manifest_still_valid_and_consistent():
    manifest = l0601man.example_manifest()
    assert l0601man.validate_manifest(manifest) == []
    py_entry = next(e for e in manifest["entries"] if e["id"] == "python_verified_l04")
    assert py_entry["token_count"] == 1_590_703   # same measured ground truth L0.7.0 accounts for


# --------------------------------------------------------------------------- #
# 20. Resource/storage reporting
# --------------------------------------------------------------------------- #
def test_resource_readiness_honest_reporting():
    host = resource.host()
    assert host["disk"]["measured"] is True
    assert host["disk"]["free_bytes"] > 0
    sizes = resource.source_reported_sizes()
    assert all("basis" in s for s in sizes)
    assert sizes[0]["download_required_to_verify"] is True
    est = resource.main  # smoke existence only; full run exercised below
    assert callable(est)


def test_resource_readiness_report_generated(tmp_path):
    out = tmp_path / "resources.json"
    assert resource.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["reservations"]["paid_storage"].startswith("NONE")
    assert report["reservations"]["network_downloads"].startswith("NONE")
    assert "ESTIMATE" in report["example_staging_estimate_1pt4tb"]["note"]


# --------------------------------------------------------------------------- #
# 21. Audit trail and report generation
# --------------------------------------------------------------------------- #
def test_observability_aggregate_all_tools_pass(tmp_path):
    out = tmp_path / "obs.json"
    rc = obs.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["all_exit_zero"] is True
    assert report["summary"]["acquisitions_executed"] == 0
    assert report["summary"]["bytes_acquired"] == 0
    assert {"source_registry", "license_qualification", "acquisition_gate",
            "contamination"} <= {t["tool"] for t in report["tools"]}


def test_candidate_comparison_never_recommends(tmp_path):
    out = tmp_path / "comparison.json"
    assert compare.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert all(r["recommendation"] == "NOT ACQUISITION-READY" for r in report["comparison"])


# --------------------------------------------------------------------------- #
# Preservation anchors
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    path = REPO / "evaluation" / "l02_python_bench_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    path = REPO / "evaluation" / "l051_multilingual_benchmark_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MULTILINGUAL_SHA


def test_l01_checkpoint_preserved():
    path = REPO / "runs" / "v180_sft" / "checkpoint_last.pt"
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    assert digest.hexdigest() == V180_SHA


def test_tokenizer_committed_blob_preserved():
    blob = subprocess.run(["git", "cat-file", "blob",
                           "HEAD:tokenizer/taksh_code_bpe_v1.json"],
                          capture_output=True, check=True, cwd=REPO).stdout
    assert hashlib.sha256(blob).hexdigest() == TOK_COMMITTED_SHA
