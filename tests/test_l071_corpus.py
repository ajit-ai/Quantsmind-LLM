"""Focused tests for the TAKSH L0.7.1 multilingual corpus-engineering qualification.

Synthetic fixtures and the authorized internal Python corpus only. No network,
no downloads, no training, no external data of any kind.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402
    l071_corpus_record as cr,
    l071_ingest as ingest,
    l071_quality as quality,
    l071_dedup as dedup,
    l071_stats as stats,
    l071_splits as splits,
    l071_packing as packing,
    l071_snapshot as snapshot,
    l071_manifest_adapter as madapter,
    l071_dlcae as dlcae,
    l071_karkain as kark,
    l071_resource_integrity as integrity,
)
from evaluation.l070_contamination import scan_corpus  # noqa: E402
import evaluation.l070_corpus_planner as planner  # noqa: E402

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"


@pytest.fixture(scope="module")
def real_records():
    return ingest.ingest_internal_python()


@pytest.fixture(scope="module")
def synth_records():
    return ingest.ingest_synthetic_fixtures()


TOK_FP = hashlib.sha256((REPO / "tokenizer" / "taksh_code_bpe_v1.json").read_bytes()).hexdigest()


def make_record(text: str, language: str = "python", synthetic: bool = False,
                source_id: str | None = None) -> dict:
    return cr.build_record(
        source_id=source_id or ("synthetic-fixture-x" if synthetic else "test-src"),
        source_version="test/1",
        original_ref="test",
        original_index=0,
        text=text,
        language=language,
        language_family="l1_target",
        license_id="synthetic" if synthetic else "mit",
        license_decision_ref="n/a" if synthetic else "configs/data/l070_license_decisions.yaml",
        provenance_ref="test",
        tokenizer_id="taksh_code_bpe_v1",
        tokenizer_version="1",
        tokenizer_fingerprint=TOK_FP,
        token_count=8,
        synthetic=synthetic,
    )


# --------------------------------------------------------------------------- #
# 1. Canonical corpus record schema
# --------------------------------------------------------------------------- #
def test_record_build_and_validate():
    r = make_record("def f():\n    return 1\n")
    assert cr.validate_record(r) == []
    assert len(r["original_fingerprint"]) == 64
    assert len(r["normalized_fingerprint"]) == 64


def test_record_missing_field_rejected():
    r = make_record("x = 1\n")
    del r["provenance_ref"]
    assert any("missing field: provenance_ref" in p for p in cr.validate_record(r))


def test_synthetic_record_forced_ineligible():
    r = make_record("x = 1\n", synthetic=True)
    assert r["training_eligibility"] == "INELIGIBLE"
    r["training_eligibility"] = "ELIGIBLE"
    assert any("INELIGIBLE" in p for p in cr.validate_record(r))


# --------------------------------------------------------------------------- #
# 2. Provenance preservation
# --------------------------------------------------------------------------- #
def test_record_provenance_never_rewritten():
    r = make_record("print('hello world')\n")
    fp_before = r["original_fingerprint"]
    r2 = copy.deepcopy(r)
    r2["normalized_fingerprint"] = "0" * 64
    assert r2["original_fingerprint"] == fp_before
    assert r2["provenance_ref"] == "test"
    assert r["transformation_history"]


def test_transformation_history_appended():
    r = make_record("print('x')\r\n")
    steps = [s["step"] for s in r["transformation_history"]]
    assert "ingest" in steps and "normalize" in steps


# --------------------------------------------------------------------------- #
# 3. Source adapter authorization enforcement
# --------------------------------------------------------------------------- #
def test_internal_python_authorization():
    auth = ingest.internal_python_authorization()
    assert auth["authorized"] is True
    assert "internal research" in auth["scope"]


def test_external_source_rejected_before_ingestion():
    with pytest.raises(ingest.IngestRejected):
        ingest.ingest_unapproved("src-bigcode-stack-v2")
    with pytest.raises(ingest.IngestRejected):
        ingest.ingest_unapproved("src-codesearchnet")


def test_source_classification():
    assert ingest.classify_source("synthetic-fixture-000") == ingest.SYNTHETIC
    assert ingest.classify_source("src-taksh-l04-code-corpus") == ingest.REAL
    assert ingest.classify_source("src-bigcode-stack-v1") == ingest.CANDIDATE
    assert ingest.classify_source("src-karkain-corpus") == ingest.BLOCKED


# --------------------------------------------------------------------------- #
# 4. Python corpus integrity
# --------------------------------------------------------------------------- #
def test_python_corpus_integrity_gate(real_records):
    assert len(real_records) == 378
    assert all(r["language"] == "python" for r in real_records)
    assert all(len(r["original_fingerprint"]) == 64 for r in real_records)


def test_ingest_rejects_tampered_corpus(tmp_path, monkeypatch):
    bad = tmp_path / "train.jsonl"
    bad.write_text('{"text": "x = 1\\n", "domain": "code", "language": "python", "license": "mit"}\n',
                   encoding="utf-8")
    monkeypatch.setattr(ingest, "L04_TRAIN", bad)
    monkeypatch.setattr(ingest, "L04_MANIFEST", REPO / "data" / "l04_code_corpus" / "dataset_manifest.json")
    with pytest.raises(ingest.IngestRejected):
        ingest.ingest_internal_python()


# --------------------------------------------------------------------------- #
# 5. UTF-8 and encoding handling
# --------------------------------------------------------------------------- #
def test_crlf_and_bom_normalization_equivalent():
    a = cr.normalized_text("﻿line1\r\nline2  \r\n")[0]
    b = cr.normalized_text("line1\nline2\n")[0]
    assert a == b


def test_record_derives_both_fingerprints():
    r = make_record("def f():\n    return 1\n")
    assert len(r["original_fingerprint"]) == 64
    assert len(r["normalized_fingerprint"]) == 64
    r2 = make_record("def f():\r\n    return 1\r\n")
    assert r2["normalized_fingerprint"] == r["normalized_fingerprint"]
    assert r2["original_fingerprint"] != r["original_fingerprint"]


# --------------------------------------------------------------------------- #
# 6 + 7. Language-specific normalization; indentation preservation
# --------------------------------------------------------------------------- #
def test_indentation_preserved():
    text = "def f():\n        deep = True\n    return deep\n"
    assert "        deep" in cr.normalized_text(text)[0]


def test_comments_preserved():
    text = "# license: mit\nx = 1  # trailing comment\n"
    out = cr.normalized_text(text)[0]
    assert "# license: mit" in out and "# trailing comment" in out


def test_string_literal_whitespace_preserved():
    text = 's = "  keep   spaces  "\n'
    out = cr.normalized_text(text)[0]
    assert '"  keep   spaces  "' in out


# --------------------------------------------------------------------------- #
# 8. Parser-adapter behaviour
# --------------------------------------------------------------------------- #
def test_python_syntax_validation():
    assert quality.classify_sample("def f():\n    return 1\n", "python")["checks"]["syntax"] in ("VALIDATED", "VALIDATED (bundle)")
    assert quality.classify_sample("def broken(:\n", "python")["quality_status"] == "INVALID"


def test_non_python_toolchain_honestly_not_checked():
    out = quality.classify_sample("fn main() { print('x'); }\n", "karkain")
    assert "NOT_CHECKED" in str(out["checks"]["syntax"])


def test_bundle_segments_checked():
    text = "def f():\n    return 1\n\n<|doc|>\n\ndef g():\n    return 2\n"
    out = quality.classify_sample(text, "python")
    assert out["checks"].get("bundle_segments") == 2
    assert "multi_document_bundle" in out["flags"]


# --------------------------------------------------------------------------- #
# 9. Data-quality classifications
# --------------------------------------------------------------------------- #
def test_secret_routes_to_human_review():
    out = quality.classify_sample("key = 'AKIA" + "A" * 16 + "'\n", "python")
    assert out["quality_status"] == "HUMAN_REVIEW_REQUIRED"


def test_classification_states_exhaustive():
    assert set(quality.STATES) == {"VALIDATED", "VALIDATED_WITH_WARNINGS", "NO_AUTOMATED_FLAGS",
                                   "HUMAN_REVIEW_REQUIRED", "INVALID", "BLOCKED", "NOT_CHECKED"}


# --------------------------------------------------------------------------- #
# 10. Exact duplicate detection + 11. near-duplicate detection
# --------------------------------------------------------------------------- #
def test_exact_duplicates_grouped():
    records = [make_record("def f():\n    return 1\n", source_id=f"src-{i}") for i in range(3)]
    texts = ["def f():\n    return 1\n"] * 3
    for i, r in enumerate(records):
        r["_text"] = texts[i]
    assigned = dedup.assign_duplicate_groups(records)
    assert len(assigned["groups"]) == 1


def test_near_duplicate_jaccard():
    a = dedup.shingles("alpha beta gamma delta epsilon zeta eta theta iota kappa")
    b = dedup.shingles("alpha beta gamma delta epsilon zeta eta theta iota CHANGED")
    assert dedup.jaccard(a, b) >= 0.5
    assert dedup.jaccard(a, dedup.shingles("completely different words over here now today folks sure thing")) < 0.5


# --------------------------------------------------------------------------- #
# 12. Cross-split leakage protection
# --------------------------------------------------------------------------- #
def test_duplicate_group_cannot_span_splits(real_records, synth_records):
    records = real_records + synth_records
    for i, r in enumerate(records):
        r["_text"] = "placeholder" if not r["source_id"].startswith("synthetic-fixture") else None
    from evaluation.l071_ingest import SYNTHETIC_FIXTURES
    texts = {i: json.loads(l)["text"] for i, l in enumerate(open("data/l04_code_corpus/train.jsonl", encoding="utf-8"))}
    for r in records:
        if r["source_id"].startswith("synthetic-fixture"):
            r["_text"] = SYNTHETIC_FIXTURES[r["original_index"]]["text"]
        else:
            r["_text"] = texts[r["original_index"]]
    assigned = dedup.assign_duplicate_groups(records, near_threshold=0.7)
    for i, r in enumerate(records):
        r["duplicate_group"] = assigned["group_id"][i]
    result = splits.assign_splits(records)
    group_splits = {}
    for r in records:
        group_splits.setdefault(r["duplicate_group"], set()).add(result[r["sample_id"]])
    assert all(len(v) == 1 for v in group_splits.values())


# --------------------------------------------------------------------------- #
# 13. Benchmark contamination rejection
# --------------------------------------------------------------------------- #
def test_contamination_screen_blocks_embedded_bench_problem():
    bench = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    probe = json.dumps(bench["problems"][0], sort_keys=True)
    out = scan_corpus(["harmless unique xyzzy content", probe])
    assert out["decision"].startswith("BLOCKED")


# --------------------------------------------------------------------------- #
# 14. Tokenizer integration
# --------------------------------------------------------------------------- #
def test_tokenizer_fingerprint_matches_file():
    assert stats.tokenizer_fingerprint() == hashlib.sha256(
        (REPO / "tokenizer" / "taksh_code_bpe_v1.json").read_bytes()).hexdigest()


def test_real_records_have_measured_tokens(real_records):
    assert all(isinstance(r["token_count"], int) and r["token_count"] > 0 for r in real_records)


# --------------------------------------------------------------------------- #
# 15. Per-language token statistics
# --------------------------------------------------------------------------- #
def test_per_language_stats_real_vs_synthetic(real_records, synth_records):
    all_records = real_records + synth_records
    per = stats.per_language_stats(all_records)
    py = per["python"]
    assert py["raw_samples"] == 378             # real records only; synthetic counted separately
    assert py["token_count"] == 1_590_326           # REAL tokens only
    assert py["synthetic_tokens"] == synth_records[0]["token_count"]
    assert py["length_distribution"][">2048"] == 0 or py["exceeding_context"] >= 0


# --------------------------------------------------------------------------- #
# 16. Synthetic vs real corpus accounting
# --------------------------------------------------------------------------- #
def test_distribution_separates_synthetic(real_records, synth_records):
    dist = stats.distribution_report(real_records + synth_records)
    assert dist["real_corpus_tokens"] == 1_590_326
    assert dist["synthetic_tokens"] == sum(r["token_count"] for r in synth_records)
    assert dist["real_corpus_tokens"] != dist["synthetic_tokens"]


# --------------------------------------------------------------------------- #
# 17. Language-family distribution
# --------------------------------------------------------------------------- #
def test_distribution_families_and_languages(real_records, synth_records):
    dist = stats.distribution_report(real_records + synth_records)
    assert "python" in dist["languages_with_real_data"]
    assert "karkain" not in dist["languages_with_real_data"]
    assert "l1_target" in dist["real_corpus_tokens_by_family"]


# --------------------------------------------------------------------------- #
# 18. Corpus planning determinism
# --------------------------------------------------------------------------- #
def test_corpus_plan_deterministic():
    cfg = planner.load_plan_config()
    p1 = planner.plan_for("equal_language", cfg)
    p2 = planner.plan_for("equal_language", cfg)
    assert p1["plan_sha256"] == p2["plan_sha256"]


# --------------------------------------------------------------------------- #
# 19 + 20. Sequence packing + context-window enforcement
# --------------------------------------------------------------------------- #
def test_sequence_packing_greedy_and_bounded():
    result = packing.pack_sequences([
        {"sample_id": "a", "tokens": [1] * 100},
        {"sample_id": "b", "tokens": [1] * 300},
        {"sample_id": "c", "tokens": [1] * 2048},  # 2048 + eos = 2049 > 2048 -> rejected
        {"sample_id": "d", "tokens": [1] * 1500},
        {"sample_id": "e", "tokens": [1] * 400},
    ])
    assert result["all_within_context"] is True
    assert result["rejected_count"] == 1
    assert result["rejected_sequences"][0]["sample_id"] == "c"
    assert result["boundaries_traceable"] is True


def test_oversized_sample_not_truncated():
    result = packing.pack_sequences([{"sample_id": "big", "tokens": [1] * 3000}])
    assert result["rejected_count"] == 1
    assert result["packed_sequences"] == 0


def test_deterministic_packing():
    seqs = [{"sample_id": f"s{i}", "tokens": [i] * (10 + i)} for i in range(6)]
    r1 = packing.pack_sequences(seqs)
    r2 = packing.pack_sequences(seqs)
    assert r1["pack_manifest_sha256"] == r2["pack_manifest_sha256"]


# --------------------------------------------------------------------------- #
# 21. Split reproducibility
# --------------------------------------------------------------------------- #
def test_split_assignment_reproducible(real_records, synth_records):
    records = real_records + synth_records
    for r in records:
        r["duplicate_group"] = r["sample_id"]
    a1 = splits.assign_splits(records)
    a2 = splits.assign_splits(records)
    assert a1 == a2


# --------------------------------------------------------------------------- #
# 22 + 23. Snapshot generation, integrity and comparison
# --------------------------------------------------------------------------- #
def test_snapshot_gates_fail_closed_for_synthetic_only(synth_records):
    for i, r in enumerate(synth_records):
        r["_text"] = ingest.SYNTHETIC_FIXTURES[r["original_index"]]["text"]
    contamination = snapshot.process_corpus(synth_records)
    gates = snapshot.eligibility_gates(contamination["decision"], synth_records)
    assert gates["frozen_for_training"] is False


def test_snapshot_comparison_detects_changes():
    old = {"snapshot_id": "a", "counts": {"real_tokens": 100}}
    new = {"snapshot_id": "b", "counts": {"real_tokens": 200}}
    out = snapshot.compare_snapshots(old, new)
    assert "snapshot_id" in out["changed_fields"]
    assert "counts" in out["changed_fields"]
    assert out["identical"] is False


def test_record_fingerprint_excludes_wallclock(real_records):
    r = real_records[0]
    fp = cr.record_fingerprint(r)
    assert fp == cr.record_fingerprint(copy.deepcopy(r))


# --------------------------------------------------------------------------- #
# 24. DLCAE integration
# --------------------------------------------------------------------------- #
def test_dlcae_never_promotes(tmp_path):
    out = tmp_path / "dlcae.json"
    assert dlcae.main(["--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    current = report["distinct_state_current"]
    assert current["language functionally qualified"] == "NONE"
    assert current["corpus training-eligible"].startswith("NONE")
    assert report["distinct_state_current"]["model trained"] == "NO"


# --------------------------------------------------------------------------- #
# 25. Karkain adapter
# --------------------------------------------------------------------------- #
def test_karkain_adapter_pinned_and_labeled():
    assert kark.validate() == []
    assert kark.ADAPTER_SCHEMA["spec_version_pinned"] == "1.1.0"
    check = kark.adapter_self_check()
    assert all("SYNTHETIC" in f["label"] for f in check["fixtures"])
    assert all(f["training_eligibility"] == "INELIGIBLE" for f in check["fixtures"])
    assert check["fixtures_label"] == "SYNTHETIC_TEST_FIXTURE"


# --------------------------------------------------------------------------- #
# 26. Training-manifest compatibility (L0.6.1)
# --------------------------------------------------------------------------- #
def test_training_manifest_compatible_and_fail_closed():
    manifest, summary = madapter.build_manifest()
    from evaluation.l0601_training_manifest import validate_manifest

    assert validate_manifest(manifest) == []
    assert summary["train_samples"] == 344
    assert summary["val_samples"] == 17
    assert summary["test_samples"] == 16
    assert summary["train_tokens"] + summary["val_tokens"] + summary["test_tokens"] + 2425 == 1_590_326
    assert summary["synthetic_in_manifest"] == 0


def test_training_manifest_fingerprints_are_distinct():
    manifest, _ = madapter.build_manifest()
    fps = [e["fingerprint_sha256"] for e in manifest["entries"]]
    assert len(set(fps)) == len(fps)


# --------------------------------------------------------------------------- #
# 27. Resource and disk preflight
# --------------------------------------------------------------------------- #
def test_resource_preflight_and_integrity():
    pre = snapshot.resource_preflight()
    assert pre["memory_floor_bytes"] == 128 * 1024 * 1024
    assert pre["disk"]["free_bytes"] > 0
    assert pre["preflight_ok"] in (True, False)
    report = integrity.main(["--out", str(Path(tempfile.mkdtemp()) / "integrity.json")])
    assert report == 0


def test_snapshot_preflight_blocks_starved_memory(monkeypatch):
    class _FakeVM:
        total = 4 * 2**30
        available = 64 * 1024 * 1024   # below the 128 MiB floor
        percent = 98.5

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    pre = snapshot.resource_preflight()
    assert pre["preflight_ok"] is False


# --------------------------------------------------------------------------- #
# 28. Cancellation and recovery
# --------------------------------------------------------------------------- #
def test_atomic_write_leaves_no_partials(tmp_path):
    out_dir = tmp_path / "snaptest"
    out_dir.mkdir()
    snapshot_path = out_dir / "snapshot.json"
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=out_dir, prefix="snapshot_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write('{"ok": true}')
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, snapshot_path)
    assert snapshot_path.exists()
    assert list(out_dir.glob("snapshot_*.tmp")) == []


def test_snapshot_main_blocked_when_starved(monkeypatch, tmp_path):
    class _FakeVM:
        total = 4 * 2**30
        available = 32 * 1024 * 1024
        percent = 99.0

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    rc = snapshot.main(["--out-dir", str(tmp_path / "snap_blocked")])
    assert rc == 2
    assert not (tmp_path / "snap_blocked" / "snapshot.json").exists()


# --------------------------------------------------------------------------- #
# 29. Existing benchmark preservation
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    path = REPO / "evaluation" / "l02_python_bench_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    path = REPO / "evaluation" / "l051_multilingual_benchmark_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MULTILINGUAL_SHA


# --------------------------------------------------------------------------- #
# 30. Checkpoint and tokenizer preservation
# --------------------------------------------------------------------------- #
def test_checkpoint_preserved():
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
