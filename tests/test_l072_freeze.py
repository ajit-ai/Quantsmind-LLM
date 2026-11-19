"""Focused tests for the TAKSH L0.7.2 corpus-freeze qualification.

Synthetic fixtures + the freeze artifacts committed by this milestone. No
network, no external data, no training.
"""

from __future__ import annotations

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
    l071_ingest as l071ing,
    l071_dedup as dedup,
    l071_karkain as kark,
    l072_reslice as rs,
    l072_snapshot as snap,
    l072_manifest_check as mcheck,
    l072_coverage as coverage,
    l072_resource_integrity as rintr,
)

ARTIFACT = REPO / "artifacts" / "l072_l04_freeze"
PARENT = REPO / "artifacts" / "l071_l04_snapshot"

FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"
V180_SHA = "9b54103695054ecb30d8521ffceb62f443d2d1fd8d064993edc2d5070cb7f666"
TOK_COMMITTED_SHA = "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7"
PY_TAG = hashlib.sha256((REPO / "tokenizer" / "taksh_code_bpe_v1.json").read_bytes()).hexdigest()


def child_payload(text: str, **kwargs):
    return cr.build_record(
        source_id="synthetic-fixture-test",
        source_version="l072-test/1",
        original_ref="synthetic",
        original_index=0,
        text=text,
        language="python",
        language_family="l1_target",
        license_id="mit",
        license_decision_ref="configs/data/l070_license_decisions.yaml",
        provenance_ref="synthetic",
        tokenizer_id="taksh_code_bpe_v1",
        tokenizer_version="1",
        tokenizer_fingerprint=PY_TAG,
        token_count=rs.tok_count(text),
        synthetic=True,
        **kwargs,
    )


def make_fake_parent():
    parent = child_payload("x = 1\n")
    parent["sample_id"] = "parent-0000"
    parent["transformation_history"] = [
        {"step": "ingest", "version": "l071-pipeline/1", "detail": "index=0"},
    ]
    return parent


BUNDLE = """def alpha():
    return 1

<|doc|>

def beta():
    return 2
"""

# --------------------------------------------------------------------------- #
# 1. Source-aware re-slicing
# --------------------------------------------------------------------------- #
def test_reslice_bundle_markers():
    parent = make_fake_parent()
    children, quar = rs.reslice_record(parent, BUNDLE)
    assert len(children) == 2
    assert {c["file_index"] for c in children} == {0, 1}
    assert all(c["extraction_method"] == "marker_split" for c in children)
    assert all(c["parent_sample_id"] == "parent-0000" for c in children)


# --------------------------------------------------------------------------- #
# 2. Original-to-derived lineage
# --------------------------------------------------------------------------- #
def test_lineage_fields():
    parent = make_fake_parent()
    children, _ = rs.reslice_record(parent, BUNDLE)
    for c in children:
        assert c["parent_sample_id"] == parent["sample_id"]
        assert c["parent_original_fingerprint"] == parent["original_fingerprint"]
        assert len(c["original_fingerprint"]) == 64
        assert len(c["segment_fingerprint"]) == 64
        assert any(s["step"] == "reslice" for s in c["transformation_history"])


# --------------------------------------------------------------------------- #
# 3. Structural boundary preservation
# --------------------------------------------------------------------------- #
def test_python_children_parse():
    parent = make_fake_parent()
    children, _ = rs.reslice_record(parent, BUNDLE)
    import ast

    for c in children:
        ast.parse(c["_text"])


# --------------------------------------------------------------------------- #
# 4 + 5 + 6. Oversized handling; 2048 enforcement; no silent truncation
# --------------------------------------------------------------------------- #
def test_oversized_class_subdivided_under_limit():
    methods = []
    for i in range(320):
        methods.append(f"    def method_{i:03d}(self):\n        return {i}\n")
    big_class = "import os\n\nclass Huge:\n" + "".join(methods)
    parent = make_fake_parent()
    children, quar = rs.reslice_record(parent, big_class)
    assert children
    assert all(c["token_count"] <= rs.CONTEXT_LIMIT for c in children)
    assert not quar


def test_monolithic_unit_quarantined():
    giant_literal = "DATA = [" + ", ".join(str(i) for i in range(40000)) + "]\n"
    parent = make_fake_parent()
    children, quar = rs.reslice_record(parent, giant_literal)
    assert any(q["reason"] == "monolithic_unit_exceeds_context" for q in quar)


def test_no_fragment_exceeds_limit():
    # build a mid-size oversized function that must be isolated whole or quarantined
    body = "def mono():\n" + "".join(f"    x{i} = '{('a' * 120)}'\n" for i in range(700))
    parent = make_fake_parent()
    children, quar = rs.reslice_record(parent, body)
    assert all(c["token_count"] <= rs.CONTEXT_LIMIT for c in children)


# --------------------------------------------------------------------------- #
# 7. Quarantine behaviour
# --------------------------------------------------------------------------- #
def test_unparseable_file_quarantined():
    parent = make_fake_parent()
    children, quar = rs.reslice_record(parent, "def broken(:\n    this is not python\n")
    assert children == []
    assert any(q["reason"] == "file_not_parseable_quarantine" for q in quar)
    assert quar[0]["parent_sample_id"] == "parent-0000"


# --------------------------------------------------------------------------- #
# 8. Tokenizer fingerprint compatibility
# --------------------------------------------------------------------------- #
def test_children_tokenizer_fingerprint(real_meta=None):
    if real_meta is None:
        records = [json.loads(l) for l in (ARTIFACT / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        records = real_meta
    assert all(len(r["tokenizer_fingerprint"]) == 64 for r in records)
    assert all(r["tokenizer_fingerprint"] == PY_TAG for r in records)


# --------------------------------------------------------------------------- #
# 9 + 10. Exact and near-duplicate handling
# --------------------------------------------------------------------------- #
def test_exact_duplicate_canonical_selection():
    a = child_payload("dup = 1\n"); a["sample_id"] = "a" * 8
    b = child_payload("dup = 1\n"); b["sample_id"] = "b" * 8
    a["_text"] = b["_text"] = "dup = 1\n"
    assigned = dedup.assign_duplicate_groups([a, b])
    fp = {i: None for i in range(2)}
    members = list(assigned["groups"].values())[0]
    assert len(members) == 2


def test_near_duplicate_detected():
    t1 = "shared prefix one two three four five six seven eight nine ten\n"
    t2 = "shared prefix one two three four five six seven eight nine NINE\n"
    shingles1 = dedup.shingles(t1)
    shingles2 = dedup.shingles(t2)
    assert dedup.jaccard(shingles1, shingles2) > 0.5


# --------------------------------------------------------------------------- #
# 11 + 12. Parent-level and duplicate-group leakage prevention
# --------------------------------------------------------------------------- #
def test_parent_and_dup_colocation():
    children = []
    for i in range(6):
        parent = f"parent-{i % 2}"          # 2 parents, 3 children each
        c = child_payload(f"def f_{i}():\n    return {i}\n")
        c["parent_sample_id"] = parent
        c["original_index"] = i
        children.append(c)
    for c in children:
        c["duplicate_group"] = "dup-x"     # one big dup group across parents
    assignments = snap.assign_splits_parent_aware(children)
    dup_splits = {assignments[c["sample_id"]] for c in children}
    assert len(dup_splits) == 1            # dup group never spans splits
    parent_splits = {assignments[c["sample_id"]] for c in children if c["parent_sample_id"] == "parent-0"}
    assert len(parent_splits) == 1         # all children of one parent co-locate


# --------------------------------------------------------------------------- #
# 13 + 14. Split determinism + integrity (real artifact)
# --------------------------------------------------------------------------- #
def test_split_assignment_deterministic():
    kids = []
    for i in range(8):
        c = child_payload(f"def g_{i}():\n    return {i}\n")
        c["parent_sample_id"] = f"p{i % 3}"
        c["sample_id"] = f"sample-{i:04d}"
        kids.append(c)
    a1 = snap.assign_splits_parent_aware(kids)
    a2 = snap.assign_splits_parent_aware(kids)
    assert a1 == a2


def test_real_artifact_no_group_spans_splits():
    records = [json.loads(l) for l in (ARTIFACT / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_group = {}
    for r in records:
        by_group.setdefault(r.get("duplicate_group"), set()).add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())


def test_real_artifact_split_summary():
    snap_json = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    assert snap_json["split_counts"] == {"train": 954, "val": 51, "test": 47}
    assert snap_json["split_tokens"]["train"] + snap_json["split_tokens"]["val"] + snap_json["split_tokens"]["test"] == 1_380_638


# --------------------------------------------------------------------------- #
# 15. License-condition preservation
# --------------------------------------------------------------------------- #
def test_license_conditions_carried_into_children():
    parent = make_fake_parent()
    children, _ = rs.reslice_record(parent, BUNDLE)
    for c in children:
        assert c["license_id"] == parent["license_id"]
        assert c["license_decision_ref"] == parent["license_decision_ref"]
        assert c["provenance_ref"] == parent["provenance_ref"]
        # the freeze never implies external distribution rights
        assert "configs/data/l070_license_decisions.yaml" in c["license_decision_ref"]


# --------------------------------------------------------------------------- #
# 16 + 17 + 18. Training permission; owner fail-closed; technical vs legal
# --------------------------------------------------------------------------- #
def test_training_eligibility_blocked_for_everything():
    eligible = json.loads((ARTIFACT / "eligibility.json").read_text(encoding="utf-8"))
    aggregate = eligible["aggregate"]
    assert aggregate["training_eligible"] == 0
    assert aggregate["verdict"].startswith("TRAINING_ELIGIBILITY_BLOCKED")
    for ev in eligible["evaluations"]:
        assert ev["training_eligible"] is False
        assert ev["dimensions"]["owner_authorization_for_run"] == "MISSING"
        assert ev["dimensions"]["training_use_permission"] == "CONDITIONAL"


def test_technical_eligibility_counted_separately():
    eligible = json.loads((ARTIFACT / "eligibility.json").read_text(encoding="utf-8"))
    agg = eligible["aggregate"]
    assert agg["technically_eligible"] == 1050
    assert agg["technically_eligible"] != agg["training_eligible"]


def test_owner_authorization_never_granted():
    eligible = json.loads((ARTIFACT / "eligibility.json").read_text(encoding="utf-8"))
    for ev in eligible["evaluations"]:
        assert ev["dimensions"]["owner_authorization_for_run"] != "PASS"


# --------------------------------------------------------------------------- #
# 19. Synthetic-data exclusion
# --------------------------------------------------------------------------- #
def test_freeze_contains_real_corpus_only():
    records = [json.loads(l) for l in (ARTIFACT / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert all(r["source_id"] == "src-taksh-l04-code-corpus" for r in records)


def test_synthetic_fixtures_never_eligible():
    synth = l071ing.ingest_synthetic_fixtures()
    assert all(r["training_eligibility"] == "INELIGIBLE" for r in synth)


# --------------------------------------------------------------------------- #
# 20. Benchmark contamination rejection
# --------------------------------------------------------------------------- #
def test_contamination_registry_intact_and_clean():
    artifact = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    assert artifact["contamination"]["decision"].startswith("CLEAN")
    assert artifact["contamination"]["benchmarks_intact"] is True


def test_contaminated_text_blocks():
    bench = json.loads((REPO / "evaluation" / "l02_python_bench_v1.json").read_text(encoding="utf-8"))
    probe = json.dumps(bench["problems"][0], sort_keys=True)
    from evaluation.l070_contamination import scan_corpus

    out = scan_corpus(["harmless unique xyzzy content", probe])
    assert out["decision"].startswith("BLOCKED")


# --------------------------------------------------------------------------- #
# 21. Manifest compatibility
# --------------------------------------------------------------------------- #
def test_manifest_compatible_gate_closed(tmp_path):
    out = tmp_path / "compat.json"
    rc = mcheck.main(["--out", str(out)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["manifest_valid"] is True
    assert report["authorization_gate"]["l0601_gate"].startswith("CLOSED")
    assert report["environment"]["frozen_for_training"] is False


# --------------------------------------------------------------------------- #
# 22 + 23. Snapshot reproducibility, integrity, 24. immutability of parent
# --------------------------------------------------------------------------- #
def test_snapshot_integrity_hashes_match_records():
    snap_json = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    fps = sorted(cr.record_fingerprint(r) for r in
                 [json.loads(l) for l in (ARTIFACT / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])
    recomputed = hashlib.sha256("".join(fps).encode()).hexdigest()
    assert snap_json["integrity_hashes"]["records_sha256"] == recomputed


def test_snapshot_id_recorded():
    snap_json = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    assert snap_json["snapshot_id"].startswith("l072-l04-")
    assert snap_json["frozen_for_training"] is False
    assert snap_json["frozen_verdict"] == "FROZEN_RESEARCH_ONLY"


def test_parent_l071_snapshot_immutable():
    parent_snap = json.loads((PARENT / "snapshot.json").read_text(encoding="utf-8"))
    l072_snap = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    actual_parent_hash = hashlib.sha256((PARENT / "snapshot.json").read_bytes()).hexdigest()
    assert l072_snap["parent_snapshot"]["snapshot_sha256"] == actual_parent_hash
    assert parent_snap["snapshot_id"] == l072_snap["parent_snapshot"]["snapshot_id"]


def test_snapshot_atomic_write_no_temp_leaves(tmp_path):
    target = tmp_path / "x.json"
    snap.atomic_write(target, "{}")
    assert target.read_text(encoding="utf-8") == "{}"
    assert list(tmp_path.glob("x.json.*.tmp")) == []


# --------------------------------------------------------------------------- #
# 25. DLCAE registry preservation
# --------------------------------------------------------------------------- #
def test_registries_preserved():
    for path, expected in [
        ("configs/feasibility/l051_languages.yaml", "075d4b568f052c8ebede69ccaf21cc4d0fd15a4f01df455f1431505c354ef39d"),
        ("configs/foundation/l060_foundation.yaml", "01a90464e6f2155177a1032430286ce3c991997863a637d6aef9c7cf5e4a2fa3"),
    ]:
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected


# --------------------------------------------------------------------------- #
# 26. Karkain status preservation
# --------------------------------------------------------------------------- #
def test_karkain_remains_discovered_and_unqualified():
    assert kark.validate() == []
    import yaml

    strategy = yaml.safe_load((REPO / "configs" / "data" / "l070_karkain_data_strategy.yaml").read_text(encoding="utf-8"))
    assert strategy["identity"]["qualification_status"] == "DISCOVERED"
    assert strategy["identity"]["spec_version_pinned"] == "1.1.0"


# --------------------------------------------------------------------------- #
# 27 + 28. Resource limits + interruption recovery
# --------------------------------------------------------------------------- #
def test_resource_preflight_honest():
    pre = snap.resource_preflight()
    assert pre["memory_floor_bytes"] == 128 * 1024 * 1024
    assert pre["disk"]["free_bytes"] > 0
    assert pre["preflight_ok"] in (True, False)


def test_snapshot_main_blocks_when_starved(monkeypatch, tmp_path):
    class _VM:
        total = 4 * 2**30
        available = 8 * 1024 * 1024
        percent = 99.9

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    rc = snap.main(["--out-dir", str(tmp_path / "blocked")])
    assert rc == 2
    assert not (tmp_path / "blocked" / "snapshot.json").exists()


# --------------------------------------------------------------------------- #
# 29. L0.7.1 snapshot preserved byte-identical
# --------------------------------------------------------------------------- #
def test_l071_snapshot_bytes_unchanged():
    digest = hashlib.sha256((PARENT / "snapshot.json").read_bytes()).hexdigest()
    l072_snap = json.loads((ARTIFACT / "snapshot.json").read_text(encoding="utf-8"))
    assert l072_snap["parent_snapshot"]["snapshot_sha256"] == digest


# --------------------------------------------------------------------------- #
# 30. Benchmarks and checkpoints preserved
# --------------------------------------------------------------------------- #
def test_frozen_python_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA


def test_multilingual_benchmark_preserved():
    assert hashlib.sha256((REPO / "evaluation" / "l051_multilingual_benchmark_v1.json").read_bytes()).hexdigest() == MULTILINGUAL_SHA


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
