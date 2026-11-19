"""TAKSH L0.7.1 — corpus snapshot manager.

Runs the authorized internal Python corpus (plus synthetic fixtures, clearly
labeled) through the full engineering pipeline and emits a versioned,
self-verifying snapshot manifest under artifacts/l071_l04_snapshot/.

The internal corpus is processed under its documented conditions (internal
research; no model artifact; no distribution). The snapshot is NEVER frozen
for training: acquisition, legal, provenance and quality gates must ALL be
satisfied for that, and owner authorization is outstanding by definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402
    l071_corpus_record as cr,
    l071_ingest as ingest,
    l071_quality as quality,
    l071_dedup as dedup,
    l071_stats as stats,
    l071_splits as splits,
)

ARTIFACT_DIR = REPO / "artifacts" / "l071_l04_snapshot"
REGISTRY_REFS = [
    "configs/data/l070_source_registry.yaml",
    "configs/data/l070_license_decisions.yaml",
    "configs/data/l070_contamination_registry.yaml",
]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource_preflight() -> dict:
    import shutil

    try:
        import psutil

        vm = psutil.virtual_memory()
        memory = {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
                  "percent_used": float(vm.percent), "measured": True}
    except Exception:
        memory = {"total_bytes": None, "available_bytes": None, "measured": False}
    disk = shutil.disk_usage(REPO)
    ok = (memory.get("available_bytes") or 0) >= 128 * 1024 * 1024
    return {"memory": memory,
            "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "memory_floor_bytes": 128 * 1024 * 1024,
            "preflight_ok": bool(ok)}


def process_corpus(records: list[dict]) -> dict:
    """Quality -> dedup -> contamination -> splits -> stats. Mutates records in place."""
    for r in records:
        text = r["_text"]
        q = quality.classify_sample(text, r["language"])
        r["quality_status"] = q["quality_status"]

    # dedup (near + exact); group membership decides status
    assigned = dedup.assign_duplicate_groups(records, near_threshold=0.7)
    group_of: dict[int, list[int]] = assigned["groups"]
    member_list = {}
    for gid, idxs in group_of.items():
        for i in idxs:
            member_list[i] = idxs
    for i, r in enumerate(records):
        members = member_list[i]
        fps = {records[j]["original_fingerprint"] for j in members}
        r["duplicate_group"] = assigned["group_id"][i]
        r["duplicate_status"] = "UNIQUE" if len(members) == 1 else ("EXACT_DUPLICATE" if len(fps) == 1 else "NEAR_DUPLICATE")

    # benchmark contamination on the REAL records (synthetic flagged CLEAN by policy of their own class)
    texts = [r["_text"] for r in records]
    contamination = dedup.screen_benchmark_contamination(texts)
    for hit in contamination["exact_matches"]:
        records[hit["sample_index"]]["contamination_status"] = "CONTAMINATED"
        records[hit["sample_index"]]["exclusion_reason"] = "BENCHMARK_CONTAMINATION"
        records[hit["sample_index"]]["training_eligibility"] = "INELIGIBLE"
    for hit in contamination["near_duplicates"]:
        records[hit["sample_index"]]["contamination_status"] = "CONTAMINATED"
        records[hit["sample_index"]]["exclusion_reason"] = "BENCHMARK_CONTAMINATION_NEAR"
        records[hit["sample_index"]]["training_eligibility"] = "INELIGIBLE"
    for r in records:
        if r["contamination_status"] == "NOT_SCANNED":
            r["contamination_status"] = "CLEAN"
        if r["quality_status"] == "INVALID":
            r["exclusion_reason"] = "QUALITY_INVALID"
            r["training_eligibility"] = "INELIGIBLE"

    assignments = splits.assign_splits(records)
    for r in records:
        r["split"] = assignments[r["sample_id"]]

    # eligibility decision: only for REAL, uncontaminated, non-excluded, dup-canonical records
    for r in records:
        if r["source_id"].startswith("synthetic-fixture"):
            r["training_eligibility"] = "INELIGIBLE"
        elif r["exclusion_reason"]:
            r["training_eligibility"] = "INELIGIBLE"
        elif r["contamination_status"] == "CONTAMINATED":
            r["training_eligibility"] = "INELIGIBLE"
        else:
            r["training_eligibility"] = "NOT_DETERMINED"   # owner authorization still required

    return contamination


def eligibility_gates(contamination_decision: str, records: list[dict]) -> dict:
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    contaminated = sum(1 for r in real if r["contamination_status"] == "CONTAMINATED")
    gates = {
        "acquisition_authorized": False,   # L0.7.0: zero authorizations
        "legal_review_complete": True,     # internal l04 APPROVED_WITH_CONDITIONS (internal research)
        "provenance_verified": True,       # dataset_manifest.json audited L0.4; train.jsonl sha256 verified at ingest
        "quality_gate_pass": all(r["quality_status"] not in ("INVALID", "BLOCKED")
                                  for r in real if not r.get("exclusion_reason")),
        "contamination_screen_pass": contamination_decision.startswith("CLEAN") and contaminated == 0,
        "owner_authorization_recorded": False,
    }
    gates["frozen_for_training"] = all(gates.values())
    return gates


def compare_snapshots(old: dict, new: dict) -> dict:
    changes = {}
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            changes[key] = {"old": old.get(key), "new": new.get(key)}
    return {"changed_fields": changes,
            "identical": not changes,
            "old_snapshot_id": old.get("snapshot_id"),
            "new_snapshot_id": new.get("snapshot_id")}


def build_snapshot(records: list[dict], contamination: dict, preflight: dict) -> tuple[dict, list[dict]]:
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    synthetic = [r for r in records if r["source_id"].startswith("synthetic-fixture")]

    fingerprint_blob = "".join(sorted(cr.record_fingerprint(r) for r in real)).encode()
    snapshot_id = "l071-l04-" + hashlib.sha256(fingerprint_blob).hexdigest()[:16]

    per_language = stats.per_language_stats(records)
    split_report = splits.split_report(records)
    distribution = stats.distribution_report(records)
    contamination_decision = contamination["decision"]
    gates = eligibility_gates(contamination_decision, real)

    registry_versions = {ref: file_sha256(REPO / ref) for ref in REGISTRY_REFS}

    metadata_records = []
    for r in records:
        keep = {k: v for k, v in r.items() if k != "_text"}
        metadata_records.append(keep)

    snapshot = {
        "schema": "taksh.corpus_snapshot/1",
        "snapshot_id": snapshot_id,
        "created_utc": cr.utcnow(),
        "source_registry_versions": registry_versions,
        "license_decision_refs": ["configs/data/l070_license_decisions.yaml"],
        "input_fingerprints": {
            "train_jsonl_sha256": file_sha256(REPO / "data" / "l04_code_corpus" / "train.jsonl"),
            "tokenizer_blob_sha256": "fbedf003774183c4d14ba891ea2751705a303b1946868b9699ee9713074ef7a7",
            "contamination_registry_sha256": registry_versions["configs/data/l070_contamination_registry.yaml"],
        },
        "pipeline_version": cr.PIPELINE_VERSION,
        "normalization_version": cr.NORMALIZER_VERSION,
        "tokenizer_id": "taksh_code_bpe_v1",
        "tokenizer_fingerprint": stats.tokenizer_fingerprint(),
        "dedup": {"method": "exact sha256 + normalized sha256 + 5-gram shingle Jaccard>=0.7",
                  "config_ref": "configs/data/l071_pipeline_config.yaml"},
        "contamination_registry_version": registry_versions["configs/data/l070_contamination_registry.yaml"],
        "split_configuration": {"ratios": [0.9, 0.05, 0.05], "assignment": "sha256(dup_group) seeded"},
        "language_coverage": {"python": sum(r["token_count"] for r in real if r["language"] == "python")},
        "counts": {
            "real_records": len(real),
            "synthetic_records": len(synthetic),
            "real_tokens": sum(r["token_count"] for r in real),
            "synthetic_tokens": sum(r["token_count"] for r in synthetic),
            "unique_fingerprints": len({r["original_fingerprint"] for r in real}),
        },
        "rejection_statistics": {
            "excluded_total": sum(1 for r in real if r.get("exclusion_reason")),
            "exclusion_reasons": dict(__import__("collections").Counter(r["exclusion_reason"] for r in real if r.get("exclusion_reason"))),
        },
        "quality_statistics": dict(__import__("collections").Counter(r["quality_status"] for r in real)),
        "duplicate_statistics": dict(__import__("collections").Counter(r["duplicate_status"] for r in real)),
        "per_language_stats": per_language,
        "split_report": split_report,
        "distribution": distribution,
        "contamination": {
            "decision": contamination_decision,
            "benchmarks_verified_intact": True,
            "exact_matches": contamination["exact_matches"],
            "near_duplicates": contamination["near_duplicates"],
        },
        "integrity_hashes": {
            "records_sha256": hashlib.sha256(
                json.dumps(metadata_records, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest(),
        },
        "gates": gates,
        "frozen_for_training": gates["frozen_for_training"],
        "labels": {
            "python corpus": "REAL_AUTHORIZED_CORPUS",
            "fixtures": "SYNTHETIC_TEST_FIXTURE",
            "candidate sources": "CANDIDATE_SOURCE (not ingested)",
            "coverage elsewhere": "UNVERIFIED_SOURCE",
        },
        "preflight": preflight,
    }
    return snapshot, metadata_records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight = resource_preflight()
    if not preflight["preflight_ok"]:
        print("BLOCKED: available memory below floor for processing")
        return 2

    real = ingest.ingest_internal_python()
    synthetic = ingest.ingest_synthetic_fixtures()

    # carry raw text separately (records must not persist content)
    import json as _json

    texts = {}
    with (REPO / "data" / "l04_code_corpus" / "train.jsonl").open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            row = _json.loads(line)
            if row.get("domain") == "code":
                texts[idx] = row["text"]
    for r in real:
        r["_text"] = texts[r["original_index"]]
    for r in synthetic:
        r["_text"] = ingest.SYNTHETIC_FIXTURES[r["original_index"]]["text"]
    records = real + synthetic

    contamination = process_corpus(records)
    snapshot, metadata_records = build_snapshot(records, contamination, preflight)

    snapshot_path = out_dir / "snapshot.json"
    records_path = out_dir / "records_meta.jsonl"

    import os
    import tempfile

    fd, tmp_snapshot = tempfile.mkstemp(dir=out_dir, prefix="snapshot_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, indent=2))
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp_snapshot, snapshot_path)          # atomic on the same filesystem

    fd, tmp_records = tempfile.mkstemp(dir=out_dir, prefix="records_", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        for r in metadata_records:
            fh.write(json.dumps(r, separators=(",", ":"), default=str) + "\n")
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp_records, records_path)            # atomic

    print(f"snapshot {snapshot['snapshot_id']}")
    print(f"real={snapshot['counts']['real_records']} synthetic={snapshot['counts']['synthetic_records']} tokens={snapshot['counts']['real_tokens']:,}")
    print(f"contamination: {snapshot['contamination']['decision']}")
    print(f"frozen_for_training={snapshot['frozen_for_training']} (owner authorization outstanding)")
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
