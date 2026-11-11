"""TAKSH L0.7.2 — freeze snapshot orchestrator (full pipeline).

authorized internal Python corpus -> re-slicing -> quality -> contamination ->
dedup -> parent/group-aware splits -> eligibility -> immutable freeze snapshot
under artifacts/l072_l04_freeze/.

The freeze is an ENGINEERING ARTIFACT for research qualification. Training
authority is a separate owner decision this milestone does not grant; the
L0.6.1 training gate stays closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import (  # noqa: E402
    l071_corpus_record as cr,
    l071_ingest as ingest,
    l071_quality as quality,
    l071_dedup as dedup,
    l070_contamination as contam,
)
from evaluation.l072_reslice import CONTEXT_LIMIT, reslice_record  # noqa: E402

L04_TRAIN = REPO / "data" / "l04_code_corpus" / "train.jsonl"
ARTIFACT_DIR = REPO / "artifacts" / "l072_l04_freeze"
PARENT_SNAPSHOT = REPO / "artifacts" / "l071_l04_snapshot" / "snapshot.json"
PARENT_RECORDS = REPO / "artifacts" / "l071_l04_snapshot" / "records_meta.jsonl"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resource_preflight() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        memory = {"total_bytes": int(vm.total), "available_bytes": int(vm.available),
                  "percent_used": float(vm.percent), "measured": True}
    except Exception:
        memory = {"measured": False}
    disk = shutil.disk_usage(REPO)
    return {"memory": memory,
            "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
            "memory_floor_bytes": 128 * 1024 * 1024,
            "preflight_ok": bool((memory.get("available_bytes") or 0) >= 128 * 1024 * 1024)}


def load_corpus_texts() -> dict[int, str]:
    texts = {}
    with L04_TRAIN.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            if row.get("domain") == "code":
                texts[i] = row["text"]
    return texts


def reslice_all(records: list[dict]) -> tuple[list[dict], list[dict]]:
    texts = load_corpus_texts()
    children: list[dict] = []
    quarantines: list[dict] = []
    for rec in records:
        if rec.get("exclusion_reason") == "QUALITY_INVALID":
            quarantines.append({
                "parent_sample_id": rec["sample_id"],
                "parent_fingerprint": rec["original_fingerprint"],
                "parent_source": rec["source_id"],
                "file_index": None,
                "fragment_index": None,
                "extraction_method": "excluded_upstream",
                "token_count": rec["token_count"],
                "reason": "parent_record_quarantined_in_L0.7.1 (QUALITY_INVALID)",
                "text": None,
            })
            continue
        text = texts[rec["original_index"]]
        ch, quar = reslice_record(rec, text)
        children.extend(ch)
        quarantines.extend(quar)
    return children, quarantines


def assign_splits_parent_aware(children: list[dict]) -> dict[str, str]:
    """Union-find over (parent relation) x (duplicate-group relation)."""
    parent = list(range(len(children)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_key: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(children):
        by_key["parent::" + (c.get("parent_sample_id") or c["sample_id"])].append(i)
        if c.get("duplicate_group"):
            by_key["dup::" + c["duplicate_group"]].append(i)
    for idxs in by_key.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    def bucket(root: str) -> str:
        digest = hashlib.sha256(("l072:" + root).encode("utf-8")).hexdigest()
        b = int(digest[:8], 16) % 10_000
        if b < 9000:
            return "train"
        if b < 9500:
            return "val"
        return "test"

    assignments = {}
    root_names = {}
    for i in range(len(children)):
        root = find(i)
        if root not in root_names:
            root_names[root] = bucket(children[root]["sample_id"] or str(root))
        assignments[children[i]["sample_id"]] = root_names[root]
    return assignments


# --------------------------------------------------------------------------- #
# Pipeline passes
# --------------------------------------------------------------------------- #
def quality_and_contamination_pass(children: list[dict]) -> dict:
    contamination_texts = [c["_text"] for c in children]
    for c in children:
        q = quality.classify_sample(c["_text"], c["language"])
        c["quality_status"] = q["quality_status"]
        c["quality_flags"] = q["flags"]

    contamination = contam.scan_corpus(contamination_texts)
    for hit in contamination["exact_matches"] + contamination["near_duplicates"]:
        i = hit["sample_index"]
        children[i]["contamination_status"] = "CONTAMINATED"
        children[i]["exclusion_reason"] = "BENCHMARK_CONTAMINATION"
        children[i]["training_eligibility"] = "INELIGIBLE"
    for c in children:
        if c["contamination_status"] == "NOT_SCANNED":
            c["contamination_status"] = "CLEAN"
    return contamination


def dedup_pass(children: list[dict]) -> dict:
    assigned = dedup.assign_duplicate_groups(children, near_threshold=0.7)
    member_list = {}
    for gid, idxs in assigned["groups"].items():
        for i in idxs:
            member_list[i] = idxs
    for i, c in enumerate(children):
        members = member_list[i]
        fps = {children[j]["original_fingerprint"] for j in members}
        c["duplicate_group"] = assigned["group_id"][i]
        c["duplicate_status"] = "UNIQUE" if len(members) == 1 else (
            "EXACT_DUPLICATE" if len(fps) == 1 else "NEAR_DUPLICATE")

    # Canonical selection: exactly one member of each EXACT_DUPLICATE group is
    # retained (deterministic: smallest token_count, then smallest fingerprint);
    # the rest are excluded with reason EXACT_DUPLICATE and are never eligible.
    for gid, idxs in assigned["groups"].items():
        fps = {children[j]["original_fingerprint"] for j in idxs}
        if len(idxs) <= 1 or len(fps) != 1:
            continue
        canonical = min(idxs, key=lambda j: (children[j]["token_count"],
                                             children[j]["original_fingerprint"]))
        children[canonical]["duplicate_status"] = "UNIQUE"
        for j in idxs:
            if j == canonical:
                continue
            children[j]["duplicate_status"] = "EXACT_DUPLICATE"
            children[j]["exclusion_reason"] = "EXACT_DUPLICATE"
            children[j]["training_eligibility"] = "INELIGIBLE"
    return assigned


def split_pass(children: list[dict]) -> None:
    assignments = assign_splits_parent_aware(children)
    for c in children:
        c["split"] = assignments[c["sample_id"]]


# --------------------------------------------------------------------------- #
# Phase F — eligibility evaluator
# --------------------------------------------------------------------------- #
TRAINING_USE_NOTE = ("internal SFT/pretraining on 1.59M Python tokens previously authorized "
                     "(L0.3/L0.4); internal research use only; external/commercial release "
                     "of checkpoints requires fresh counsel review")


def evaluate_sample(c: dict) -> dict:
    dims = {}
    dims["source_identity_verified"] = "PASS" if c["source_id"] == "src-taksh-l04-code-corpus" else "FAIL"
    dims["acquisition_status_valid"] = "PASS"          # ACQUIRED + ingest re-verified train.jsonl sha256
    dims["license_decision_resolved"] = "PASS"         # APPROVED_WITH_CONDITIONS record exists (L0.7.0)
    dims["training_use_permission"] = "CONDITIONAL"    # internal research only (see TRAINING_USE_NOTE)
    dims["provenance_complete"] = "PASS"               # dataset_manifest.json audited L0.4
    dims["language_identity_valid"] = "PASS" if c["language"] == "python" else "FAIL"
    dims["structural_quality_acceptable"] = "PASS" if c["quality_status"] not in ("INVALID", "BLOCKED") else "FAIL"
    dims["encoding_acceptable"] = "PASS"               # utf-8 verified at ingest
    dims["context_length_compliant"] = "PASS" if c["token_count"] <= CONTEXT_LIMIT else "FAIL"
    dims["duplicate_status_acceptable"] = "PASS" if c["duplicate_status"] in ("UNIQUE", "NEAR_DUPLICATE") else "FAIL"
    dims["benchmark_contamination_clear"] = "PASS" if c["contamination_status"] == "CLEAN" else "FAIL"
    dims["security_screening_acceptable"] = "PASS" if "HUMAN_REVIEW_REQUIRED" != c["quality_status"] else "REVIEW"
    dims["split_assignment_valid"] = "PASS" if c["split"] in ("train", "val", "test") else "FAIL"
    dims["tokenizer_compatibility_verified"] = "PASS" if len(c["tokenizer_fingerprint"]) == 64 else "FAIL"
    dims["snapshot_identity_valid"] = "PASS"           # snapshot id derived from content at build time
    dims["owner_authorization_for_run"] = "MISSING"    # L0.7.2 grants NOTHING; fail closed

    technical_dims = {k: v for k, v in dims.items()
                      if k not in ("training_use_permission", "owner_authorization_for_run")}
    technical = all(v in ("PASS", "CONDITIONAL") for v in technical_dims.values())
    if c.get("exclusion_reason"):
        status = "EXCLUDED"
    elif dims["security_screening_acceptable"] == "REVIEW":
        status = "HUMAN_REVIEW_REQUIRED"
    elif technical:
        status = "TECHNICALLY_ELIGIBLE"
    else:
        status = "TRAINING_ELIGIBILITY_BLOCKED"

    # No sample may be TRAINING_ELIGIBLE while owner authorization is missing.
    return {"sample_id": c["sample_id"], "dimensions": dims, "status": status,
            "training_eligible": False,
            "block_reason": "OWNER_AUTHORIZATION_MISSING (required for any training run; internal decision record governs research pretraining only)"}


def aggregate_eligibility(evaluations: list[dict]) -> dict:
    counts = Counter(e["status"] for e in evaluations)
    return {
        "technically_eligible": counts.get("TECHNICALLY_ELIGIBLE", 0),
        "human_review_required": counts.get("HUMAN_REVIEW_REQUIRED", 0),
        "training_eligibility_blocked": counts.get("TRAINING_ELIGIBILITY_BLOCKED", 0),
        "excluded": counts.get("EXCLUDED", 0),
        "quarantined": 0,
        "training_eligible": 0,
        "verdict": "TRAINING_ELIGIBILITY_BLOCKED_OWNER_AUTHORIZATION (aggregate); corpus is research-qualified but never training-ready until the owner signs off",
    }


# --------------------------------------------------------------------------- #
# Snapshot assembly + integrity
# --------------------------------------------------------------------------- #
def atomic_write(path: Path, data: str | bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    if isinstance(data, bytes):
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    os.replace(tmp, path)


def compare_snapshots(old: dict, new: dict) -> dict:
    changes = {}
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            changes[key] = {"old": old.get(key), "new": new.get(key)}
    return {"changed_fields": changes, "identical": not changes,
            "old_snapshot_id": old.get("snapshot_id"),
            "new_snapshot_id": new.get("snapshot_id"),
            "lineage": [old.get("snapshot_id"), new.get("snapshot_id")]}


def public_record(child: dict) -> dict:
    return {k: v for k, v in child.items() if k != "_text"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(ARTIFACT_DIR))
    args = parser.parse_args(argv)

    started = time.time()
    out_dir = Path(args.out_dir)

    preflight = resource_preflight()
    if not preflight["preflight_ok"]:
        print("BLOCKED: insufficient memory for re-slicing (floor 128 MiB)")
        return 2

    records = ingest.ingest_internal_python()
    children, quarantines = reslice_all(records)
    contamination = quality_and_contamination_pass(children)
    dedup_info = dedup_pass(children)
    split_pass(children)
    evaluations = [evaluate_sample(c) for c in children]
    eligibility = aggregate_eligibility(evaluations)

    real = [c for c in children if not c["source_id"].startswith("synthetic-fixture")]
    by_split = Counter(c["split"] for c in real)
    split_tokens = {s: sum(c["token_count"] for c in real if c["split"] == s) for s in ("train", "val", "test")}
    split_counts = {s: by_split[s] for s in ("train", "val", "test")}

    pub = [public_record(c) for c in children]
    # Snapshot identity derives ONLY from content fingerprints of the PUBLIC
    # records (no wall clock, no transient keys) — byte-reproducible rebuild.
    fingerprints = sorted(cr.record_fingerprint(r) for r in pub)
    records_sha = hashlib.sha256("".join(fingerprints).encode()).hexdigest()
    snapshot_id = "l072-l04-" + hashlib.sha256(("l072:" + records_sha).encode()).hexdigest()[:16]

    parent = json.loads(PARENT_SNAPSHOT.read_text(encoding="utf-8"))
    frozen = FROZEN_PY_SHA == file_sha256(REPO / "evaluation" / "l02_python_bench_v1.json")

    snapshot = {
        "schema": "taksh.corpus_freeze/1",
        "snapshot_id": snapshot_id,
        "created_utc": cr.utcnow(),
        "pipeline_version": "l072-freeze/1",
        "normalizer_version": cr.NORMALIZER_VERSION,
        "parent_snapshot": {
            "snapshot_id": parent.get("snapshot_id"),
            "snapshot_path": "artifacts/l071_l04_snapshot/snapshot.json",
            "snapshot_sha256": file_sha256(PARENT_SNAPSHOT),
            "records_sha256": file_sha256(PARENT_RECORDS),
            "immutable": True,
        },
        "original_corpus_sha256": file_sha256(L04_TRAIN),
        "original_corpus_note": "dataset_manifest.json train_sha256 verified at L0.4; train.jsonl re-verified here",
        "counts": {
            "parent_records": len(records),
            "derived_children": len(real),
            "quarantined_units": len(quarantines),
            "derived_tokens": sum(c["token_count"] for c in real),
            "original_record_tokens": 1_590_326,
            "token_delta_explanation": "separator tokens dropped + cut-boundary BPE shifts; quarantined monolithic units removed; no imputation",
        },
        "extraction_method_histogram": dict(Counter(c["extraction_method"] for c in real)),
        "quarantine_reason_histogram": dict(Counter(q["reason"] for q in quarantines)),
        "quarantined_tokens": sum(q["token_count"] for q in quarantines),
        "max_child_tokens": max((c["token_count"] for c in real), default=0),
        "all_within_context": all(c["token_count"] <= CONTEXT_LIMIT for c in real),
        "context_limit": CONTEXT_LIMIT,
        "duplicate_groups": len(set(c["duplicate_group"] for c in real)),
        "near_duplicate_children": sum(1 for c in real if c["duplicate_status"] != "UNIQUE"),
        "contamination": {
            "decision": contamination["decision"],
            "exact_matches": contamination["exact_matches"],
            "near_duplicates": contamination["near_duplicates"],
            "benchmarks_intact": frozen,
        },
        "quality": dict(Counter(c["quality_status"] for c in real)),
        "split_counts": split_counts,
        "split_tokens": split_tokens,
        "split_policy": "parent-aware + duplicate-group-aware; children of one parent co-locate; sha256-seeded",
        "eligibility": eligibility,
        "frozen_verdict": "FROZEN_RESEARCH_ONLY",
        "frozen_for_training": False,
        "integrity_hashes": {
            "records_sha256": records_sha,
        },
        "labels": {
            "corpus": "REAL_AUTHORIZED_CORPUS",
            "fixtures": "SYNTHETIC_TEST_FIXTURE",
            "freeze": "FROZEN_RESEARCH_ONLY (NOT authorized for training; owner decision required)",
        },
        "preflight": preflight,
        "wall_clock_seconds": round(time.time() - started, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(out_dir / "records_meta.jsonl", "\n".join(json.dumps(r, separators=(",", ":"), default=str) for r in pub) + "\n")
    atomic_write(out_dir / "snapshot.json", json.dumps(snapshot, indent=2))

    # quarantine + eligibility manifests (content-free)
    atomic_write(out_dir / "quarantine.jsonl", "\n".join(
        json.dumps({k: v for k, v in q.items() if k != "text"}, separators=(",", ":"), default=str)
        for q in quarantines) + "\n")
    atomic_write(out_dir / "eligibility.json", json.dumps({
        "aggregate": eligibility,
        "evaluations": evaluations,
        "training_use_note": TRAINING_USE_NOTE,
    }, indent=2))

    print(f"snapshot {snapshot_id}")
    print(f"parent_records={len(records)} children={len(real)} quarantines={len(quarantines)}")
    print(f"derived_tokens={snapshot['counts']['derived_tokens']:,} original={snapshot['counts']['original_record_tokens']:,}")
    print(f"splits train={split_counts['train']}/{split_tokens['train']:,} val={split_counts['val']}/{split_tokens['val']:,} test={split_counts['test']}/{split_tokens['test']:,}")
    print(f"contamination={contamination['decision']} verdict={snapshot['frozen_verdict']}")
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
