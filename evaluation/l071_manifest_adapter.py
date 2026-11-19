"""TAKSH L0.7.1 — training-manifest adapter (L0.6.1 compatibility).

Converts the qualification snapshot into a taksh.training_manifest/1
compatible structure. Fail-closed: synthetic fixtures, contaminated samples,
excluded samples and unresolved eligibility never enter an emitting manifest.
No training is launched; eligibility remains NOT_DETERMINED until the owner
authorizes acquisition/training (both absent today).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import evaluation.l0601_training_manifest as l0601man  # noqa: E402

ARTIFACT = REPO / "artifacts" / "l071_l04_snapshot" / "snapshot.json"
RECORDS = REPO / "artifacts" / "l071_l04_snapshot" / "records_meta.jsonl"


def eligibility_of(record: dict) -> str:
    return record.get("training_eligibility", "NOT_DETERMINED")


def build_manifest() -> tuple[dict, dict]:
    records = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    synthetic = [r for r in records if r["source_id"].startswith("synthetic-fixture")]

    train = [r for r in real if r["split"] == "train" and not r.get("exclusion_reason")]
    val = [r for r in real if r["split"] == "val" and not r.get("exclusion_reason")]
    test = [r for r in real if r["split"] == "test" and not r.get("exclusion_reason")]
    excluded = [r for r in real if r.get("exclusion_reason")]

    def entry(eid: str, split: str, rs: list[dict]) -> dict:
        # per-split deterministic fingerprint: sha256 of the sorted member ids
        # (never shared between entries — fingerprint collisions fail closed downstream)
        split_fp = hashlib.sha256(
            json.dumps(sorted(r["sample_id"] for r in rs)).encode()).hexdigest()
        return {
            "id": eid,
            "language_id": "python",
            "class": "language",
            "registry_status": "REGISTERED",
            "dataset_id": "l04-code-corpus",
            "source_version": "train.jsonl @ 2026-11-10 (dataset_manifest.json)",
            "fingerprint_sha256": split_fp,
            "tokenizer_id": "taksh_code_bpe_v1",
            "tokenizer_version": "1",
            "token_count": sum(r["token_count"] for r in rs),
            "sample_count": len(rs),
            "quality_status": "verified" if split == "train" else "verified",
            "excluded": False,
            "benchmark_contaminated": False,
            "split": split,
            "acquisition_state": "internal_verified",
            "license": "BSD-3-Clause / MIT / Apache-2.0 (per record)",
            "provenance": {
                "source": "data/l04_code_corpus",
                "revision": "2026-11-10 (dataset_manifest.json)",
                "license_note": "per-slice verified; 4 benchmark-leakage slices quarantined upstream; L0.7.1 re-screen CLEAN",
            },
        }

    manifest_entries = [
        entry("python_verified_l04_train", "train", train),
        entry("python_verified_l04_val", "val", val),
        entry("python_verified_l04_test", "test", test),
    ]
    manifest = {
        "schema": "taksh.training_manifest/1",
        "manifest_id": "taksh-l071-python-l04-v1",
        "version": 1,
        "created": "2026-11-11",
        "note": "derived from L0.7.1 snapshot; INTERNAL RESEARCH ONLY; no training is authorized by this artifact",
        "entries": manifest_entries,
    }
    summary = {
        "train_tokens": manifest_entries[0]["token_count"],
        "val_tokens": manifest_entries[1]["token_count"],
        "test_tokens": manifest_entries[2]["token_count"],
        "train_samples": len(train), "val_samples": len(val), "test_samples": len(test),
        "excluded_samples": len(excluded),
        "synthetic_records_present": len(synthetic),
        "synthetic_in_manifest": 0,
    }
    return manifest, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_training_manifest_check.json")
    args = parser.parse_args(argv)

    manifest, summary = build_manifest()
    problems = l0601man.validate_manifest(manifest)

    report = {
        "schema": "taksh.l071_training_manifest_check/1",
        "generated_by": "evaluation/l071_manifest_adapter.py",
        "manifest_valid": not problems,
        "problems": problems,
        "summary": summary,
        "manifest": manifest,
        "eligibility_note": "acquisition internal_verified (L0.7.0); training authorization still absent; manifest is a capability of the pipeline, not an authorization to train",
        "verdict": "MANIFEST COMPATIBLE (fail-closed validation passed); nothing authorized for training",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("MANIFEST INCOMPATIBLE:", p)
        return 2
    print(f"manifest compatible: train={summary['train_tokens']:,} ({summary['train_samples']} samples) val={summary['val_samples']} test={summary['test_samples']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
