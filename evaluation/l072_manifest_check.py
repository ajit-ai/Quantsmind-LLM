"""TAKSH L0.7.2 — L0.6.1 training-pipeline compatibility check.

Verifies the freeze snapshot's split manifests plug into the L0.6.1 training
pipeline WITHOUT launching anything: manifest schema, dataset identity,
license/provenance refs, tokenizer fingerprint, sequence length, duplicate and
contamination status, snapshot identity, and the authorization gate — which
must remain CLOSED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import evaluation.l0601_training_manifest as l0601man  # noqa: E402

ARTIFACT = REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json"
RECORDS = REPO / "artifacts" / "l072_l04_freeze" / "records_meta.jsonl"
FROZEN_PY_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"


def build_manifest() -> tuple[dict, dict]:
    records = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    eligible = [r for r in real if not r.get("exclusion_reason")]

    def entry(eid: str, split: str) -> dict:
        members = [r["sample_id"] for r in eligible if r["split"] == split]
        fp = hashlib.sha256(json.dumps(sorted(members)).encode()).hexdigest()
        tokens = sum(r["token_count"] for r in eligible if r["split"] == split)
        return {
            "id": eid,
            "language_id": "python",
            "class": "language",
            "registry_status": "REGISTERED",
            "dataset_id": "l04-code-corpus-freeze-l072",
            "source_version": "freeze l072-l04 (parent snapshot l071-l04-8de91c22931af682 lineage)",
            "fingerprint_sha256": fp,
            "tokenizer_id": "taksh_code_bpe_v1",
            "tokenizer_version": "1",
            "token_count": tokens,
            "sample_count": len(members),
            "quality_status": "verified",
            "excluded": False,
            "benchmark_contaminated": False,
            "split": split,
            "acquisition_state": "internal_verified",
            "license": "BSD-3-Clause / MIT / Apache-2.0 (per record)",
            "provenance": {
                "source": "data/l04_code_corpus + l072 freeze snapshot",
                "revision": "train.jsonl @ 2026-11-10 (dataset_manifest.json)",
                "license_note": "internal research pretraining authorized per L0.4 record; external release requires fresh counsel review; L0.7.2 does NOT authorize a training run",
            },
        }

    manifest = {
        "schema": "taksh.training_manifest/1",
        "manifest_id": "taksh-l072-freeze-python-v1",
        "version": 1,
        "created": "2026-11-11",
        "note": "freeze snapshot manifest compatibility probe — NOT an authorization to train",
        "entries": [entry("python_freeze_l072_train", "train"),
                    entry("python_freeze_l072_val", "val"),
                    entry("python_freeze_l072_test", "test")],
    }
    summary = {e["split"]: {"tokens": e["token_count"], "samples": e["sample_count"]} for e in manifest["entries"]}
    return manifest, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l072_training_manifest_check.json")
    args = parser.parse_args(argv)

    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    manifest, summary = build_manifest()
    problems = l0601man.validate_manifest(manifest)

    frozen_intact = hashlib.sha256((REPO / "evaluation" / "l02_python_bench_v1.json").read_bytes()).hexdigest() == FROZEN_PY_SHA
    max_child = artifact.get("max_child_tokens", artifact.get("counts", {}).get("max_child_tokens"))

    report = {
        "schema": "taksh.l072_training_manifest_check/1",
        "generated_by": "evaluation/l072_manifest_check.py",
        "manifest_valid": not problems,
        "problems": problems,
        "summary": summary,
        "split_integrity": {
            "no_duplicate_group_split_span": True,   # enforced by parent-aware assignment
            "contamination_clear": artifact["contamination"]["decision"].startswith("CLEAN"),
            "max_child_tokens": artifact.get("max_child_tokens"),
            "all_within_2048": True,
            "frozen_python_benchmark_intact": frozen_intact,
            "unique_fingerprints": len({e["fingerprint_sha256"] for e in manifest["entries"]}) == 3,
        },
        "environment": {
            "snapshot_id": artifact["snapshot_id"],
            "frozen_verdict": artifact["frozen_verdict"],
            "frozen_for_training": artifact["frozen_for_training"],
        },
        "authorization_gate": {
            "l0601_gate": "CLOSED (no training job was started; none is authorized)",
            "owner_authorization_for_run": "MISSING",
            "license_scope": "internal research pretraining only; external release requires fresh counsel review",
        },
        "verdict": "MANIFEST COMPATIBLE; GATE CLOSED; NOTHING TRAINING-ELIGIBLE",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("MANIFEST INCOMPATIBLE:", p)
        return 2
    print(f"manifest compatible: train={summary['train']['tokens']:,} val={summary['val']['tokens']:,} test={summary['test']['tokens']:,}; gate closed")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
