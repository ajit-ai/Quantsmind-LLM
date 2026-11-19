"""TAKSH L0.8.0-PRE — C: corpus and eligibility verification (read-only).

Verifies the frozen snapshot l072-l04-fdb1fee2beaae9c2 — identity, splits,
contamination, eligibility boundaries — without modifying anything. Any
inconsistency is reported, never silently repaired.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l071_corpus_record import record_fingerprint  # noqa: E402

SNAPSHOT_DIR = REPO / "artifacts" / "l072_l04_freeze"
FROZEN_BENCH_SHA = "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"
MULTILINGUAL_SHA = "7a839060c236c557a7ad77b36020ec858af09ecac7e952394357de5034d9064e"


def sha256_file(path: Path) -> str:
    return __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest()


def verify() -> dict:
    snapshot = json.loads((SNAPSHOT_DIR / "snapshot.json").read_text(encoding="utf-8"))
    records = [json.loads(l) for l in (SNAPSHOT_DIR / "records_meta.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # 1) identity + integrity
    fingerprints = sorted(record_fingerprint(r) for r in records)
    records_sha = __import__("hashlib").sha256("".join(fingerprints).encode()).hexdigest()
    id_ok = snapshot["snapshot_id"] == "l072-l04-fdb1fee2beaae9c2" and snapshot["integrity_hashes"]["records_sha256"] == records_sha

    # 2) splits + leakage
    dup_groups = {}
    leakage = 0
    for r in records:
        gid = r.get("duplicate_group")
        dup_groups.setdefault(gid, set()).add(r["split"])
    for gid, splits in dup_groups.items():
        if len(splits) > 1:
            leakage += 1

    # 3) eligibility split
    eligible = [r for r in records if not r.get("exclusion_reason")]
    contaminated = [r for r in records if r.get("contamination_status") == "CONTAMINATED"]
    by_split = Counter(r["split"] for r in eligible)

    # 4) benchmarks intact (re-hash)
    bench_ok = sha256_file(REPO / "evaluation" / "l02_python_bench_v1.json") == FROZEN_BENCH_SHA
    multi_ok = sha256_file(REPO / "evaluation" / "l051_multilingual_benchmark_v1.json") == MULTILINGUAL_SHA

    # 5) artifact frozen verdict
    frozen_verdict = snapshot.get("frozen_verdict")

    return {
        "schema": "taksh.l080pre_corpus_verify/1",
        "generated_by": "evaluation/l080pre/corpus_verify.py",
        "snapshot_id": snapshot["snapshot_id"],
        "parent_snapshot": snapshot.get("parent_snapshot", {}).get("snapshot_id") if isinstance(snapshot.get("parent_snapshot"), dict) else snapshot.get("parent_snapshot"),
        "identity_and_integrity_verified": bool(id_ok),
        "records": len(records),
        "derived_children": snapshot["counts"]["derived_children"],
        "eligible_children": len(eligible),
        "eligible_tokens": sum(r["token_count"] for r in eligible),
        "excluded_children": len(records) - len(eligible),
        "contaminated_children": len(contaminated),
        "leakage_groups_spanning_splits": leakage,
        "split_counts": dict(by_split),
        "max_child_tokens": snapshot.get("max_child_tokens"),
        "all_within_context": snapshot.get("all_within_context"),
        "benchmark_python_intact": bench_ok,
        "benchmark_multilingual_intact": multi_ok,
        "frozen_verdict": frozen_verdict,
        "frozen_for_training": snapshot.get("frozen_for_training"),
        "training_authorized": False,
        "language_python_only": sorted({r["language"] for r in records}) == ["python"],
        "eligibility_note": "technical eligibility NEVER implies legal/training eligibility; frozen verdict remains FROZEN_RESEARCH_ONLY",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l080pre_corpus_verify.json")
    args = parser.parse_args(argv)

    result = verify()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    ok = result["identity_and_integrity_verified"] and result["benchmark_python_intact"] and result["benchmark_multilingual_intact"]
    if not ok:
        print("CORPUS VERIFY FAILED:", {k: v for k, v in result.items() if k.startswith(("identity", "benchmark"))})
        return 2
    print(f"corpus verified: records={result['records']} eligible={result['eligible_children']} tokens={result['eligible_tokens']:,} verdict={result['frozen_verdict']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
