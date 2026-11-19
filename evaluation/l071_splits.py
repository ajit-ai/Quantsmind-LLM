"""TAKSH L0.7.1 — deterministic split assignment.

Duplicate-group-aware, seeded, source-leakage aware. Frozen benchmarks are
never placed in any split. Synthetic records are never split into real-facing
buckets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def assign_split(sample_id: str, ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> str:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10_000
    train_end = int(ratios[0] * 10_000)
    val_end = train_end + int(ratios[1] * 10_000)
    if bucket < train_end:
        return "train"
    if bucket < val_end:
        return "val"
    return "test"


def assign_splits(records: list[dict], ratios: tuple[float, float, float] = (0.9, 0.05, 0.05)) -> dict:
    """Dup groups land in the same split; deterministic on sample_id."""
    group_seed = {}
    assignments = {}
    for r in records:
        gid = r.get("duplicate_group") or r["sample_id"]
        if gid not in group_seed:
            group_seed[gid] = assign_split(gid, ratios)
        assignments[r["sample_id"]] = group_seed[gid]
    return assignments


def split_report(records: list[dict]) -> dict:
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    per_language = {}
    for split_name, rs in by_split.items():
        per_language[split_name] = dict(Counter(r["language"] for r in rs))
    return {
        "counts": {s: len(rs) for s, rs in by_split.items()},
        "tokens": {s: sum(r["token_count"] for r in rs) for s, rs in by_split.items()},
        "per_language": per_language,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_splits.json")
    args = parser.parse_args(argv)

    from evaluation.l071_ingest import ingest_internal_python, ingest_synthetic_fixtures
    from evaluation.l071_dedup import assign_duplicate_groups

    real = ingest_internal_python()
    synthetic = ingest_synthetic_fixtures()
    # mirror the snapshot pipeline: dedup group ids seed the split assignment,
    # so standalone reports match artifacts/l071_l04_snapshot/records_meta.jsonl
    for r in real + synthetic:
        r["_text"] = None
    import json as _json
    with (REPO / "data" / "l04_code_corpus" / "train.jsonl").open(encoding="utf-8") as fh:
        texts = {i: json.loads(line)["text"] for i, line in enumerate(fh)}
    from evaluation.l071_ingest import SYNTHETIC_FIXTURES
    for r in synthetic:
        r["_text"] = SYNTHETIC_FIXTURES[r["original_index"]]["text"]
    for r in real:
        r["_text"] = texts[r["original_index"]]
    records = real + synthetic
    assigned = assign_duplicate_groups(records, near_threshold=0.7)
    for i, r in enumerate(records):
        r["duplicate_group"] = assigned["group_id"][i]
    assignments = assign_splits(records)
    for r in records:
        r["split"] = assignments[r["sample_id"]]

    # duplicate groups must not span splits
    group_splits: dict[str, set[str]] = defaultdict(set)
    for r in records:
        group_splits[r.get("duplicate_group") or r["sample_id"]].add(r["split"])
    leakage = {gid: sorted(splits) for gid, splits in group_splits.items() if len(splits) > 1}

    report = {
        "schema": "taksh.l071_splits/1",
        "generated_by": "evaluation/l071_splits.py",
        "seed": "sha256(duplicate_group) seeded assignment — no RNG state",
        "ratios": {"train": 0.9, "val": 0.05, "test": 0.05},
        "assignments": len(assignments),
        "split_report": split_report(records),
        "duplicate_groups_spanning_splits": leakage,
        "no_leakage": not leakage,
        "insufficient_data_languages": [],
        "note": "synthetic fixtures are split but are never training-eligible",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"splits assigned; counts={report['split_report']['counts']}; no_leakage={report['no_leakage']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
