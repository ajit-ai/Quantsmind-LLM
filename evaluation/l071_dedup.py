"""TAKSH L0.7.1 — deduplication and contamination protection.

Deterministic, auditable, duplicate-group-aware. Nothing is silently deleted:
every exclusion carries a reason and the audit record retains fingerprints.
Benchmark contamination uses the L0.7.0 registry (imported, not duplicated).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation import l071_corpus_record as cr  # noqa: E402
from evaluation.l070_contamination import fingerprint as bench_fingerprint  # noqa: E402
from evaluation.l070_contamination import scan_corpus as contamination_scan  # noqa: E402


def raw_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def shingles(text: str, size: int = 5) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < size:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a and b) else 0.0


def assign_duplicate_groups(records: list[dict], near_threshold: float = 0.7) -> dict:
    """Exact-match groups first, then near-duplicate merge on normalized text."""
    by_raw: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_raw[r["original_fingerprint"]].append(i)

    parent = list(range(len(records)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for idxs in by_raw.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    norm_shingles = [shingles(r.get("_text", "")) for r in records]
    for i in range(len(records)):
        if not norm_shingles[i]:
            continue
        for j in range(i + 1, len(records)):
            if find(i) == find(j):
                continue
            if jaccard(norm_shingles[i], norm_shingles[j]) >= near_threshold:
                union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)

    group_id = {i: f"dup-{find(i):08x}" for i in range(len(records))}
    return {"groups": {gid: idxs for gid, idxs in groups.items()},
            "group_id": group_id}


def audit_exclusions(records: list[dict]) -> list[dict]:
    audit = []
    for r in records:
        if r.get("exclusion_reason"):
            audit.append({"sample_id": r["sample_id"],
                          "original_fingerprint": r["original_fingerprint"],
                          "reason": r["exclusion_reason"]})
    return audit


def screen_benchmark_contamination(texts: list[str]) -> dict:
    return contamination_scan(texts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_dedup.json")
    args = parser.parse_args(argv)

    fixtures = [
        {"text": "def f(x):\n    return x * 2\n", "language": "python", "synthetic": True},
        {"text": "def f(x):\n    return x * 2\n", "language": "python", "synthetic": True},
        {"text": "def f(x):\n    y = x * 2\n    return y\n", "language": "python", "synthetic": True},
        {"text": "fn f(x: Int) -> Int { x * 2 }\n", "language": "karkain", "synthetic": True},
    ]
    tok_fp = hashlib.sha256((REPO / "tokenizer" / "taksh_code_bpe_v1.json").read_bytes()).hexdigest()
    records = []
    for i, f in enumerate(fixtures):
        rec = cr.build_record(source_id=f"synthetic-fixture-{i:03d}", source_version="l071-syn-dedup/1",
                              original_ref=f"synthetic[{i}]", original_index=i, text=f["text"],
                              language=f["language"], language_family="l1_target",
                              license_id="synthetic", license_decision_ref="n/a",
                              provenance_ref="configs/data/l071_pipeline_config.yaml#synthetic",
                              tokenizer_id="taksh_code_bpe_v1", tokenizer_version="1",
                              tokenizer_fingerprint=tok_fp, token_count=8, synthetic=True)
        rec["_text"] = f["text"]
        records.append(rec)

    assigned = assign_duplicate_groups(records, near_threshold=0.6)
    for i, r in enumerate(records):
        gid = assigned["group_id"][i]
        members = assigned["groups"][next(g for g, idxs in assigned["groups"].items() if i in idxs)]
        r["duplicate_group"] = gid
        r["duplicate_status"] = "UNIQUE" if len(members) == 1 else "EXACT_DUPLICATE"

    contamination = screen_benchmark_contamination([f["text"] for f in fixtures])
    audit = audit_exclusions(records)

    report = {
        "schema": "taksh.l071_dedup/1",
        "generated_by": "evaluation/l071_dedup.py",
        "fixtures_label": "synthetic",
        "groups": {g: idxs for g, idxs in assigned["groups"].items()},
        "statuses": [r["duplicate_status"] for r in records],
        "benchmark_contamination": {
            "decision": contamination["decision"],
            "exact_matches": contamination["exact_matches"],
            "near_duplicates": contamination["near_duplicates"],
        },
        "exclusion_audit": audit,
        "policy": "no silent deletion; groups preserved; benchmark collisions block eligibility",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"dedup: {len(assigned['groups'])} groups across {len(records)} synthetic samples; contamination: {contamination['decision']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
