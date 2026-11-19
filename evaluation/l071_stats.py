"""TAKSH L0.7.1 — tokenization statistics and language distribution.

Integrates the canonical tokenizer (read-only). All reported token counts are
measured by this run or are explicitly labeled estimates. Synthetic records
are counted separately and never as real corpus. No tokens are invented.
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

from evaluation.l051_language_registry import load_registry as load_languages  # noqa: E402

CANONICAL_TOK = REPO / "tokenizer" / "taksh_code_bpe_v1.json"
CONTEXT_LIMIT = 2048


def tokenizer_fingerprint() -> str:
    return hashlib.sha256(CANONICAL_TOK.read_bytes()).hexdigest()


def per_language_stats(records: list[dict]) -> dict[str, dict]:
    by_lang: dict[str, list[dict]] = {}
    for r in records:
        by_lang.setdefault(r["language"], []).append(r)

    stats = {}
    for lang, rs in by_lang.items():
        real_rs = [r for r in rs if not r["source_id"].startswith("synthetic-fixture")]
        synth_rs = [r for r in rs if r["source_id"].startswith("synthetic-fixture")]
        real_length_values = [r["token_count"] for r in real_rs]
        buckets = {"<=64": 0, "65-256": 0, "257-1024": 0, "1025-2048": 0, ">2048": 0}
        for n in real_length_values:
            if n <= 64:
                buckets["<=64"] += 1
            elif n <= 256:
                buckets["65-256"] += 1
            elif n <= 1024:
                buckets["257-1024"] += 1
            elif n <= CONTEXT_LIMIT:
                buckets["1025-2048"] += 1
            else:
                buckets[">2048"] += 1
        exclusions = Counter(r["exclusion_reason"] for r in rs if r.get("exclusion_reason"))
        by_quality = Counter(r.get("quality_status", "NOT_CHECKED") for r in rs)
        by_dup = Counter(r.get("duplicate_status", "NOT_SCANNED") for r in rs)
        real_tokens = sum(r["token_count"] for r in real_rs)
        synth_tokens = sum(r["token_count"] for r in synth_rs)
        stats[lang] = {
            "raw_samples": len(real_rs),
            "unique_fingerprints": len({r["original_fingerprint"] for r in real_rs}),
            "token_count": real_tokens,
            "synthetic_tokens": synth_tokens,
            "avg_tokens_per_sample": round(real_tokens / len(real_length_values), 2) if real_length_values else 0,
            "max_sequence_length": max(real_length_values) if real_length_values else 0,
            "context_limit": CONTEXT_LIMIT,
            "exceeding_context": buckets[">2048"],
            "length_distribution": buckets,
            "by_quality": dict(by_quality),
            "by_duplicate_status": dict(by_dup),
            "exclusion_reasons": dict(exclusions),
            "synthetic_records": sum(1 for r in rs if r["source_id"].startswith("synthetic-fixture")),
        }
    return stats


def distribution_report(records: list[dict]) -> dict:
    languages = {l["id"]: l["category"] for l in load_languages()["languages"]}
    real = [r for r in records if not r["source_id"].startswith("synthetic-fixture")]
    synthetic = [r for r in records if r["source_id"].startswith("synthetic-fixture")]

    def family_payload(rs):
        out: dict[str, int] = {}
        for r in rs:
            fam = r.get("language_family") or languages.get(r["language"], "unknown")
            out[fam] = out.get(fam, 0) + r["token_count"]
        return out

    return {
        "real_corpus_tokens": sum(r["token_count"] for r in real),
        "synthetic_tokens": sum(r["token_count"] for r in synthetic),
        "real_corpus_tokens_by_family": family_payload(real),
        "languages_with_real_data": sorted({r["language"] for r in real if r["token_count"] > 0}),
        "languages_without_data": sorted(set(languages) - {r["language"] for r in real}),
    }


def _attach_processing(records: list[dict]) -> list[dict]:
    """Apply the same quality/dedup/contamination/split chain as the snapshot,
    so the standalone stats report agrees with artifacts/l071_l04_snapshot."""
    from evaluation.l071_quality import classify_sample
    from evaluation.l071_dedup import assign_duplicate_groups, screen_benchmark_contamination
    from evaluation.l071_splits import assign_splits

    texts = {i: json.loads(l)["text"] for i, l in enumerate(
        (REPO / "data" / "l04_code_corpus" / "train.jsonl").open(encoding="utf-8"))
        if json.loads(l).get("domain") == "code"}
    from evaluation.l071_ingest import SYNTHETIC_FIXTURES

    for r in records:
        if r["source_id"].startswith("synthetic-fixture"):
            r["_text"] = SYNTHETIC_FIXTURES[r["original_index"]]["text"]
        else:
            r["_text"] = texts[r["original_index"]]

    for r in records:
        r["quality_status"] = classify_sample(r["_text"], r["language"])["quality_status"]

    assigned = assign_duplicate_groups(records, near_threshold=0.7)
    member_list = {}
    for gid, idxs in assigned["groups"].items():
        for i in idxs:
            member_list[i] = idxs
    for i, r in enumerate(records):
        members = member_list[i]
        fps = {records[j]["original_fingerprint"] for j in members}
        r["duplicate_group"] = assigned["group_id"][i]
        r["duplicate_status"] = "UNIQUE" if len(members) == 1 else ("EXACT_DUPLICATE" if len(fps) == 1 else "NEAR_DUPLICATE")

    contamination = screen_benchmark_contamination([r["_text"] for r in records])
    for hit in contamination["exact_matches"] + contamination["near_duplicates"]:
        records[hit["sample_index"]]["contamination_status"] = "CONTAMINATED"
        records[hit["sample_index"]]["exclusion_reason"] = "BENCHMARK_CONTAMINATION"
    for r in records:
        if r["contamination_status"] == "NOT_SCANNED":
            r["contamination_status"] = "CLEAN"
        if r["quality_status"] == "INVALID":
            r["exclusion_reason"] = "QUALITY_INVALID"

    assignments = assign_splits(records)
    for r in records:
        r["split"] = assignments[r["sample_id"]]
    for r in records:
        r.pop("_text", None)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_stats.json")
    args = parser.parse_args(argv)

    from evaluation.l071_ingest import ingest_internal_python, ingest_synthetic_fixtures

    real = ingest_internal_python()
    synthetic = ingest_synthetic_fixtures()
    records = _attach_processing(real + synthetic)
    per_language = per_language_stats(records)
    dist = distribution_report(records)

    report = {
        "schema": "taksh.l071_stats/1",
        "generated_by": "evaluation/l071_stats.py",
        "tokenizer_fingerprint": tokenizer_fingerprint(),
        "total_records": len(records),
        "real_records": len(real),
        "synthetic_records": len(synthetic),
        "per_language": per_language,
        "distribution": dist,
        "labels": {
            "l04_python_corpus": "REAL_AUTHORIZED_CORPUS",
            "fixtures": "SYNTHETIC_TEST_FIXTURE",
        },
        "policy": "token counts measured at ingest; synthetic tokens are reported separately; nothing estimated from file sizes",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    py = per_language.get("python", {})
    print(f"python (REAL): {py.get('raw_samples')} samples, {py.get('token_count'):,} tokens, avg {py.get('avg_tokens_per_sample')}")
    for lang, s in sorted(per_language.items()):
        print(f"{lang:12s} {s['raw_samples']:4d} samples {s['token_count']:>8,} tokens")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
