"""Merge per-model M7 evaluation records into one auditable summary.

The evaluator is run one model and one task group at a time so a long CPU job
cannot lose every result if it is interrupted. This script only reads records;
it never generates text and never changes a score.

Primary generation and secondary tasks are kept in separate raw directories.
The original ``records.jsonl`` files are never overwritten. A combined copy is
written as ``records_combined.jsonl`` for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.m7_evaluate import SCHEMA, aggregate, sha256_file

MODEL_ORDER = ("untrained", "m3_base", "m4_tuned", "m7_candidate")


def _read_records(directory: Path) -> list[dict]:
    path = directory / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8")
            .splitlines() if line.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Merge M7 evaluation records")
    parser.add_argument("--protocol", default="configs/eval/m7_protocol.yaml")
    parser.add_argument("--primary-dir", default="runs/m7_eval")
    parser.add_argument("--secondary-dir", default="runs/m7_eval_secondary")
    parser.add_argument("--out-dir", default="runs/m7_eval")
    args = parser.parse_args(argv)
    primary_dir, secondary_dir = Path(args.primary_dir), Path(args.secondary_dir)
    out_dir = Path(args.out_dir)

    models, missing, sources = [], [], {}
    for name in MODEL_ORDER:
        primary = _read_records(primary_dir / name)
        secondary = _read_records(secondary_dir / name)
        if not primary:
            missing.append(name)
            continue
        merged = {(r["task"], r["id"]): r for r in primary}
        for record in secondary:
            merged[(record["task"], record["id"])] = record
        records = list(merged.values())
        summary_path = primary_dir / name / "summary.json"
        base = json.loads(summary_path.read_text(encoding="utf-8"))
        summary = aggregate(name, base["checkpoint"], records,
                            sum(r.get("elapsed_seconds", 0) for r in records))
        summary["checkpoint_sha256"] = records[0].get("checkpoint_sha256")
        summary["record_sources"] = {
            "primary": str(primary_dir / name / "records.jsonl"),
            "secondary": (str(secondary_dir / name / "records.jsonl")
                          if secondary else None),
            "merged": str(out_dir / name / "records_combined.jsonl"),
        }
        (out_dir / name).mkdir(parents=True, exist_ok=True)
        with (out_dir / name / "records_combined.jsonl").open(
                "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        models.append(summary)
        sources[name] = summary["record_sources"]

    if not models:
        raise SystemExit("no per-model records found")
    combined = {
        "schema": SCHEMA,
        "protocol": args.protocol,
        "protocol_sha256": sha256_file(args.protocol),
        "dataset_manifest": "data/m7/m7-code-supervised-v0.manifest.json",
        "dataset_manifest_sha256": sha256_file(
            "data/m7/m7-code-supervised-v0.manifest.json"),
        "partitions": ["locked"],
        "primary_tasks": ["code_generation"],
        "primary_max_new_tokens": 192,
        "secondary_max_new_tokens": 128,
        "models": models,
        "missing_models": missing,
        "record_sources": sources,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"merged {len(models)} models -> {path}")
    for summary in models:
        generation = summary.get("partitions", {}).get("locked", {}).get(
            "tasks", {}).get("code_generation", {})
        print(f"  {summary['model']:12} records={summary['records']} "
              f"syntax={generation.get('syntax_rate')} "
              f"functional={generation.get('functional_rate')}")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
