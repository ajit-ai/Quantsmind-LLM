"""Generate the M5 comparison report from actual run artifacts.

Nothing is recomputed from memory: every number is read from a run manifest on
disk, and a run that never executed is reported as ``NOT_RUN`` or ``BLOCKED`` --
never as a zero.

    python -m evaluation.m5_report --runs runs/m5 --out reports/M5_COMPARISON.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPORT_SCHEMA = "quantsmind.m5_comparison/1"


def load_run(run_dir) -> dict | None:
    path = Path(run_dir) / "run_manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect(run_root="runs/m5"):
    """Every run directory found, classified by what actually happened."""
    root = Path(run_root)
    runs = []
    if not root.exists():
        return runs
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = load_run(directory)
        if manifest is None:
            continue
        status = manifest.get("status")
        evaluated = bool(manifest.get("complete")) and "summary" in manifest
        runs.append({
            "directory": str(directory),
            "run_id": manifest.get("run_id"),
            "status": status,
            "evaluated": evaluated,
            "blocked_reason": manifest.get("blocker"),
            "benchmark_manifest_hash": manifest.get("benchmark_manifest_hash"),
            "scoring_version": manifest.get("scoring_version"),
            "model": manifest.get("model", {}),
            "generation_config": manifest.get("generation_config"),
            "examples_total": manifest.get("examples_total"),
            "examples_evaluated": manifest.get("examples_evaluated"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "environment": manifest.get("environment", {}),
            "git": manifest.get("git", {}),
            "errors": manifest.get("errors", []),
            "summary": manifest.get("summary") if evaluated else None,
            "by_task": manifest.get("by_task") if evaluated else None,
        })
    return runs


def verify_counts(run_dir, manifest) -> dict:
    """Confirm reported counts match the raw result records.

    A report whose totals disagree with its own results file is not trustworthy,
    so the check is recorded rather than assumed.
    """
    results_path = Path(run_dir) / "results.jsonl"
    if not results_path.exists():
        return {"checked": False, "reason": "no results file"}
    rows = [json.loads(line) for line in
            results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [r for r in rows if "metrics" in r]
    empty = [r for r in rows if not r.get("metrics")]
    reported = manifest.get("examples_evaluated")
    return {
        "checked": True,
        "raw_rows": len(rows),
        "rows_with_metrics": len(scored),
        "rows_with_empty_metrics": len(empty),
        "reported_examples_evaluated": reported,
        "consistent": reported is None or reported == len(scored),
    }


def comparison_report(run_root="runs/m5", benchmark_path="evaluation/m5_benchmark.jsonl"):
    runs = collect(run_root)
    evaluated = [r for r in runs if r["evaluated"]]
    not_run = [r for r in runs if not r["evaluated"]]

    bench_manifest_path = Path(str(benchmark_path).replace(".jsonl", ".manifest.json"))
    benchmark = (json.loads(bench_manifest_path.read_text(encoding="utf-8"))
                 if bench_manifest_path.exists() else {})

    hashes = {r["benchmark_manifest_hash"] for r in evaluated
              if r["benchmark_manifest_hash"]}
    for run in runs:
        run["count_verification"] = verify_counts(run["directory"],
                                                   load_run(run["directory"]) or {})

    # Per-metric scorecard across evaluated runs, one column per run.
    metric_names = sorted({name for run in evaluated
                           for name in (run["summary"] or {})})
    scorecard = {}
    for metric in metric_names:
        row = {}
        for run in evaluated:
            entry = (run["summary"] or {}).get(metric)
            row[run["run_id"]] = entry
        scorecard[metric] = row

    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "benchmark": {
            "path": str(benchmark_path),
            "benchmark_id": benchmark.get("benchmark_id"),
            "manifest_hash": benchmark.get("manifest_hash"),
            "examples": benchmark.get("examples"),
            "scorable": benchmark.get("scorable"),
            "requires_human_review": benchmark.get("requires_human_review"),
            "by_task": benchmark.get("validation", {}).get("by_task"),
            "leakage_overlaps": benchmark.get("leakage", {})
            .get("total_overlapping_examples"),
        },
        "same_benchmark_across_runs": len(hashes) <= 1,
        "benchmark_hashes_observed": sorted(hashes),
        "runs": runs,
        "runs_evaluated": [r["run_id"] for r in evaluated],
        "runs_not_evaluated": [
            {"run_id": r["run_id"], "status": r["status"],
             "reason": r["blocked_reason"]} for r in not_run
        ],
        "scorecard": scorecard,
        "composite_score": None,
        "composite_note": (
            "No weighted aggregate is produced. Blending syntax, functional and "
            "compliance would hide the finding that matters here: a model can "
            "comply with a format instruction while emitting no code at all."
        ),
    }


def format_report(report) -> str:
    lines = []
    add = lines.append
    bench = report["benchmark"]
    add("M5 comparison")
    add(f"  benchmark : {bench['benchmark_id']} "
        f"({bench['examples']} examples, {bench['scorable']} automatically scorable)")
    add(f"  hash      : {(bench['manifest_hash'] or '')[:16]}...")
    add(f"  leakage   : {bench['leakage_overlaps']} overlap(s)")
    add(f"  same set  : {report['same_benchmark_across_runs']}")
    add("")
    for run in report["runs"]:
        if not run["evaluated"]:
            add(f"  [NOT RUN] {run['run_id']}: {run['status']}")
            add(f"           reason: {run['blocked_reason']}")
            continue
        add(f"  [EVALUATED] {run['run_id']}  ({run['examples_evaluated']}/"
            f"{run['examples_total']} examples, {run['elapsed_seconds']}s)")
        model = run["model"] or {}
        add(f"           model: {model.get('model_id')}")
        add(f"           params: {model.get('parameters')}  "
            f"sha256: {(model.get('artifact_sha256') or '')[:16]}")
        for metric, entry in sorted((run["summary"] or {}).items()):
            rate = entry["rate"]
            rate_text = "n/a" if rate is None else f"{rate:.1%}"
            flag = "  [PRELIMINARY]" if entry["preliminary"] else ""
            add(f"           {metric:12} {entry['numerator']}/"
                f"{entry['denominator']} = {rate_text}{flag}")
        add("")
    add(f"  composite score: {report['composite_note']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate the M5 comparison report")
    parser.add_argument("--runs", default="runs/m5")
    parser.add_argument("--benchmark", default="evaluation/m5_benchmark.jsonl")
    parser.add_argument("--out", default="reports/M5_COMPARISON.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    report = comparison_report(args.runs, args.benchmark)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        print(format_report(report))
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())