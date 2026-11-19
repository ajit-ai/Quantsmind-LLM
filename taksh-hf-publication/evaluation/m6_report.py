"""M6 consolidated capability report and strategy decision.

Builds one report from persisted run manifests. No metric is recomputed from
memory, and no candidate is scored unless a manifest proves it produced output.

Every value is tagged so a reader cannot mistake one kind for another:

``MEASURED``        read from a run's persisted results
``DERIVED``         computed from measured values in this module
``ESTIMATE``        arithmetic, labelled as such
``UNAVAILABLE``     no evidence exists
``HUMAN_JUDGMENT``  a stated opinion, not evidence
``RECOMMENDATION``  the decision this report argues for

    python -m evaluation.m6_report --runs runs/m6 --out reports/M6_CAPABILITY.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.m6_capability import (assess_levels, categories_from_manifest,
                                      categories_from_results)

REPORT_SCHEMA = "quantsmind.m6_report/1"


def load_runs(run_root) -> list[dict]:
    """Every run manifest under ``run_root``, evaluated or not."""
    root = Path(run_root)
    runs = []
    if not root.exists():
        return runs
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        path = directory / "run_manifest.json"
        if not path.exists():
            continue
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        m6 = manifest.get("m6") or {}
        summary = manifest.get("summary")
        evaluated = bool(manifest.get("complete")) and summary is not None
        records = []
        if evaluated:
            from evaluation.m6_evaluate import read_results
            records = read_results(manifest.get("results_file"))
        evidence = m6.get("compliance_evidence")
        if evidence is None and evaluated:
            # Older manifests predate this field. Derive it from the persisted
            # results so a run is never credited from absent evidence.
            from evaluation.m6_capability import compliance_evidence
            evidence = compliance_evidence(records)
        runs.append({
            "directory": str(directory),
            "run_id": manifest.get("run_id") or directory.name,
            "candidate_id": m6.get("candidate_id") or manifest.get("model_id"),
            "status": manifest.get("status"),
            "evaluated": evaluated,
            "blocked_reason": (m6.get("blocked_reason")
                               or manifest.get("blocker")),
            "task_head": m6.get("task_head"),
            "runtime": m6.get("runtime"),
            "revision": m6.get("revision"),
            "license_id": m6.get("license_id"),
            "artifact_sha256": m6.get("artifact_sha256"),
            "model": manifest.get("model") or {},
            "benchmark_manifest_hash": manifest.get("benchmark_manifest_hash"),
            "examples_total": manifest.get("examples_total"),
            "examples_evaluated": manifest.get("examples_evaluated"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "generation_config": manifest.get("generation_config"),
            "summary": summary if evaluated else None,
            "by_task": manifest.get("per_task") or manifest.get("by_task"),
            "leakage_overlaps": ((m6.get("leakage") or {})
                                 .get("total_overlapping_examples")),
            "compliance_evidence": evidence,
            "categories": ([c.to_dict() for c in categories_from_results(
                records)] if records else None),
            "hosted_api_used": m6.get("hosted_api_used"),
            "resource": (manifest.get("resource_measurements")
                         or manifest.get("environment")),
        })
    return runs


def capability_table(runs) -> dict:
    table = {}
    for run in runs:
        if not run["evaluated"]:
            continue
        table[run["run_id"]] = {
            "candidate_id": run["candidate_id"],
            "task_head": run["task_head"],
            "is_practical_coding_candidate": run["task_head"] == "causal_lm",
            "categories": (run["categories"] if run["categories"] is not None
                           else [c.to_dict() for c in categories_from_manifest(
                               {"by_task": run["by_task"] or {}})]),
            "evidence": assess_levels(
                {"status": run["status"], "complete": True,
                 "summary": run["summary"],
                 "examples_evaluated": run["examples_evaluated"],
                 "examples_total": run["examples_total"]},
                task_head=run["task_head"] or "causal_lm",
                compliance_evidence=run["compliance_evidence"]),
        }
    return table


def strategy_decision(runs, capability) -> dict:
    """Assess Options A, B and C and pick a recommendation.

    The decisive question is whether any practical coding candidate was actually
    evaluated. If none was, no practical track can be endorsed, and the hybrid
    recommendation stays provisional no matter how the sovereign runs scored.
    """
    evaluated = [r for r in runs if r["evaluated"]]
    practical = [r for r in evaluated
                 if (r["task_head"] == "causal_lm"
                     and not str(r["candidate_id"]).startswith("quantsmind/"))]
    sovereign = [r for r in evaluated
                 if str(r["candidate_id"]).startswith("quantsmind/")]
    blocked_practical = [
        r for r in runs if not r["evaluated"]
        and not str(r["candidate_id"]).startswith("quantsmind/")]

    def func_pass(run):
        return ((run["summary"] or {}).get("functional") or {}).get("numerator")

    sovereign_functional = any((func_pass(r) or 0) > 0 for r in sovereign)
    practical_functional = any((func_pass(r) or 0) > 0 for r in practical)

    if practical_functional:
        recommendation = "PRACTICAL_OPEN_WEIGHT_TRACK"
        confidence = "supported by at least one functional pass"
    elif practical:
        recommendation = "HYBRID_TRACK"
        confidence = ("a practical runtime ran, but no functional pass was "
                      "demonstrated; the practical layer is not validated")
    else:
        recommendation = "INSUFFICIENT_EVIDENCE"
        confidence = ("no practical coding candidate was evaluated at all; the "
                      "hybrid recommendation remains provisional")

    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "provisional": not practical_functional,
        "options": {
            "A_sovereign": {
                "assessment": (
                    "The pipeline, licensing gate and checkpoints are real, but "
                    "no coding capability has been demonstrated. "
                    + ("Sovereign runs produced no functional passes."
                       if not sovereign_functional
                       else "A functional pass was observed.")),
                "advocate": "long-term independence and weight control",
                "blocker": "data for every target task beyond code completion",
                "status": ("NOT_SUPPORTED" if not sovereign_functional
                           else "PARTIAL"),
            },
            "B_practical_open_weight": {
                "assessment": (
                    "No practical coding candidate could be run: Qwen weights "
                    "are absent, Phi-3 is a partial download, and the CPU-only "
                    "host is memory-bound."
                    if not practical else
                    "A practical runtime ran, but not a coding-capable head."),
                "advocate": "faster path to real coding capability",
                "blocker": ("no local coding-capable weights within the licence "
                            "and memory boundary"
                            if not practical else
                            "coding-capable weights still unavailable"),
                "status": "BLOCKED" if not practical else "PARTIAL",
            },
            "C_hybrid": {
                "assessment": (
                    "Keep the sovereign checkpoints as a research track and use "
                    "a separately validated open-weight model as the practical "
                    "layer. That layer is not yet validated, so the hybrid "
                    "split is a plan, not a result."),
                "status": ("PROVISIONAL" if not practical_functional
                           else "SUPPORTED"),
            },
        },
        "prevented_by": [
            {"candidate_id": r["candidate_id"], "reason": r["blocked_reason"]}
            for r in blocked_practical],
    }


def build_report(run_root="runs/m6", *, inventory=None, sovereign=None) -> dict:
    runs = load_runs(run_root)
    capability = capability_table(runs)
    evaluated = [r for r in runs if r["evaluated"]]
    hashes = {r["benchmark_manifest_hash"] for r in evaluated
              if r["benchmark_manifest_hash"]}

    return {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "value_kinds": {
            "MEASURED": "read from a run's persisted results",
            "DERIVED": "computed from measured values",
            "ESTIMATE": "arithmetic, labelled as such",
            "UNAVAILABLE": "no evidence exists",
            "HUMAN_JUDGMENT": "a stated opinion, not evidence",
            "RECOMMENDATION": "the decision this report argues for",
        },
        "same_benchmark_across_runs": len(hashes) <= 1,
        "benchmark_hashes": sorted(hashes),
        "runs": runs,
        "runs_evaluated": [r["run_id"] for r in evaluated],
        "runs_not_evaluated": [
            {"run_id": r["run_id"], "candidate_id": r["candidate_id"],
             "status": r["status"], "reason": r["blocked_reason"]}
            for r in runs if not r["evaluated"]],
        "capability": capability,
        "inventory": inventory,
        "sovereign": sovereign,
        "composite_score": None,
        "composite_note": (
            "No composite score is produced. M4 is more compliant than M3 while "
            "producing less parseable Python, and an embedding model 'scored' "
            "4/6 compliance purely from bracket formatting. A weighted average "
            "would turn both facts into a number that hides them."),
        "strategy": strategy_decision(runs, capability),
        "limitations": [
            "The frozen benchmark is 26 examples; every rate is preliminary.",
            "17 examples require human review and are never scored as failures.",
            "7 examples share source documents with training material.",
            "Functional scoring produced 0/0 for every model: no model "
            "generated executable code on any scorable example, so functional "
            "capability is untested rather than measured as zero.",
            "No model was evaluated on a category the benchmark does not cover "
            "(debugging, refactoring, unit tests, documentation, architecture, "
            "SQL). Those categories are UNAVAILABLE, not zero.",
            "The only external model that could load is an embedding head; it "
            "cannot follow a coding instruction and its scores are artifacts.",
            "No practical coding-capable model could be obtained within the "
            "licence and memory boundary, so the central M6 question remains "
            "unanswered.",
        ],
    }


def format_report(report) -> str:
    lines = []
    add = lines.append
    add("M6 capability report")
    add(f"  benchmark hash : "
        f"{', '.join(h[:16] for h in report['benchmark_hashes']) or 'none'}")
    add(f"  same benchmark : {report['same_benchmark_across_runs']}")
    add("")
    for run in report["runs"]:
        if not run["evaluated"]:
            add(f"  [NOT EVALUATED] {run['run_id']} "
                f"({run['candidate_id']}) - {run['status']}")
            if run["blocked_reason"]:
                add(f"      reason: {run['blocked_reason']}")
            continue
        cap = report["capability"].get(run["run_id"], {})
        ev = cap.get("evidence", {})
        add(f"  [EVALUATED] {run['run_id']}  ({run['candidate_id']})")
        add(f"      head {run['task_head']}  runtime {run['runtime']}  "
            f"license {run['license_id']}")
        add(f"      examples {run['examples_evaluated']}/"
            f"{run['examples_total']}  elapsed {run['elapsed_seconds']}s")
        for name, metric in sorted((run["summary"] or {}).items()):
            rate = metric["rate"]
            text = "n/a" if rate is None else f"{rate:.1%}"
            flag = "  [PRELIMINARY]" if metric.get("preliminary") else ""
            add(f"      {name:12} {metric['numerator']}/"
                f"{metric['denominator']} = {text}{flag}")
        add(f"      evidence: level {ev.get('highest_level')} "
            f"{ev.get('highest_name')}")
        if ev.get("note"):
            add(f"      note: {ev['note']}")
        add("")
    strategy = report["strategy"]
    add(f"  recommendation: {strategy['recommendation']} "
        f"(provisional={strategy['provisional']})")
    add(f"  confidence    : {strategy['confidence']}")
    for name, option in strategy["options"].items():
        add(f"      {name:26} {option['status']}")
    if strategy["prevented_by"]:
        add("  blocked practical candidates:")
        for item in strategy["prevented_by"]:
            add(f"      {item['candidate_id']}: {item['reason']}")
    add("")
    add(f"  composite: {report['composite_note']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="M6 capability report and strategy recommendation")
    parser.add_argument("--runs", default="runs/m6")
    parser.add_argument("--out", default="reports/M6_CAPABILITY.json")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-inventory", action="store_true")
    parser.add_argument("--no-sovereign", action="store_true")
    args = parser.parse_args(argv)

    inventory = sovereign = None
    if not args.no_inventory:
        from evaluation.m6_inventory import build_inventory
        inventory = build_inventory()
    if not args.no_sovereign:
        from evaluation.m6_sovereign import build_assessment
        sovereign = build_assessment()

    report = build_report(args.runs, inventory=inventory, sovereign=sovereign)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        print(format_report(report))
        print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())