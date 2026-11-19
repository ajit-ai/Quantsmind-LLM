"""Aggregate v2.2.0 evaluations, applying the negative-control guard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.generalization_eval import paired_delta, summarize
from training.generalization_tasks import (build_cells, build_eval_cases,
                                           validation_cases)


def _overlaps() -> dict:
    cases = build_eval_cases()
    train_prompt = set()
    for records in build_cells().values():
        train_prompt.update(f"{r['instruction']}\n{r['signature']}\n"
                            for r in records)
    validation_prompt = {f"{r['instruction']}\n{r['signature']}\n"
                         for r in validation_cases()}
    result = {}
    for part in ("E1", "E2", "E3", "E4", "E5", "E6"):
        selected = [c for c in cases if c["partition"] == part]
        result[part] = {
            "cases": len(selected),
            "exact_prompt_in_cells": sum(
                1 for c in selected
                if f"{c['instruction']}\n{c['signature']}\n" in train_prompt),
            "exact_prompt_in_validation": sum(
                1 for c in selected
                if f"{c['instruction']}\n{c['signature']}\n" in validation_prompt),
        }
    return result

RUNS = ("baseline", "cellA", "cellB", "cellC", "cellC_repeat")


def _load(root: Path, run: str, filename="generalization.json"):
    path = root / run / "evaluations" / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    for record in records:
        if record["partition"] == "E6":
            record["exact"] = False
            record["structural"] = False
            record["composition_match"] = False
    return data, records


def _by_partition(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["partition"]].append(record)
    return dict(grouped)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root)
    data = {}
    for run in RUNS:
        payload, records = _load(root, run)
        data[run] = {"payload": payload, "records": records,
                     "metrics": summarize(records),
                     "by_partition_records": _by_partition(records)}
    baseline = data["baseline"]["records"]
    comparisons = {}
    for run in RUNS[1:]:
        comparisons[run] = {
            "overall": paired_delta(baseline, data[run]["records"]),
            "by_partition": {
                key: paired_delta(data["baseline"]["by_partition_records"][key],
                                  data[run]["by_partition_records"][key])
                for key in sorted(data["baseline"]["by_partition_records"])
            },
        }
    repeat_a = {r["case_id"]: r["continuation"] for r in data["cellC"]["records"]}
    repeat_b = {r["case_id"]: r["continuation"]
                for r in data["cellC_repeat"]["records"]}
    repeat_identical = repeat_a == repeat_b
    analysis = {
        "version": "2.2.0",
        "suite_digest": data["baseline"]["payload"]["evaluation_suite_digest"],
        "contamination": {"held_out_wells": _overlaps(),
                          "verdict": "PASS if E1/E2/E4/E5/E6 are all zero; "
                                     "E3 duplicates are the base variant and "
                                     "intentionally in-distribution"},
        "runs": {run: data[run]["metrics"] for run in RUNS},
        "comparisons": comparisons,
        "repeat": {"cellC_repeat_identical": repeat_identical,
                   "cases_compared": len(repeat_a)},
    }
    supplement = root / "cellC_long2_eval.json"
    if supplement.is_file():
        payload = json.loads(supplement.read_text(encoding="utf-8"))
        records = payload["records"]
        for record in records:
            if record["partition"] == "E6":
                record["exact"] = False
                record["structural"] = False
                record["composition_match"] = False
        analysis["supplement_cellC_1500_steps"] = {
            "metrics": summarize(records),
            "comparison": {"overall": paired_delta(baseline, records),
                            "by_partition": {
                                key: paired_delta(
                                    data["baseline"]["by_partition_records"][key],
                                    [r for r in records
                                     if r["partition"] == key])
                                for key in sorted(data["baseline"]["by_partition_records"])
                            }},
        }
    path = root / "analysis.json"
    path.write_text(json.dumps(analysis, indent=1), encoding="utf-8")
    print(json.dumps({
        "runs": {run: {k: analysis["runs"][run]["overall"][k]
                       for k in ("cases", "parses", "exact", "structural",
                                 "binding", "control_flow", "composition_match")}
                 for run in RUNS},
        "repeat_identical": repeat_identical,
        "supplement": analysis.get("supplement_cellC_1500_steps", {}).get("metrics", {}).get("overall"),
    }, indent=1))
    print(path)


if __name__ == "__main__":
    main()
