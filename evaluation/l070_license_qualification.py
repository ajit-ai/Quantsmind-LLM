"""TAKSH L0.7.0 — license qualification decision-record validator.

Loads configs/data/l070_license_decisions.yaml and enforces:

  * five+ assessment axes per candidate, never collapsed into one boolean;
  * unknown/ambiguous decisions fail closed (REJECTED or unresolved);
  * APPROVED* requires an evidence source + official reference + review date;
  * owner approval state is explicit and separate from legal decision;
  * nothing here is legal advice; records classify, owners decide.

The reusable heuristic license check comes from the L0.6.0 readiness module
(evaluation.l060_training_readiness.license_ok); this module adds the
decision-record layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DECISIONS_PATH = REPO / "configs" / "data" / "l070_license_decisions.yaml"

DECISIONS = frozenset({
    "APPROVED", "APPROVED_WITH_CONDITIONS",
    "HUMAN_LEGAL_REVIEW_REQUIRED", "REJECTED", "UNKNOWN",
})
AXES = [
    "source_license", "file_package_licenses", "dependency_licenses",
    "redistribution_restrictions", "commercial_use_permissions",
    "derivative_work_restrictions", "ml_training_permissions",
    "terms_of_service", "dataset_specific_restrictions", "copyright_attribution",
]
FAIL_CLOSED = frozenset({"HUMAN_LEGAL_REVIEW_REQUIRED", "UNKNOWN", "REJECTED"})


def load_decisions(path: Path = DECISIONS_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_record(record: dict) -> list[str]:
    problems: list[str] = []
    for field in ("source_id", "evidence_source", "exact_license_id", "official_reference",
                  "axes", "owner_approval_state", "overall_decision"):
        if field not in record or record[field] in (None, ""):
            problems.append(f"missing field: {field}")
    if problems:
        return problems

    axes = record["axes"]
    for axis in AXES:
        if axis not in axes:
            problems.append(f"missing axis: {axis}")
            continue
        decision = axes[axis].get("decision")
        if decision not in DECISIONS:
            problems.append(f"axis {axis} has invalid decision {decision!r}")
        if decision in ("APPROVED", "APPROVED_WITH_CONDITIONS"):
            if not axes[axis].get("note"):
                problems.append(f"axis {axis}: APPROVED* requires a note with conditions")
    if record["overall_decision"] not in DECISIONS:
        problems.append(f"overall_decision invalid: {record['overall_decision']!r}")

    overall = record["overall_decision"]
    if overall in ("APPROVED", "APPROVED_WITH_CONDITIONS"):
        if not record.get("official_reference") or record["official_reference"] == "TBD":
            problems.append("APPROVED* requires an official reference")
        # an APPROVED overall cannot have any FAIL_CLOSED axis
        for axis in AXES:
            if axes.get(axis, {}).get("decision") in FAIL_CLOSED:
                problems.append(f"overall APPROVED* but axis {axis} is {axes[axis]['decision']}")
    return problems


def summarize(data: dict) -> dict:
    from collections import Counter

    overall = Counter(r.get("overall_decision") for r in data.get("records", []))
    axis_blocked = Counter()
    for r in data.get("records", []):
        for axis, body in r.get("axes", {}).items():
            if body.get("decision") in FAIL_CLOSED:
                axis_blocked[axis] += 1
    return {"overall": dict(overall), "fail_closed_axes": dict(axis_blocked)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_license_decisions.json")
    args = parser.parse_args(argv)

    data = load_decisions()
    problems: list[str] = []
    if data.get("schema") != "taksh.license_qualification/1":
        problems.append("schema mismatch")
    if not data.get("review_date"):
        problems.append("review_date required")
    for record in data.get("records", []):
        for p in validate_record(record):
            problems.append(f"{record.get('source_id', '?')}: {p}")

    summary = summarize(data)
    report = {
        "schema": "taksh.l070_license_decisions/1",
        "generated_by": "evaluation/l070_license_qualification.py",
        "valid": not problems,
        "problems": problems,
        "summary": summary,
        "decision_records": [
            {"source_id": r["source_id"],
             "overall_decision": r["overall_decision"],
             "owner_approval_state": r["owner_approval_state"],
             "unresolved_questions": r.get("unresolved_questions", [])}
            for r in data.get("records", [])
        ],
        "no_legal_advice": "engineering decision records only; licensed counsel review required before any acquisition",
        "verdict": "NO SOURCE IS APPROVED FOR ACQUISITION; all external candidates require legal review",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("DECISIONS INVALID:", p)
        return 2
    print(f"license decisions valid; overall: {summary['overall']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
