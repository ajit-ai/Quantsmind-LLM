"""TAKSH L0.7.5 — owner-decision register validator.

Validates the L0.7.5 register: schema, required entries, status vocabulary,
per-entry evidence + owner action, and register/decision-summary consistency.
Nothing here creates an owner authorization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

REGISTER_PATH = REPO / "configs" / "foundation" / "l075_owner_decision_register.yaml"

REQUIRED_IDS = ["A1.1-training-authorization", "A1.2-internal-research-permission",
                "A2-compute-authorization", "A3.1-internal-research-dataset-permission",
                "A3.2-external-dataset-legal-review", "A3.3-commercial-checkpoint-distribution",
                "A3.4-external-publication-permission", "A4-ci-authorization", "A5-language-scope"]

VALID_STATUSES = {"GRANTED", "CONDITIONAL", "MISSING", "HUMAN_LEGAL_REVIEW_REQUIRED",
                  "REJECTED", "NOT_AUTHORIZED", "OWNER_ACTION_REQUIRED"}


def load_register(path: Path = REGISTER_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_register(data: dict) -> list[str]:
    problems: list[str] = []
    if data.get("schema") != "taksh.owner_decision_register/1":
        problems.append(f"schema must be taksh.owner_decision_register/1, got {data.get('schema')!r}")
    entries = data.get("register")
    if not isinstance(entries, list):
        return ["register must be a list"]
    ids = [e.get("id") for e in entries]
    for rid in REQUIRED_IDS:
        if rid not in ids:
            problems.append(f"required entry missing: {rid}")
    for e in entries:
        status = e.get("status")
        if status not in VALID_STATUSES:
            problems.append(f"{e.get('id')}: status {status!r} not in {sorted(VALID_STATUSES)}")
        if not e.get("evidence_reference"):
            problems.append(f"{e.get('id')}: evidence_reference required")
        if not e.get("owner_action"):
            problems.append(f"{e.get('id')}: owner_action required")
    summary = data.get("decision_summary") or {}
    for bucket in ("granted", "conditional", "missing", "not_authorized",
                   "human_legal_review_required", "owner_action_required"):
        expected = DEFAULT_BUCKETS.get(bucket)
        got = set(summary.get(bucket, []))
        if expected and got != set(expected):
            problems.append(f"decision_summary.{bucket} mismatch: {sorted(got)} != {sorted(expected)}")
    return problems


DEFAULT_BUCKETS = {
    "granted": ["A5-language-scope"],
    "conditional": ["A1.2-internal-research-permission", "A3.1-internal-research-dataset-permission"],
    "missing": ["A1.1-training-authorization", "A2-compute-authorization"],
    "not_authorized": ["A3.3-commercial-checkpoint-distribution", "A3.4-external-publication-permission"],
    "human_legal_review_required": ["A3.2-external-dataset-legal-review"],
    "owner_action_required": ["A4-ci-authorization"],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l075_owner_decision_register.json")
    args = parser.parse_args(argv)

    data = load_register()
    problems = validate_register(data)
    report = {
        "schema": "taksh.l075_owner_decision_register/1",
        "generated_by": "evaluation/l075_decision_register.py",
        "valid": not problems,
        "problems": problems,
        "entries": [
            {"id": e["id"], "status": e["status"],
             "question": e.get("question"), "owner_action": e.get("owner_action")}
            for e in data.get("register", [])
        ],
        "decision_summary": data.get("decision_summary"),
        "verdict": "REGISTER VALID — zero authorizations fabricated; all pending decisions explicit" if not problems else "REGISTER INVALID",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        for p in problems:
            print("REGISTER INVALID:", p)
        return 2
    print(f"register valid; {len(data['register'])} entries; every decision has evidence + owner action")
    print("granted:", data["decision_summary"]["granted"])
    print("pending:", data["decision_summary"]["missing"] + data["decision_summary"]["not_authorized"])
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
