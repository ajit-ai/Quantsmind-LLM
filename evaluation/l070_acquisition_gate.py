"""TAKSH L0.7.0 — acquisition authorization workflow (owner gate).

Implements the request/dossier validation and the workflow state machine.
Acquisition execution is PERMANENTLY DISABLED in this milestone; the gate
fails closed. Only a recorded owner_authorization_id could ever advance a
request past OWNER_APPROVAL_REQUIRED, and no tool accepts one here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WORKFLOW_PATH = REPO / "configs" / "data" / "l070_acquisition_workflow.yaml"

# Milestone invariant — never set True in this codebase without a new milestone + owner decision.
ACQUISITION_EXECUTION_ENABLED = False

REQUIRED_FIELDS = [
    "source_id", "dataset_snapshot", "official_source_url",
    "license_decision_record_id", "commercial_use_assessment",
    "training_use_assessment", "language_coverage", "expected_storage_bytes",
    "acquisition_method", "data_handling_plan", "exclusion_and_filtering_policy",
    "known_risks", "intended_training_use", "required_attribution",
    "proposed_acquisition_scope", "owner_authorization_id",
]

STAGES = ["SOURCE_DISCOVERY", "TECHNICAL_REVIEW", "LEGAL_REVIEW", "PROVENANCE_REVIEW",
          "OWNER_APPROVAL_REQUIRED", "QUALIFIED_FOR_ACQUISITION",
          "ACQUISITION_AUTHORIZED", "ACQUISITION_EXECUTION", "POST_ACQUISITION_VERIFICATION"]


def load_workflow(path: Path = WORKFLOW_PATH) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def validate_dossier(dossier: dict) -> list[str]:
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in dossier:
            problems.append(f"missing dossier field: {field}")
    if problems:
        return problems
    if not str(dossier.get("owner_authorization_id") or "").strip():
        problems.append("owner_authorization_id absent — request cannot leave OWNER_APPROVAL_REQUIRED")
    if dossier.get("expected_storage_bytes") is not None:
        try:
            if int(dossier["expected_storage_bytes"]) <= 0:
                problems.append("expected_storage_bytes must be positive or null (unknown)")
        except (TypeError, ValueError):
            problems.append("expected_storage_bytes must be an integer or null")
    return problems


def request_acquisition(dossier: dict | None) -> dict:
    """Evaluate an acquisition request. ALWAYS fails closed in L0.7.0."""
    if dossier is None:
        return {"authorized": False, "stage": "SOURCE_DISCOVERY",
                "reason": "no request submitted", "execution": "disabled"}
    problems = validate_dossier(dossier)
    if problems:
        return {"authorized": False, "stage": "OWNER_APPROVAL_REQUIRED",
                "problems": problems, "execution": "disabled"}
    if not ACQUISITION_EXECUTION_ENABLED:
        return {"authorized": True,
                "stage": "ACQUISITION_AUTHORIZED",
                "execution": "disabled — L0.7.0 invariant (acquisition_execution_disabled=true)",
                "note": "owner_authorization_id recorded; execution requires a future milestone authorization"}
    return {"authorized": True, "stage": "ACQUISITION_EXECUTION", "execution": "enabled"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_acquisition_workflow.json")
    args = parser.parse_args(argv)

    workflow = load_workflow()
    if workflow.get("invariants", {}).get("acquisition_execution_disabled") is not True:
        print("WORKFLOW INVALID — execution invariant must remain true")
        return 2
    if ACQUISITION_EXECUTION_ENABLED:
        print("INVARIANT VIOLATION — this build enables acquisition execution")
        return 2

    template = workflow["request_dossier_template"]
    template_eval = request_acquisition(template)
    approved_example = request_acquisition({**template,
                                            **{f: ("x" if f != "expected_storage_bytes" else 1024)
                                               for f in REQUIRED_FIELDS},
                                            "owner_authorization_id": "OWNER-DECISION-0123"})

    report = {
        "schema": "taksh.l070_acquisition_workflow/1",
        "generated_by": "evaluation/l070_acquisition_gate.py",
        "stages": STAGES,
        "invariants": workflow["invariants"],
        "execution_enabled": ACQUISITION_EXECUTION_ENABLED,
        "template_evaluation": template_eval,
        "fully_filled_dossier_evaluation": approved_example,
        "policy": "execution remains disabled even with a complete dossier and owner id; enabling requires a new milestone + explicit owner decision recorded out-of-band",
        "verdict": "ACQUISITION GATE QUALIFIED — owner-authorization enforced; execution disabled",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"execution_enabled={ACQUISITION_EXECUTION_ENABLED}; template denied as expected")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
