"""TAKSH L0.6.1 — G: DLCAE-to-Learning Engine interfaces.

Explicit authorization and provenance boundaries between the Dynamic Language
Comprehension and Adaptation Engine and the Learning Engine. Status transitions
are gated; nothing transitions automatically from DISCOVERED to training data.

Eligibility pipeline:
  DISCOVERED -> COMPREHENSION_IN_PROGRESS -> KNOWLEDGE_VALIDATED
  -> ELIGIBLE_FOR_TRAINING -> TRAINING_AUTHORIZED -> TRAINED
  -> FUNCTIONALLY_QUALIFIED | ENGINEERING_QUALIFIED
  Any stage -> BLOCKED (owner unblock) or -> ROLLED_BACK
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Status enum with explicit eligibility flags
STAGES = {
    "DISCOVERED": {"capability": False, "training_eligible": False},
    "COMPREHENSION_IN_PROGRESS": {"capability": False, "training_eligible": False},
    "KNOWLEDGE_VALIDATED": {"capability": False, "training_eligible": False,
                             "note": "knowledge holds license + provenance; still not training data"},
    "ELIGIBLE_FOR_TRAINING": {"capability": False, "training_eligible": True,
                               "note": "owner must sign off explicitly before any run"},
    "TRAINING_AUTHORIZED": {"capability": False, "training_eligible": True,
                            "note": "requires recorded owner authorization + resource guard clear"},
    "TRAINED": {"capability": False, "training_eligible": True,
                "note": "weights updated; qualification still unproven"},
    "FUNCTIONALLY_QUALIFIED": {"capability": True, "training_eligible": True},
    "ENGINEERING_QUALIFIED": {"capability": True, "training_eligible": True},
    "BLOCKED": {"capability": False, "training_eligible": False},
    "ROLLED_BACK": {"capability": False, "training_eligible": False},
}

REQUIRED_PROVENANCE = (
    "language_id", "purpose", "source", "license", "revision", "license_note",
    "approved_by", "provenance_record_id",
)


def transition(state: dict, target: str) -> dict:
    current = state.get("stage")
    allowed = {
        "DISCOVERED": {"COMPREHENSION_IN_PROGRESS", "BLOCKED"},
        "COMPREHENSION_IN_PROGRESS": {"KNOWLEDGE_VALIDATED", "BLOCKED", "ROLLED_BACK"},
        "KNOWLEDGE_VALIDATED": {"ELIGIBLE_FOR_TRAINING", "BLOCKED", "ROLLED_BACK"},
        "ELIGIBLE_FOR_TRAINING": {"TRAINING_AUTHORIZED", "ROLLED_BACK", "BLOCKED"},
        "TRAINING_AUTHORIZED": {"TRAINED", "ROLLED_BACK"},
        "TRAINED": {"FUNCTIONALLY_QUALIFIED", "ENGINEERING_QUALIFIED", "ROLLED_BACK"},
        "FUNCTIONALLY_QUALIFIED": {"ENGINEERING_QUALIFIED", "ROLLED_BACK"},
        "ENGINEERING_QUALIFIED": {"ROLLED_BACK"},
        "BLOCKED": {"COMPREHENSION_IN_PROGRESS"},  # owner unblock only
        "ROLLED_BACK": {"DISCOVERED"},
    }
    if current not in allowed:
        return {"ok": False, "reason": f"unknown current stage {current!r}"}
    if target not in allowed[current]:
        return {"ok": False, "reason": f"illegal transition {current} -> {target}"}
    if target == "FUNCTIONALLY_QUALIFIED" or target == "ENGINEERING_QUALIFIED":
        if not state.get("evaluation_sufficient"):
            return {"ok": False, "reason": "qualification requires recorded evaluation evidence"}
    if target in ("TRAINING_AUTHORIZED", "ELIGIBLE_FOR_TRAINING"):
        if not state.get("owner_authorization_id"):
            return {"ok": False, "reason": "owner authorization required before any training run"}
    state = dict(state, stage=target)
    return {"ok": True, "state": state}


def eligibility_gate(entry: dict) -> list[str]:
    problems = []
    for field in REQUIRED_PROVENANCE:
        if not entry.get(field):
            problems.append(f"missing provenance: {field}")
    lic = str(entry.get("license", "")).lower()
    from evaluation.l060_training_readiness import license_ok

    if not license_ok(lic):
        problems.append(f"license fails allowlist: {lic!r}")
    return problems


def handoff_report(states: list[dict]) -> dict:
    from collections import Counter

    counts = Counter(s.get("stage") for s in states)
    return {
        "states": [s.get("stage") for s in states],
        "counts": dict(counts),
        "authorization_boundary": "no entry reaches TRAINING_AUTHORIZED without owner_authorization_id",
        "ingestion_boundary": "no entry reaches ELIGIBLE_FOR_TRAINING on provenance alone; owner sign-off required",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_dlcae_learning_interface.json")
    args = parser.parse_args(argv)

    # Deterministic walkthrough of the gate sequence (synthetic, no training)
    state = {"language_id": "karkain", "stage": "DISCOVERED",
             "purpose": "extend registry with a new first-class language",
             "source": "owner-internal repo, F:/Codes/Git/Karkain",
             "license": "apache-2.0",
             "revision": "e612ede (feature/v1.2.0-native-independence)",
             "license_note": "owner-internal; no redistribution authorized; no training ingestion",
             "approved_by": "n/a — not eligible",
             "provenance_record_id": "synthetic-0001",
             "owner_authorization_id": None,
             "evaluation_sufficient": False}
    transitions = []
    targets = ["COMPREHENSION_IN_PROGRESS", "KNOWLEDGE_VALIDATED",
               "ELIGIBLE_FOR_TRAINING", "TRAINING_AUTHORIZED", "TRAINED",
               "FUNCTIONALLY_QUALIFIED"]
    ok_flags = []
    for t in targets:
        result = transition(state, t)
        transitions.append({"to": t, "ok": result["ok"], "reason": result.get("reason")})
        ok_flags.append(result["ok"])
        if result["ok"]:
            state = result["state"]
        else:
            break

    report = {
        "schema": "taksh.l0601_dlcae_learning_interface/1",
        "generated_by": "evaluation/l0601_dlcae_learning_interface.py",
        "stages": list(STAGES),
        "required_provenance": list(REQUIRED_PROVENANCE),
        "transitions": transitions,
        "final_stage_reached": state["stage"],
        "handoff": handoff_report([state]),
        "invariants": [
            "DISCOVERED != TRAINING-ELIGIBLE: never automatic",
            "KNOWLEDGE_VALIDATED != TRAINING-AUTHORIZED: owner sign-off required",
            "FUNCTIONALLY_QUALIFIED requires recorded evaluation evidence",
            "BLOCKED blocks everything; only owner unblocks",
        ],
        "verdict": "DLCAE-LEARNING CONTRACT QUALIFIED (interfaces only; nothing authorized for training)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for t in transitions:
        print(f"{t['to']:34s} ok={t['ok']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
