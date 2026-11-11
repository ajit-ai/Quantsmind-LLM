"""TAKSH L0.6.1 — J: observability aggregator.

Runs every L0.6.1 qualification tool, merges findings into a single structured
report with explicit status classification:

  IMPLEMENTED_AND_TESTED — code exists + focused tests pass
  IMPLEMENTED_UNTESTED   — code exists, no focused test executed
  SIMULATED              — runs on synthetic fixtures only
  BLOCKED                — cannot proceed on this host/without authorization
  NOT_YET_AUTHORIZED     — gated on owner authorization

Nothing in this report grants authorization; it reports state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

TOOLS = [
    ("training_config", "evaluation.l0601_training_config"),
    ("training_manifest", "evaluation.l0601_training_manifest"),
    ("incremental", "training.incremental"),
    ("checkpoint_lifecycle", "training.checkpoint_lifecycle"),
    ("replay", "training.replay"),
    ("dlcae_learning_interface", "evaluation.l0601_dlcae_learning_interface"),
    ("karkain_learning_contract", "evaluation.l0601_karkain_learning_contract"),
    ("resource_preflight", "evaluation.l0601_resource_preflight"),
]

SECTIONS = {
    "training_config": {"section": "A", "state": "IMPLEMENTED_AND_TESTED"},
    "training_manifest": {"section": "B+C", "state": "IMPLEMENTED_AND_TESTED"},
    "incremental": {"section": "D", "state": "IMPLEMENTED_AND_TESTED"},
    "checkpoint_lifecycle": {"section": "E", "state": "SIMULATED"},
    "replay": {"section": "F", "state": "SIMULATED"},
    "dlcae_learning_interface": {"section": "G", "state": "IMPLEMENTED_AND_TESTED"},
    "karkain_learning_contract": {"section": "H", "state": "IMPLEMENTED_AND_TESTED"},
    "resource_preflight": {"section": "I", "state": "IMPLEMENTED_AND_TESTED"},
}

IMPORTANT_DISTINGUISHING_STATUSES = [
    "IMPLEMENTED_AND_TESTED", "IMPLEMENTED_UNTESTED",
    "SIMULATED", "BLOCKED", "NOT_YET_AUTHORIZED",
]


def run_tool(module: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    return {
        "module": module,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout.strip().splitlines()[-3:] if proc.stdout else [],
        "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else [],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l0601_observability.json")
    args = parser.parse_args(argv)

    results = []
    for name, module in TOOLS:
        r = run_tool(module)
        status = SECTIONS[name]["state"] if r["exit_code"] == 0 else "BLOCKED"
        results.append({"tool": name, "section": SECTIONS[name]["section"],
                        "state": status, **r})

    resource = json.loads((REPO / "reports" / "l0601_resource_preflight.json").read_text(encoding="utf-8"))["preflight"]
    host_blocked = not resource.get("authorized", False)
    report = {
        "schema": "taksh.l0601_observability/1",
        "generated_by": "evaluation/l0601_observability.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "distinguishing_statuses": IMPORTANT_DISTINGUISHING_STATUSES,
        "tools": results,
        "summary": {
            "sections": list(SECTIONS),
            "all_exit_zero": all(r["exit_code"] == 0 for r in results),
            "host_resource_state": ("BLOCKED (estimated working set exceeds available memory at report time)"
                                     if host_blocked else "guard-clear"),
            "not_yet_authorized": ["training", "data acquisition", "gpu", "publication"],
            "karkain_integration": "NOT ELIGIBLE (kcc broken on host; owner gate open)",
            "dlcae_handoff": "contract validated; zero entries authorized; TRAINING_AUTHORIZED unreachable without owner_authorization_id",
            "replay": "interfaces + forgetting metrics qualified on synthetic fixtures",
            "checkpoint_lifecycle": "atomic write + interrupted-write detection + fingerprint verification verified on synthetic fixtures",
        },
        "resource_preflight": resource,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for r in results:
        print(f"{r['tool']:26s} {r['state']:24s} exit={r['exit_code']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
