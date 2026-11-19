"""TAKSH L0.6.2 — observability aggregate.

Runs every L0.6.2 tool, merges findings into one structured report, and
labels every number as measured or estimated. Does not authorize anything.
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
    ("inference_qualification", "evaluation.l062_inference_qualification"),
    ("dlcae_inference_interface", "evaluation.l062_dlcae_inference_interface"),
    ("karkain_inference", "evaluation.l062_karkain_inference"),
]

SECTIONS = {
    "inference_qualification": {"section": "5-11", "state": "IMPLEMENTED_AND_TESTED"},
    "dlcae_inference_interface": {"section": "12", "state": "IMPLEMENTED_AND_TESTED"},
    "karkain_inference": {"section": "13", "state": "IMPLEMENTED_AND_TESTED"},
}


def run_tool(module: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    return {
        "module": module,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout.strip().splitlines()[-4:] if proc.stdout else [],
        "stderr_tail": proc.stderr.strip().splitlines()[-4:] if proc.stderr else [],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l062_observability.json")
    args = parser.parse_args(argv)

    results = []
    for name, module in TOOLS:
        r = run_tool(module)
        state = SECTIONS[name]["state"] if r["exit_code"] == 0 else "BLOCKED"
        results.append({"tool": name, "section": SECTIONS[name]["section"],
                        "state": state, **r})

    qualification_path = REPO / "reports" / "l062_inference_qualification.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8")) if qualification_path.is_file() else {}
    designated = qualification.get("designated_checkpoint", {})
    outcome = designated.get("outcome", {})
    decision = designated.get("decision", {})

    report = {
        "schema": "taksh.l062_observability/1",
        "generated_by": "evaluation/l062_observability.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "distinguishing_statuses": [
            "IMPLEMENTED_AND_TESTED", "IMPLEMENTED_UNTESTED", "SIMULATED",
            "BLOCKED", "NOT_YET_AUTHORIZED",
        ],
        "tools": results,
        "summary": {
            "all_exit_zero": all(r["exit_code"] == 0 for r in results),
            "designated_checkpoint": {
                "decision": decision.get("decision"),
                "outcome_state": outcome.get("state"),
                "available_bytes_at_decision": decision.get("available_bytes_at_decision"),
                "required_bytes_estimate": decision.get("required_bytes_estimate"),
                "artifact": "qm-code-v160-15m (d_model 384, ~28.8M params) — NOT the canonical taksh-foundation-111m",
                "labels": "measured memory; estimated requirement (file + headroom)",
            },
            "canonical_foundation_111m": "artifact does not exist; foundation model never trained; canonical inference impossible today",
            "inference_infrastructure": "QUALIFIED on synthetic fixtures; see tests/test_l062_inference.py",
            "capability_benchmark": "NOT RUN — frozen l02 benchmark unchanged (854c7ca8…); 0/30 stands",
            "karkain": "NOT ELIGIBLE — no KCC on this host; interfaces qualified only",
            "dlcae": "inference interfaces qualified; registry never auto-upgraded",
            "external_providers": "DISABLED — local-only by construction; no network path exists in the engine",
            "not_yet_authorized": ["training", "data acquisition", "gpu", "publication", "external api calls"],
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['tool']:30s} {r['state']:24s} exit={r['exit_code']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
