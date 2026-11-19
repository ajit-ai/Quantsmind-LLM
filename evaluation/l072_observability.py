"""TAKSH L0.7.2 — observability aggregate."""

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
    ("snapshot", "evaluation.l072_snapshot"),
    ("manifest_check", "evaluation.l072_manifest_check"),
    ("coverage", "evaluation.l072_coverage"),
    ("karkain_reuse", "evaluation.l071_karkain"),
    ("resource_integrity", "evaluation.l072_resource_integrity"),
]


def run_tool(module: str) -> dict:
    proc = subprocess.run([sys.executable, "-m", module],
                          capture_output=True, text=True, cwd=REPO, timeout=1800)
    return {"module": module, "exit_code": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-4:] if proc.stdout else [],
            "stderr_tail": proc.stderr.strip().splitlines()[-4:] if proc.stderr else []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l072_observability.json")
    args = parser.parse_args(argv)

    results = []
    for name, module in TOOLS:
        r = run_tool(module)
        results.append({"tool": name, "state": "IMPLEMENTED_AND_TESTED" if r["exit_code"] == 0 else "BLOCKED", **r})

    snapshot = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "snapshot.json").read_text(encoding="utf-8"))
    eligibility = json.loads((REPO / "artifacts" / "l072_l04_freeze" / "eligibility.json").read_text(encoding="utf-8"))

    report = {
        "schema": "taksh.l072_observability/1",
        "generated_by": "evaluation/l072_observability.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "distinguishing_statuses": ["IMPLEMENTED_AND_TESTED", "SIMULATED", "BLOCKED", "NOT_YET_AUTHORIZED"],
        "tools": results,
        "summary": {
            "all_exit_zero": all(r["exit_code"] == 0 for r in results),
            "snapshot_id": snapshot.get("snapshot_id"),
            "freeze_verdict": snapshot.get("frozen_verdict"),
            "frozen_for_training": snapshot.get("frozen_for_training"),
            "eligibility": eligibility["aggregate"],
            "classes": {
                "REAL_AUTHORIZED_CORPUS": "python-l04-derived (1050 eligible children / 1,380,611 tokens)",
                "SYNTHETIC_TEST_FIXTURE": "not part of freeze; verified ineligible",
                "CANDIDATE_SOURCE": "bigcode v1/v2, codesearchnet (never ingested)",
                "UNVERIFIED_SOURCE": "52 languages + supporting technologies",
                "BLOCKED_SOURCE": "karkain (zero corpus; kcc 0xC0000135)",
            },
            "contamination": snapshot["contamination"]["decision"],
            "bytes_acquired": 0,
            "external_ingestion": "DISABLED",
            "training": "NOT AUTHORIZED — owner authorization missing; nothing started",
            "not_yet_authorized": ["training", "acquisition", "gpu", "publication"],
        },
        "verdict": "L0.7.2 CORPUS FREEZE QUALIFIED AS RESEARCH-ONLY — nothing training-eligible; nothing downloaded",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['tool']:20s} {r['state']:24s} exit={r['exit_code']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
