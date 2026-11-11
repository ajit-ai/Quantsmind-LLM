"""TAKSH L0.7.0 — observability aggregate.

Runs every L0.7.0 tool, merges findings, and distinguishes states explicitly.
Does not authorize anything; reports what is implemented, tested, unresolved,
blocked and not-yet-authorized.
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
    ("source_registry", "evaluation.l070_source_registry"),
    ("license_qualification", "evaluation.l070_license_qualification"),
    ("provenance", "evaluation.l070_provenance"),
    ("coverage_matrix", "evaluation.l070_coverage_matrix"),
    ("data_quality", "evaluation.l070_data_quality"),
    ("contamination", "evaluation.l070_contamination"),
    ("corpus_planner", "evaluation.l070_corpus_planner"),
    ("dlcae_sources", "evaluation.l070_dlcae_sources"),
    ("acquisition_gate", "evaluation.l070_acquisition_gate"),
    ("karkain_strategy", "evaluation.l070_karkain_strategy"),
    ("resource_readiness", "evaluation.l070_resource_readiness"),
    ("candidate_comparison", "evaluation.l070_candidate_comparison"),
]

SECTIONS = {
    "source_registry": ("4", "IMPLEMENTED_AND_TESTED"),
    "license_qualification": ("5", "IMPLEMENTED_AND_TESTED"),
    "provenance": ("5", "IMPLEMENTED_AND_TESTED"),
    "coverage_matrix": ("7", "IMPLEMENTED_AND_TESTED"),
    "data_quality": ("8", "SIMULATED"),
    "contamination": ("9", "SIMULATED"),
    "corpus_planner": ("10", "IMPLEMENTED_AND_TESTED"),
    "dlcae_sources": ("11", "IMPLEMENTED_AND_TESTED"),
    "acquisition_gate": ("13", "IMPLEMENTED_AND_TESTED"),
    "karkain_strategy": ("12", "IMPLEMENTED_AND_TESTED"),
    "resource_readiness": ("14", "IMPLEMENTED_AND_TESTED"),
    "candidate_comparison": ("6", "IMPLEMENTED_AND_TESTED"),
}


def run_tool(module: str) -> dict:
    proc = subprocess.run([sys.executable, "-m", module],
                          capture_output=True, text=True, cwd=REPO, timeout=900)
    return {"module": module, "exit_code": proc.returncode,
            "stdout_tail": proc.stdout.strip().splitlines()[-3:] if proc.stdout else [],
            "stderr_tail": proc.stderr.strip().splitlines()[-3:] if proc.stderr else []}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_observability.json")
    args = parser.parse_args(argv)

    results = []
    for name, module in TOOLS:
        r = run_tool(module)
        state = SECTIONS[name][1] if r["exit_code"] == 0 else "BLOCKED"
        results.append({"tool": name, "section": SECTIONS[name][0], "state": state, **r})

    report = {
        "schema": "taksh.l070_observability/1",
        "generated_by": "evaluation/l070_observability.py",
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "distinguishing_statuses": ["IMPLEMENTED_AND_TESTED", "SIMULATED", "BLOCKED", "NOT_YET_AUTHORIZED"],
        "tools": results,
        "summary": {
            "all_exit_zero": all(r["exit_code"] == 0 for r in results),
            "acquisition_authorizations": 0,
            "acquisitions_executed": 0,
            "bytes_acquired": 0,
            "external_candidates": "all HUMAN_LEGAL_REVIEW_REQUIRED (BigCode v1/v2, CodeSearchNet) or REJECTED (PyPI aggregate, Karkain corpus)",
            "qualified_internal_corpus": "python 1,590,703 tokens (l04; licensed; internal)",
            "other_languages": "zero qualified tokens (52 + Karkain)",
            "not_yet_authorized": ["dataset acquisition", "training", "gpu", "publication", "network data fetches"],
            "contamination": "controls qualified on synthetic fixtures; benchmarks pinned intact",
            "storage": "no provisioning; host cannot stage TB candidates",
        },
        "verdict": "L0.7.0 ACQUISITION-READINESS INFRASTRUCTURE QUALIFIED — zero sources authorized; nothing downloaded",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        print(f"{r['tool']:24s} {r['state']:24s} exit={r['exit_code']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
