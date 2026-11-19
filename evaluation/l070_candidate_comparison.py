"""TAKSH L0.7.0 — candidate source comparison (deterministic).

Compares the known candidates on the axes that matter for acquisition
readiness. No source is recommended for acquisition; the comparison exists to
make the owner's future decision legible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l070_source_registry import load_registry  # noqa: E402
from evaluation.l070_license_qualification import load_decisions  # noqa: E402
from evaluation.l070_provenance import assess  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_candidate_comparison.json")
    args = parser.parse_args(argv)

    registry = load_registry()
    decisions = {r["source_id"]: r for r in load_decisions()["records"]}

    rows = []
    for src in registry["sources"]:
        prov = assess(src)
        dec = decisions.get(src["source_id"], {})
        rows.append({
            "source_id": src["source_id"],
            "name": src["name"],
            "languages_covered": src["languages_covered"] if src["source_id"] != "src-bigcode-stack-v2" else f"{len(src['languages_covered'])} listed",
            "reported_size": src["integrity"].get("expected_size_bytes"),
            "access_friction": ("gated + ToU acceptance" if "gated" in json.dumps(src).lower()
                                else "SWH+INRIA agreement + S3" if src["source_id"] == "src-bigcode-stack-v2"
                                else "open (repo)" if src["source_id"] == "src-codesearchnet"
                                else "no lawful bulk path" if src["source_id"] == "src-pypi-permissive"
                                else "internal"),
            "license_class": src["license_id"],
            "overall_license_verdict": dec.get("overall_decision"),
            "provenance_verdict": prov["verdict"],
            "contamination_risk": "high (frozen bench 30 problems exist publicly at scale)" if src["source_id"].startswith("src-bigcode") else "medium" if src["source_id"] == "src-codesearchnet" else "unknown",
            "acquisition_status": src["acquisition_status"],
            "recommendation": "NOT ACQUISITION-READY",
        })

    report = {
        "schema": "taksh.l070_candidate_comparison/1",
        "generated_by": "evaluation/l070_candidate_comparison.py",
        "comparison": rows,
        "notes": [
            "no candidate is recommended for acquisition",
            "all external candidates require HUMAN_LEGAL_REVIEW_REQUIRED completion",
            "provenance strength: The Stack v2 (SWHID) > The Stack v1 > CodeSearchNet > PyPI aggregate",
            "the only acquisition-ready corpus today is the internal l04 Python corpus",
        ],
        "verdict": "COMPARISON COMPLETE — zero external sources qualified",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['source_id']:26s} license={r['overall_license_verdict']} provenance={r['provenance_verdict']} -> {r['recommendation']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
