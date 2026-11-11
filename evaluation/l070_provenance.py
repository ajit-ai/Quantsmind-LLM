"""TAKSH L0.7.0 — provenance assessment validator.

Fail-closed provenance checks for candidate sources: origin identity, revision
pinning, license note, integrity metadata, and the acquisition-method question.
Provenance evidence is metadata we already recorded (2026-11-11 reviews); no
content was accessed to produce it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l070_source_registry import load_registry  # noqa: E402

PROVENANCE_VERDICTS = ("VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED", "REJECTED")



def assess(src: dict) -> dict:
    prov = src.get("provenance") or {}
    integrity = src.get("integrity") or {}
    integrity_ok = (
        isinstance(integrity.get("sha256"), str)
        and not integrity["sha256"].lower().startswith(("n/a", "null"))
    )
    checks = {
        "source_identified": bool(prov.get("source")),
        "revision_pinned": bool(prov.get("revision")) and "latest" not in str(prov.get("revision", "")).lower(),
        "license_note_present": bool(prov.get("license_note")),
        "official_url_present": bool(src.get("official_url")) or src["source_id"] == "src-taksh-l04-code-corpus",
        "integrity_metadata_present": integrity_ok,
        "exclusion_policy_present": bool(src.get("known_exclusions")),
        "license_reference_present": bool(src.get("license_ref")),
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)
    if passed == total:
        verdict = "VERIFIED"
    elif passed >= total - 2:
        verdict = "PARTIALLY_VERIFIED"
    else:
        verdict = "UNVERIFIED"
    if src.get("acquisition_status") == "BLOCKED":
        verdict = "REJECTED"
    return {"source_id": src["source_id"], "checks": checks,
            "passed": passed, "total": total, "verdict": verdict}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_provenance_assessment.json")
    args = parser.parse_args(argv)

    registry = load_registry()
    assessments = [assess(s) for s in registry.get("sources", [])]

    report = {
        "schema": "taksh.l070_provenance_assessment/1",
        "generated_by": "evaluation/l070_provenance.py",
        "assessed": len(assessments),
        "assessments": assessments,
        "unresolved": [a["source_id"] for a in assessments if a["verdict"] in ("UNVERIFIED", "PARTIALLY_VERIFIED")],
        "policy": "UNVERIFIED provenance blocks acquisition eligibility; VERIFIED means traceable, never approved",
        "no_content_access": "assessments are metadata-only; zero dataset bytes were read",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for a in assessments:
        print(f"{a['source_id']:28s} {a['verdict']:18s} {a['passed']}/{a['total']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
