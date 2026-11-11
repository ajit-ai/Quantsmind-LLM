"""TAKSH L0.7.0 — DLCAE to source-qualification handoff.

The bridge between language comprehension (DLCAE) and data-source qualification.
A source is never "associated" with a language automatically; a qualification
job spec is emitted for human/owner review, and eligibility is computed from
measured evidence only. No path promotes a language to capability.

Flow (L0.7.0 scope, steps 3-8 of 10):
  3. comprehension -> structured knowledge   (assumed upstream, recorded)
  4. knowledge provenance/validation          (recorded per job)
  5. relevant sources discovered               (registry lookup)
  6. licensing + quality reviewed             (decision + provenance modules)
  7. eligible sources associated               (PROPOSAL only; owner gates execution)
  8. Learning Engine receives only approved manifests (L0.6.1 gate)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.l070_provenance import assess  # noqa: E402
from evaluation.l070_source_registry import load_registry  # noqa: E402


def comprehension_available(language_id: str) -> bool:
    try:
        from evaluation.l062_dlcae_inference_interface import comprehension_state_lookup

        state = comprehension_state_lookup(language_id)
        return state.get("comprehension_state") == "COMPREHENSION_COMPLETE"
    except Exception:
        return False


def best_license_overall(source_id: str) -> str:
    decisions_path = REPO / "configs" / "data" / "l070_license_decisions.yaml"
    import yaml

    data = yaml.safe_load(decisions_path.read_text(encoding="utf-8"))
    for record in data.get("records", []):
        if record.get("source_id") == source_id:
            return record.get("overall_decision", "UNKNOWN")
    return "UNKNOWN"


def qualification_job(language_id: str, source_id: str, *, registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    src = next((s for s in registry["sources"] if s["source_id"] == source_id), None)
    if src is None:
        return {"language_id": language_id, "source_id": source_id,
                "ok": False, "reason": "source not found in registry"}

    covers = language_id in src.get("languages_covered", [])
    provenance = assess(src)
    license_overall = best_license_overall(source_id)
    comprehension = comprehension_available(language_id)

    gates = {
        "source_covers_language": covers,
        "provenance_verified": provenance["verdict"] == "VERIFIED",
        "license_review_complete": license_overall in ("APPROVED", "APPROVED_WITH_CONDITIONS"),
        "comprehension_complete": comprehension,
        "contamination_screen_passed": False,   # unknown until real corpus exists; fail closed
        "owner_authorized": False,              # never granted by this tool
    }
    eligible = all(gates.values())
    return {
        "language_id": language_id,
        "source_id": source_id,
        "gates": gates,
        "eligible_for_learning_manifest": eligible,
        "state": "proposable" if not eligible else "eligible (still NOT authorized)",
        "blocking_gates": [k for k, v in gates.items() if not v],
        "note": "eligibility computed from evidence; owner authorization still required; execution disabled",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l070_dlcae_sources.json")
    args = parser.parse_args(argv)

    jobs = [
        qualification_job("python", "src-taksh-l04-code-corpus"),
        qualification_job("python", "src-bigcode-stack-v2"),
        qualification_job("java", "src-bigcode-stack-v2"),
        qualification_job("karkain", "src-karkain-corpus"),
    ]

    report = {
        "schema": "taksh.l070_dlcae_sources/1",
        "generated_by": "evaluation/l070_dlcae_sources.py",
        "jobs": jobs,
        "eligible_for_manifests": [j for j in jobs if j["eligible_for_learning_manifest"]],
        "invariants": [
            "eligibility never implies authorization",
            "comprehension never implies corpus availability",
            "corpus availability never implies model learning",
            "nothing auto-promotes registry qualification status",
        ],
        "verdict": "ZERO sources ELIGIBLE today (contamination screens and owner authorization outstanding)",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for j in jobs:
        print(f"{j['language_id']:8s} <- {j['source_id']:26s} gates={sum(j['gates'].values())}/6 eligible={j['eligible_for_learning_manifest']}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
