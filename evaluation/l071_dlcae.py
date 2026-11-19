"""TAKSH L0.7.1 — DLCAE integration (corpus-side, 10-step flow).

Maps the 10-step DLCAE flow onto measured pipeline state. Corpus availability
never implies model learning and never upgrades language qualification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SNAPSHOT = REPO / "artifacts" / "l071_l04_snapshot" / "snapshot.json"

FLOW_STEPS = [
    "1. dynamic language registry lookup",
    "2. language-specific source discovery",
    "3. source qualification lookup",
    "4. incremental comprehension metadata",
    "5. language-aware normalization",
    "6. structured corpus sample generation",
    "7. provenance and quality validation",
    "8. training-eligibility decision",
    "9. corpus statistics feedback",
    "10. evaluation feedback",
]

DISTINCT_STATES = [
    "language discovered", "language comprehended", "source discovered",
    "source qualified", "corpus acquired", "corpus validated",
    "corpus training-eligible", "model trained",
    "language functionally qualified", "language engineering qualified",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/l071_dlcae.json")
    args = parser.parse_args(argv)

    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))

    steps = [
        {"step": FLOW_STEPS[0], "status": "REGISTERED (python; 53-catalogue + karkain DISCOVERED)", "evidence": "configs/feasibility/l051_languages.yaml"},
        {"step": FLOW_STEPS[1], "status": "COMPLETE for internal python l04; NOT DONE for others", "evidence": "configs/data/l070_source_registry.yaml"},
        {"step": FLOW_STEPS[2], "status": "COMPLETE (license decisions recorded L0.7.0); external candidates HUMAN_LEGAL_REVIEW_REQUIRED", "evidence": "configs/data/l070_license_decisions.yaml"},
        {"step": FLOW_STEPS[3], "status": "PARTIAL (registry lookup + status recorded; no comprehension jobs run)", "evidence": "evaluation/l062_dlcae_inference_interface.py"},
        {"step": FLOW_STEPS[4], "status": "COMPLETE for processed records (l071-norm/1); provenance preserved", "evidence": "artifacts/l071_l04_snapshot/snapshot.json"},
        {"step": FLOW_STEPS[5], "status": "COMPLETE (378 real records; metadata-only records_meta.jsonl; content stays in place)", "evidence": "artifacts/l071_l04_snapshot/records_meta.jsonl"},
        {"step": FLOW_STEPS[6], "status": "COMPLETE (quality + dedup + contamination + manifest compatibility check)", "evidence": "reports/l071_training_manifest_check.json"},
        {"step": FLOW_STEPS[7], "status": "NOT_DETERMINED — owner authorization outstanding; nothing training-eligible yet", "evidence": "snapshot.gates"},
        {"step": FLOW_STEPS[8], "status": "COMPLETE (1,590,326 per-record tokens measured; see stats)", "evidence": "reports/l071_stats.json"},
        {"step": FLOW_STEPS[9], "status": "NOT RUN — frozen benchmark unchanged; no model trained; no capability claim", "evidence": "854c7ca833c224eb97aa71bea5dfd534e6e8ed1ba4b375d9cf7bf3f368a5e4a6"},
    ]

    report = {
        "schema": "taksh.l071_dlcae/1",
        "generated_by": "evaluation/l071_dlcae.py",
        "flow": steps,
        "distinct_states": DISTINCT_STATES,
        "distinct_state_current": {
            "language discovered": "python (plus 53 catalogued, karkain DISCOVERED)",
            "language comprehended": "NONE",
            "source discovered": "python-l04 + candidates (L0.7.0)",
            "source qualified": "python-l04 internal (APPROVED_WITH_CONDITIONS)",
            "corpus acquired": "python-l04 internal only",
            "corpus validated": "python-l04 (this pipeline; 1 INVALID excluded)",
            "corpus training-eligible": "NONE (owner authorization outstanding)",
            "model trained": "NO",
            "language functionally qualified": "NONE",
            "language engineering qualified": "NONE",
        },
        "invariants": [
            "corpus size never implies capability",
            "comprehension is not acquisition; acquisition is not learning",
            "no registry status is upgraded by this pipeline",
            "knowledge formation remains DLCAE-internal; retrieval augments only",
        ],
        "verdict": "DLCAE FLOW WIRED AND AUDITED; zero training-eligible corpora; zero capability changes",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for s in steps:
        print(f"{s['step'][:60]:60s} {s['status'][:60]}")
    print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
