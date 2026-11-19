"""Promote-or-reject gate for v1.2.0 Increment 10 (copy-first).

Gates (predeclared in the Increment 9 future-training table): exact
identifier-copy probes >= 10/13, supplemental functional >= 3/48 across
>= 3 behaviours, AD-1 binding loss <= 1. Also reports C3 binding/functional
direction. Prints the decision; writes nothing to any checkpoint
directory. Exit 0 = decision computed (check ``promote``); 2 = hash
verification failed first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training.experiment_artifacts import file_sha256
from training.v120_experiment10 import DEFAULT_CONFIG, decide_promotion10, load_config


def _summarize(candidate_side: dict) -> dict:
    copy_exact = int(round(candidate_side["copy_report"]["all"]["exact_copy"] * 13))
    supp_report = candidate_side["supplemental_report"]
    supp_passes = sum(1 for r in candidate_side["supplemental"] if r["functional"]["passed"])
    supp_behaviors = sum(1 for entry in candidate_side["supplemental_by_behavior"].values() if entry["functional_rate"])
    ad1 = candidate_side["ad1"]
    c3 = candidate_side["c3"]
    return {
        "copy_exact": copy_exact,
        "supplemental_passes": supp_passes,
        "supplemental_behaviors": supp_behaviors,
        "ad1_parses": sum(r["parses"] for r in ad1),
        "ad1_binding": sum(r["binding"] for r in ad1),
        "c3_parses": sum(r["parses"] for r in c3),
        "c3_binding": sum(r["binding"] for r in c3),
        "c3_functional": sum(r["functional"]["passed"] is True for r in c3),
        "c3_functional_cases": sum(r["functional"]["passed"] is not None for r in c3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Decide whether an Increment 10 candidate may be promoted")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config = load_config(args.config)
    payload = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    baseline_path = Path(payload["baseline_checkpoint"])
    candidate_path = Path(payload["candidate_checkpoint"])

    hashes_ok = True
    verification = {}
    if not baseline_path.is_file():
        hashes_ok = False
        verification["baseline"] = "missing"
    else:
        baseline_sha = file_sha256(baseline_path)
        verification["baseline_sha256"] = baseline_sha
        if baseline_sha != config["baseline_checkpoint_sha256"]:
            hashes_ok = False
            verification["baseline"] = "hash mismatch"
    if not candidate_path.is_file():
        hashes_ok = False
        verification["candidate"] = "missing"
    else:
        verification["candidate_sha256"] = file_sha256(candidate_path)
    for name in ("ad1", "c3"):
        fixture = config["frozen_fixtures"][name]
        path = Path(fixture["path"])
        if not path.is_file() or file_sha256(path) != fixture["sha256"]:
            hashes_ok = False
            verification[f"fixture_{name}"] = "hash mismatch"
        else:
            verification[f"fixture_{name}_sha256"] = fixture["sha256"]

    container = payload.get("summary", payload)
    base = _summarize(container["baseline"])
    cand = _summarize(container["candidate"])
    from training.curriculum_quality import detect_mode_collapse

    collapsed = (
        detect_mode_collapse(container["candidate"]["ad1"])["collapsed"]
        or detect_mode_collapse(container["candidate"]["c3"])["collapsed"]
    )
    decision = decide_promotion10(
        copy_exact=cand["copy_exact"],
        supplemental_passes=cand["supplemental_passes"],
        supplemental_behaviors=cand["supplemental_behaviors"],
        baseline_ad1_binding=base["ad1_binding"],
        candidate_ad1_binding=cand["ad1_binding"],
        baseline_ad1_parses=base["ad1_parses"],
        candidate_ad1_parses=cand["ad1_parses"],
        baseline_c3_functional=(base["c3_functional"] / base["c3_functional_cases"]) if base["c3_functional_cases"] else 0.0,
        candidate_c3_functional=(cand["c3_functional"] / cand["c3_functional_cases"]) if cand["c3_functional_cases"] else 0.0,
        candidate_collapsed=collapsed,
        hashes_ok=hashes_ok and not any("mismatch" in str(v) or "missing" in str(v) for v in verification.values()),
    )
    output = {
        "evaluation": str(args.evaluation),
        "verification": verification,
        "baseline": base,
        "candidate": cand,
        "decision": decision,
        "c3_note": (
            f"C3 binding {base['c3_binding']}->{cand['c3_binding']}, "
            f"functional {base['c3_functional']}/{base['c3_functional_cases']}->{cand['c3_functional']}/{cand['c3_functional_cases']}"
        ),
    }
    print(json.dumps(output, indent=2))
    if not hashes_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
